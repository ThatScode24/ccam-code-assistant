"""Generate doctor-style French descriptions for every CCAM act with a local LLM (vLLM).

Each prompt shows the target act *and its sibling acts* so the model keeps the details
that distinguish them (approach, laterality, size, number...). Output is resumable:
codes already present in queries.jsonl are skipped.

Usage:
  python -m ccam_ai.generate                      # all codes
  python -m ccam_ai.generate --limit 20 --print   # quick quality check
"""
from __future__ import annotations

import argparse
import json
import os
import re

# FlashInfer's sampler JIT-compiles CUDA kernels and needs nvcc; the torch sampler doesn't.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from .abbrev import hints_for
from .common import JARGON_FILE, QUERIES_FILE, load_codes, read_jsonl, strip_accents

DEFAULT_MODEL = "Qwen/Qwen3-14B-AWQ"

SYSTEM = (
    "Tu es un médecin français expérimenté. Tu sais comment les médecins décrivent "
    "réellement leurs actes quand ils demandent un codage : souvent en quelques mots, "
    "avec le vocabulaire et les abréviations de leur spécialité."
)

PROMPT = """Exemple, pour l'acte « cholécystectomie, par cœlioscopie » :
{{"descriptions": ["cholécystectomie coelio", "ablation de la vésicule biliaire sous coelioscopie", "chole sous coelio pour lithiase vésiculaire symptomatique", "Mme D., 54 ans, coliques hépatiques à répétition : cholécystectomie laparoscopique", "cholecystectomie par laparo", "exérèse vésicule biliaire par voie coelioscopique"]}}

Acte CCAM cible : « {label} »
Rubrique : {chapter} > {parent}
{siblings}
Écris {n} descriptions différentes qu'un médecin pourrait écrire pour désigner CET acte précis.

Styles à répartir :
- 4 notes très courtes (3 à 8 mots), comme une demande de codage rapide ;
- 3 phrases simples, style compte rendu ;
- 2 descriptions avec un bref contexte patient (âge, indication) ;
- 3 notes tapées vite : sans accents, abréviations, ou une faute de frappe.

Règles :
- Reformule : synonymes et termes usuels plutôt que le libellé officiel (exérèse → ablation, résection ; etc.).
- N'utilise une abréviation que si elle est réellement employée pour CET acte ou cet organe. N'emprunte jamais d'abréviation d'une autre spécialité.
- Garde tous les éléments qui distinguent l'acte cible des actes voisins (voie d'abord, technique, latéralité, nombre, taille, durée, localisation).
- N'invente pas de résultat, de suites opératoires ni de détail qui changerait l'acte (pas de côté gauche/droit si l'acte n'en précise pas).
- Ne mentionne jamais de code CCAM.

Réponds uniquement avec un objet JSON : {{"descriptions": ["...", "..."]}}"""

# Fixed instructions come first and the per-code part last, so vLLM's prefix cache shares
# the instructions across requests (much higher throughput on a 16 GB card).
# 2nd pass: the first pass almost never used real acronyms ("PTH", "sonde JJ") nor named the
# condition behind the act ("kyste sébacé", "naevus"), which is how doctors actually write.
JARGON_PROMPT = """Tâche : pour l'acte CCAM indiqué en bas, écrire des notes courtes (3 à 12 mots, style télégraphique) qu'un médecin de la spécialité écrirait réellement dans un dossier ou une demande de codage.

- La moitié des notes emploient les sigles et abréviations que les praticiens de cette spécialité utilisent vraiment pour cet acte, cet organe ou cet examen (par exemple, selon la spécialité : PTH, PTG, LCA, AMO, sonde JJ, RTUP, NLPC, FOGD, CPRE, IRM, TDM, ETT, ECG, IVG, DIU, PL, coelio, écho, Rx). N'utilise un sigle que s'il désigne bien cet acte ; s'il n'en existe pas, utilise les raccourcis oraux usuels (chole, appendic, écho abdo...).
- L'autre moitié nomment la maladie, la lésion ou l'indication qui motive habituellement cet acte, avec la localisation (par exemple, pour l'exérèse d'une lésion souscutanée : « exérèse kyste sébacé dos 2 cm », « ablation lipome épaule » ; pour l'exérèse d'une lésion superficielle de la peau : « exérèse naevus joue », « excision carcinome basocellulaire nez »).

Règles :
- Garde les éléments qui distinguent l'acte cible des actes voisins (voie d'abord, technique, taille, nombre, durée, localisation).
- Sigles français uniquement (pas de sigles anglais). Ne remplace jamais une technique par une autre (IRM ≠ scanner, coelioscopie ≠ laparotomie, endoscopie ≠ chirurgie ouverte).
- Un côté (G/D) est permis seulement si l'organe est pair.
- N'invente pas de résultat ni de détail qui changerait l'acte.
- Ne mentionne jamais de code CCAM.

Exemple, pour l'acte « cholécystectomie, par cœlioscopie » :
{{"descriptions": ["chole coelio", "cholécystectomie coelio pour LV symptomatique", "chole sous coelio, cholécystite aiguë lithiasique", "ablation VB coelio pour coliques hépatiques", "chole coelio pour polype vésiculaire", "cholecystectomie coelio + cholangio"]}}

Réponds uniquement avec un objet JSON : {{"descriptions": ["...", "..."]}}

---
Acte CCAM cible : « {label} »
Rubrique : {chapter} > {parent}
{siblings}{hints}
Écris {n} notes pour CET acte."""

SCHEMA = {
    "type": "object",
    "properties": {"descriptions": {"type": "array", "items": {"type": "string"}}},
    "required": ["descriptions"],
}


def build_prompt(c: dict, by_code: dict, n: int, template: str = PROMPT) -> str:
    sibs = [by_code[s]["label"] for s in c.get("siblings", []) if s in by_code]
    sib_txt = ""
    if sibs:
        sib_txt = "Actes voisins, à NE PAS confondre avec l'acte cible :\n" + "\n".join(
            f"- {s}" for s in sibs
        ) + "\n"
    hints = hints_for(c["label"])
    hint_txt = (f"Sigles / raccourcis usuels pour cet acte : {', '.join(hints)} "
                f"(emploie-les dans au moins 2 notes).\n") if hints else ""
    return template.format(
        label=c["label"], chapter=c.get("chapter") or "-", parent=c.get("parent_label") or "-",
        siblings=sib_txt, n=n, hints=hint_txt,
    )


def parse(text: str, n: int) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        items = json.loads(text[text.index("{"):]).get("descriptions", [])
    except (ValueError, AttributeError):
        # truncated / unclosed JSON (model kept listing items): salvage the complete strings
        items = [json.loads(f'"{s}"') for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text)]
        items = [s for s in items if s != "descriptions"]
    seen, out = set(), []
    for d in items:
        d = " ".join(str(d).split())
        key = re.sub(r"[^a-z0-9]", "", strip_accents(d.lower()))
        if 8 <= len(d) <= 300 and key not in seen and not re.search(r"\b[A-Z]{4}\d{3}\b", d):
            seen.add(key)
            out.append(d)
    return out[: n + 2]


def sampling_params(max_tokens: int, temperature: float):
    from vllm import SamplingParams

    kw = dict(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    try:  # newer vLLM
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(structured_outputs=StructuredOutputsParams(json=SCHEMA), **kw)
    except (ImportError, TypeError):
        pass
    try:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(guided_decoding=GuidedDecodingParams(json=SCHEMA), **kw)
    except (ImportError, TypeError):
        return SamplingParams(**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=12, help="descriptions per code")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--codes", nargs="*", help="only these codes")
    ap.add_argument("--print", action="store_true", help="print samples, don't save")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.82,
                    help="fraction of GPU memory for vLLM (leave room for the desktop)")
    ap.add_argument("--kv-fp8", action="store_true",
                    help="fp8 KV cache: ~2x more concurrent sequences on a 16 GB card")
    ap.add_argument("--mode", choices=["base", "jargon"], default="base",
                    help="jargon = 2nd pass with acronyms + named conditions (-> queries_jargon.jsonl)")
    args = ap.parse_args()
    template = JARGON_PROMPT if args.mode == "jargon" else PROMPT
    out_file = JARGON_FILE if args.mode == "jargon" else QUERIES_FILE
    max_tokens = 500 if args.mode == "jargon" else 900

    from vllm import LLM

    codes = load_codes()
    by_code = {c["code"]: c for c in codes}
    done = set()
    if out_file.exists() and not args.print:
        done = {r["code"] for r in read_jsonl(out_file)}
    todo = [c for c in codes if c["code"] not in done]
    if args.codes:
        todo = [c for c in todo if c["code"] in set(args.codes)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} codes to generate ({len(done)} already done)")
    if not todo:
        return

    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              **({'kv_cache_dtype': 'fp8'} if args.kv_fp8 else {}),
              enable_prefix_caching=True)
    sp = sampling_params(max_tokens=max_tokens, temperature=0.9)
    tok = llm.get_tokenizer()

    def chat(c):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(c, by_code, args.n, template)}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    failed = []
    for i in range(0, len(todo), args.chunk):
        batch = todo[i : i + args.chunk]
        outs = llm.generate([chat(c) for c in batch], sp)
        rows = []
        for c, o in zip(batch, outs):
            descs = parse(o.outputs[0].text, args.n)
            if len(descs) < 3:
                failed.append(c["code"])
                continue
            rows += [{"code": c["code"], "query": d} for d in descs]
            if args.print:
                print(f"\n### {c['code']}  {c['label']}")
                for d in descs:
                    print("   -", d)
        if not args.print:
            with open(out_file, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{min(i + args.chunk, len(todo))}/{len(todo)}] +{len(rows)} queries, {len(failed)} failed")
    if failed:
        print("failed codes (rerun to retry):", failed[:30])


if __name__ == "__main__":
    main()
