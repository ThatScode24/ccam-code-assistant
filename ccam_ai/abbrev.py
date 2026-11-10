"""Query-time expansion of French medical abbreviations.

Doctors write "PTH G", "sonde JJ", "FOGD"; the CCAM says "remplacement de l'articulation
coxofémorale par prothèse totale", "endoprothèse urétérale", "endoscopie oeso-gastro-duodénale".
`expand()` keeps the original text and appends the expansion in brackets, phrased with the
CCAM's own vocabulary (remnographie = IRM, scanographie = scanner...), so both BM25 and the
dense model see words that match the official labels.

Only unambiguous abbreviations are listed. Ambiguous ones (PAC = port-à-cath or pontage
aorto-coronarien, AG, laparo...) are deliberately left out.

Keys of 1-2 characters are matched case-sensitively (so "D" = droit but not "d'"), longer
keys case-insensitively.
"""
from __future__ import annotations

import re

ABBREV: dict[str, str] = {
    # orthopédie
    "PTH": "prothèse totale de hanche, remplacement de l'articulation coxofémorale par prothèse totale",
    "PTG": "prothèse totale du genou, remplacement de l'articulation du genou par prothèse tricompartimentaire",
    "PUC": "prothèse unicompartimentaire du genou",
    "PUG": "prothèse unicompartimentaire du genou",
    "PIH": "prothèse intermédiaire de hanche, prothèse fémorale cervicocéphalique",
    "PTE": "prothèse totale d'épaule, remplacement de l'articulation scapulohumérale par prothèse totale",
    "LCA": "ligament croisé antérieur du genou",
    "LCP": "ligament croisé postérieur du genou",
    "AMO": "ablation de matériel d'ostéosynthèse",
    "ECM": "enclouage centromédullaire, ostéosynthèse par clou centromédullaire",
    # urologie
    "JJ": "endoprothèse urétérale, sonde double J",
    "double J": "endoprothèse urétérale",
    "sonde jj": "endoprothèse urétérale",
    "RTUP": "résection transurétrale d'une hypertrophie de la prostate, par urétrocystoscopie",
    "TURP": "résection transurétrale d'une hypertrophie de la prostate, par urétrocystoscopie",
    "RTUV": "résection transurétrale de tumeur de la vessie, par urétrocystoscopie",
    "RTUTV": "résection transurétrale de tumeur de la vessie, par urétrocystoscopie",
    "NLPC": "néphrolithotomie percutanée, ablation de calcul du rein par voie transcutanée",
    "URS": "urétéroscopie",
    "LEC": "lithotritie extracorporelle",
    "SAD": "sonde urétrovésicale à demeure, sondage vésical",
    # hépato-gastro
    "FOGD": "fibroscopie oeso-gastro-duodénale, endoscopie oeso-gastro-duodénale",
    "EOGD": "endoscopie oeso-gastro-duodénale",
    "OGD": "oeso-gastro-duodénoscopie",
    "CPRE": "cholangiopancréatographie rétrograde, par oeso-gastro-duodénoscopie",
    "GPE": "gastrostomie par voie transcutanée avec guidage endoscopique",
    "PEG": "gastrostomie par voie transcutanée avec guidage endoscopique",
    "SNG": "sonde nasogastrique, sonde gastrique",
    "coelio": "coelioscopie",
    "cœlio": "coelioscopie",
    # gynéco-obstétrique
    "IVG": "interruption volontaire de grossesse, évacuation d'un utérus gravide",
    "IMG": "interruption médicale de grossesse, évacuation d'un utérus gravide",
    "FCS": "fausse couche spontanée, évacuation d'un utérus gravide",
    "GEU": "grossesse extra-utérine",
    "DIU": "dispositif intra-utérin",
    "stérilet": "dispositif intra-utérin",
    "AVB": "accouchement par voie basse",
    "FIV": "fécondation in vitro",
    "HSG": "hystérosalpingographie",
    # cardio-vasculaire
    "ETT": "échocardiographie transthoracique, échographie-doppler transthoracique du coeur",
    "ETO": "échocardiographie transoesophagienne, échographie-doppler transoesophagienne du coeur",
    "ECG": "électrocardiographie",
    "holter": "électrocardiographie avec enregistrement continu pendant 24 heures",
    "MAPA": "mesure ambulatoire de la pression intraartérielle",
    "coro": "coronarographie, artériographie coronaire",
    "coronarographie": "artériographie coronaire",
    "ATL": "angioplastie transluminale, dilatation intraluminale",
    "CEE": "choc électrique externe, cardioversion",
    "PM": "stimulateur cardiaque définitif",
    "PMK": "stimulateur cardiaque définitif",
    "pacemaker": "stimulateur cardiaque définitif",
    "DAI": "défibrillateur cardiaque implantable",
    "TAVI": "pose d'une bioprothèse de la valve aortique, par voie artérielle transcutanée",
    "FAV": "fistule artérioveineuse",
    "VVC": "voie veineuse centrale, cathéter veineux central",
    "port-à-cath": "chambre implantable, pose d'un cathéter avec système diffuseur implantable souscutané",
    "portacath": "chambre implantable, pose d'un cathéter avec système diffuseur implantable souscutané",
    # imagerie / explorations
    "IRM": "remnographie",
    "TDM": "scanographie",
    "scanner": "scanographie",
    "scan": "scanographie",
    "angioscanner": "scanographie des vaisseaux",
    "TEP": "tomoscintigraphie par émission de positons",
    "PET": "tomoscintigraphie par émission de positons",
    "Rx": "radiographie",
    "radio": "radiographie",
    "écho": "échographie",
    "echo": "échographie",
    "abdo": "abdomen",
    "OPT": "radiographie panoramique dentomaxillaire",
    "DMO": "ostéodensitométrie, absorptiométrie osseuse",
    "EEG": "électroencéphalographie",
    "EMG": "électromyographie",
    "EFR": "exploration fonctionnelle respiratoire, spirométrie, pléthysmographie",
    # gestes / divers
    "PL": "ponction lombaire, ponction de liquide cérébrospinal par voie lombale",
    "BOM": "biopsie ostéomédullaire",
    "PBH": "ponction biopsie hépatique, biopsie du foie par voie transcutanée",
    "PBR": "ponction biopsie rénale, biopsie du rein par voie transcutanée",
    "IOT": "intubation orotrachéale, intubation trachéale",
    "VNI": "ventilation non invasive au masque",
    "LBA": "lavage bronchioloalvéolaire",
    "EBUS": "échoendoscopie bronchique",
    "IVT": "injection intravitréenne, injection d'agent pharmacologique dans le corps vitré",
    "IOL": "implantation de cristallin artificiel",
    "phaco": "phakoémulsification, extraction extracapsulaire du cristallin",
    "PKE": "phakoémulsification, extraction extracapsulaire du cristallin",
    "DCR": "dacryo-cysto-rhinostomie",
    # latéralité (sans effet sur le code, mais évite que "G"/"D" soient pris pour autre chose)
    "G": "gauche",
    "D": "droit",
    "OD": "oeil droit",
    "OG": "oeil gauche",
}

_WORD = r"A-Za-z0-9À-ÖØ-öø-ÿœŒ"


def _compile() -> list[tuple[re.Pattern, str, str]]:
    out = []
    # longest keys first so "sonde jj" wins over "JJ"
    for key in sorted(ABBREV, key=len, reverse=True):
        flags = 0 if len(key) <= 2 else re.IGNORECASE
        pat = re.compile(rf"(?<![{_WORD}']){re.escape(key)}(?![{_WORD}])", flags)
        out.append((pat, key, ABBREV[key]))
    return out


_PATTERNS = _compile()


def expand(query: str) -> str:
    """'PTH G voie post' -> 'PTH G voie post (prothèse totale de hanche, ...; gauche)'."""
    found, spans = [], []
    low = query.lower()
    for pat, key, exp in _PATTERNS:
        for m in pat.finditer(query):
            if any(a < m.end() and m.start() < b for a, b in spans):
                continue  # already covered by a longer key
            spans.append((m.start(), m.end()))
            first = exp.split(",")[0].lower()
            if first not in low and exp not in (e for _, e in found):
                found.append((m.start(), exp))
    found.sort()
    return f"{query} ({'; '.join(e for _, e in found)})" if found else query


# keys that are not useful as generation hints (laterality, spelling variants, plain words)
_NO_HINT = {"G", "D", "OD", "OG", "echo", "cœlio", "abdo", "scan", "portacath", "sonde jj", "PUG",
            "RTUTV", "PMK", "EOGD", "TURP", "coronarographie", "IOL"}


def _norm(s: str) -> str:
    s = s.lower().replace("œ", "oe")
    s = "".join(ch for ch in __import__("unicodedata").normalize("NFKD", s)
                if not __import__("unicodedata").combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


_HINT_PHRASES = [(k, [_norm(p) for p in v.split(",") if len(_norm(p)) >= 6])
                 for k, v in ABBREV.items() if k not in _NO_HINT]


# keys whose expansion phrases are too generic for label matching: explicit regexes instead
_HINT_REGEX = {
    "TAVI": r"bioprothese de la valve aortique.*voie arterielle transcutanee",
    "RTUP": r"resection.*prostate.*uretrocystoscopie",
    "RTUV": r"(tumeur|lesion) de la vessie.*(endoscopie|uretrocystoscopie)",
    "CPRE": r"cholangiopancreatographie retrograde"
            r"|(biliaire|choledoque|pancrea|oddi).*par oeso gastro duodenoscopie",
    "ATL": r"(dilatation intraluminale|angioplastie).*(transcutanee|transluminale)",
}


def hints_for(label: str) -> list[str]:
    """Abbreviations whose expansion appears in a CCAM label (PTH for 'coxofémorale par
    prothèse totale'): given to the generator so it uses real acronyms, not invented ones."""
    lab = _norm(label)
    out = []
    for k, phrases in _HINT_PHRASES:
        rx = _HINT_REGEX.get(k)
        if (re.search(rx, lab) if rx else any(p in lab for p in phrases)):
            out.append(k)
    return out


class Expanded:
    """Wrap any retriever so queries are abbreviation-expanded first."""

    def __init__(self, retriever):
        self.inner = retriever
        self.name = f"{retriever.name} + abbrev"

    def search(self, queries, k=10):
        return self.inner.search([expand(q) for q in queries], k=k)


if __name__ == "__main__":
    for q in ["PTH G voie postérieure pour coxarthrose", "pose de sonde JJ à droite sous AG",
              "FOGD + biopsies", "IRM cérébrale", "d'abord", "TEP-TDM", "cataracte OD phaco",
              "coelio pour appendicite"]:
        print(f"{q!r:45} -> {expand(q)!r}")
