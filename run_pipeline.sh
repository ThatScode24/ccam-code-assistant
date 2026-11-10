#!/usr/bin/env bash
# Full pipeline after data generation: split -> train (E5 + CamemBERT) -> evaluate.
# Usage: ./run_pipeline.sh            (logs to logs_pipeline.txt)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

# wait for a running generation job, if any
# (resumable: codes already in queries.jsonl are skipped)

echo "== generate (resume / retry failed codes)"; $PY -m ccam_ai.generate --chunk 256 --max-model-len 2048
echo "== split";    $PY -m ccam_ai.split
echo "== train e5"; $PY -m ccam_ai.train --base intfloat/multilingual-e5-base --out models/e5-base-ccam
echo "== train camembert"; $PY -m ccam_ai.train --base dangvantuan/sentence-camembert-base --out models/camembert-ccam
echo "== evaluate"; $PY -m ccam_ai.evaluate
echo "== PIPELINE DONE"
