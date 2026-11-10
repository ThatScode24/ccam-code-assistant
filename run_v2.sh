#!/usr/bin/env bash
# v2: 2nd generation pass (acronyms + named conditions) -> resplit -> retrain E5 -> evaluate.
# Usage: ./run_v2.sh   (logs to logs_v2.txt)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "== generate jargon $(date +%T)"; $PY -m ccam_ai.generate --mode jargon --n 6 --chunk 512 --max-model-len 2048
echo "== split $(date +%T)";    $PY -m ccam_ai.split
echo "== train e5 v2 $(date +%T)"; $PY -m ccam_ai.train --base intfloat/multilingual-e5-base --out models/e5-base-ccam-v2
echo "== evaluate $(date +%T)"; $PY -m ccam_ai.evaluate
echo "== spotcheck $(date +%T)"; $PY -m ccam_ai.spotcheck models/e5-base-ccam models/e5-base-ccam-v2 | tee results/spotcheck.txt
echo "== V2 DONE $(date +%T)"
