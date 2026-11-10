Index: 8558 CCAM codes.


### test_seen (15406 queries, 7703 codes)

| method | acc@1 | acc@3 | acc@5 | acc@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (keyword search) | 44.9% | 66.5% | 73.5% | 81.3% | 0.573 |
| BM25 (keyword search) + abbrev | 45.3% | 66.7% | 73.8% | 81.6% | 0.576 |
| intfloat/multilingual-e5-base (zero-shot) | 55.1% | 75.5% | 81.5% | 87.4% | 0.664 |
| dangvantuan/sentence-camembert-base (zero-shot) | 34.8% | 52.0% | 59.1% | 68.0% | 0.452 |
| camembert-ccam (fine-tuned) | 79.2% | 94.2% | 96.9% | 98.6% | 0.870 |
| camembert-ccam (fine-tuned) + abbrev | 79.3% | 94.2% | 96.8% | 98.6% | 0.870 |
| e5-base-ccam (fine-tuned) | 83.6% | 96.7% | 98.6% | 99.5% | 0.902 |
| e5-base-ccam (fine-tuned) + abbrev | 83.6% | 96.7% | 98.6% | 99.5% | 0.902 |
| e5-base-ccam-v2 (fine-tuned) | 84.6% | 97.0% | 98.8% | 99.6% | 0.909 |
| e5-base-ccam-v2 (fine-tuned) + abbrev | 84.6% | 97.0% | 98.8% | 99.6% | 0.909 |

### test_unseen (11732 queries, 855 codes)

| method | acc@1 | acc@3 | acc@5 | acc@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (keyword search) | 47.7% | 69.2% | 75.7% | 82.3% | 0.597 |
| BM25 (keyword search) + abbrev | 47.9% | 69.5% | 76.0% | 82.7% | 0.600 |
| intfloat/multilingual-e5-base (zero-shot) | 56.8% | 76.6% | 82.6% | 88.3% | 0.679 |
| dangvantuan/sentence-camembert-base (zero-shot) | 37.6% | 54.7% | 61.6% | 70.7% | 0.480 |
| camembert-ccam (fine-tuned) | 76.0% | 92.0% | 95.3% | 97.6% | 0.844 |
| camembert-ccam (fine-tuned) + abbrev | 76.1% | 92.0% | 95.3% | 97.6% | 0.844 |
| e5-base-ccam (fine-tuned) | 79.7% | 94.3% | 96.8% | 98.6% | 0.872 |
| e5-base-ccam (fine-tuned) + abbrev | 79.8% | 94.4% | 96.8% | 98.6% | 0.873 |
| e5-base-ccam-v2 (fine-tuned) | 79.6% | 94.6% | 97.1% | 98.9% | 0.873 |
| e5-base-ccam-v2 (fine-tuned) + abbrev | 79.5% | 94.6% | 97.1% | 98.9% | 0.873 |

### test_unseen_jargon (5105 queries, 855 codes)

| method | acc@1 | acc@3 | acc@5 | acc@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 (keyword search) | 45.4% | 66.3% | 73.0% | 79.7% | 0.573 |
| BM25 (keyword search) + abbrev | 47.0% | 67.8% | 74.6% | 81.6% | 0.588 |
| intfloat/multilingual-e5-base (zero-shot) | 50.8% | 70.4% | 77.0% | 83.3% | 0.619 |
| dangvantuan/sentence-camembert-base (zero-shot) | 32.1% | 47.8% | 54.3% | 63.0% | 0.416 |
| camembert-ccam (fine-tuned) | 68.3% | 85.9% | 90.2% | 94.1% | 0.778 |
| camembert-ccam (fine-tuned) + abbrev | 68.9% | 86.7% | 90.8% | 94.6% | 0.784 |
| e5-base-ccam (fine-tuned) | 73.0% | 89.2% | 92.8% | 96.1% | 0.818 |
| e5-base-ccam (fine-tuned) + abbrev | 73.6% | 89.6% | 93.4% | 96.4% | 0.823 |
| e5-base-ccam-v2 (fine-tuned) | 77.1% | 92.7% | 95.6% | 98.1% | 0.853 |
| e5-base-ccam-v2 (fine-tuned) + abbrev | 77.2% | 92.8% | 95.7% | 98.1% | 0.854 |
