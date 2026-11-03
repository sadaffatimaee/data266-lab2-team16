# data266-lab2-team16
DATA 266 Lab 2 – Pair_Programming_Team_16

## Repository layout
- `notebooks/` – executed notebooks (`part1-rag-lab-pair-16.ipynb`, `p1_eval_team16.ipynb`; Parts 2–3 to follow)
- `eval/` – golden set, human ratings, prompts, scenarios
- `traces/` – Part 3 JSON traces
- `requirements.txt` – pinned versions · `CONTRIBUTIONS.md` · `AI_USAGE.md`

## Part 1 – Multimodal RAG (Kaggle)
**Run the submission notebook:** open `notebooks/part1-rag-lab-pair-16.ipynb` on Kaggle with the competition data
*DATA 266 Fall 2026 – Lab 2 Part 1: Multimodal RAG* attached, **Accelerator: GPU T4**, **Internet: on**,
then **Run All** (about 25 min). It reads `QUESTIONS_PATH` (default `Questions.csv`), writes `submission.csv`
and runs `validate_submission.py`. Seed 266; uses one 16 GB GPU only.

**Run the evaluation (Task 1.4):** open `notebooks/p1_eval_team16.ipynb` on Kaggle with the same settings, plus
`eval/golden_set.csv` (and `eval/human_ratings.csv`) attached as a dataset, then **Run All** (about 40 min).
It prints the results table: baseline (dense only) vs hybrid vs final.

**Models (all open-source, no hosted APIs):**
- `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4, bitsandbytes) – figure descriptions and answers
- `BAAI/bge-small-en-v1.5` – text embeddings
- `BAAI/bge-reranker-base` – reranker
- ChromaDB (vector store), rank_bm25 (keyword search)

**Hosted APIs used:** none.

**Note:** Kaggle notebooks are not git checkouts, so Cell 1 prints an empty commit; the commit of tag `p1-v1.0` is listed in `links.txt`.

## Parts 2 and 3
To be added.
