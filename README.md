# data266-lab2-team16
DATA 266 Lab 2 – Pair_Programming_Team_16

## Repository layout
- `notebooks/` – executed notebooks (`p1_rag.ipynb`; Parts 2–3 to follow)
- `eval/` – golden set, prompts, scenarios
- `traces/` – Part 3 JSON traces
- `requirements.txt` – pinned versions · `CONTRIBUTIONS.md` · `AI_USAGE.md`

## Part 1 – Multimodal RAG (Kaggle)
**Run:** open `notebooks/p1_rag.ipynb` on Kaggle with the competition data
*DATA 266 Fall 2026 – Lab 2 Part 1: Multimodal RAG* attached, **Accelerator: GPU T4**, **Internet: on**,
then **Run All** (about 25 min). It reads `QUESTIONS_PATH` (default `Questions.csv`), writes `submission.csv`
and runs `validate_submission.py`. Seed 266; uses one 16 GB GPU only.

**Models (all open-source, no hosted APIs):**
- `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4, bitsandbytes) – figure descriptions and answers
- `BAAI/bge-small-en-v1.5` – text embeddings
- `BAAI/bge-reranker-base` – reranker
- ChromaDB (vector store), rank_bm25 (keyword search)

**Hosted APIs used:** none.

## Parts 2 and 3
To be added.
