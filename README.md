# data266-lab2-team16
DATA 266 Lab 2 – Pair_Programming_Team_16

## Repository layout
- `notebooks/` – executed notebooks: `part1-rag-lab-pair-16.ipynb`, `p1-eval-team16.ipynb`, `p2_finetune_team16.ipynb`
- `eval/` – golden set, human ratings, Part 2 prompts and ratings, Part 3 scenarios
- `traces/` – Part 3 JSON traces
- `requirements.txt` – pinned versions · `CONTRIBUTIONS.md` · `AI_USAGE.md`

## Part 1 – Multimodal RAG (Kaggle)
**Run the submission notebook:** open `notebooks/part1-rag-lab-pair-16.ipynb` on Kaggle with the competition data
*DATA 266 Fall 2026 – Lab 2 Part 1: Multimodal RAG* attached, **Accelerator: GPU T4**, **Internet: on**,
then **Run All** (about 25 min). It reads `QUESTIONS_PATH` (default `Questions.csv`), writes `submission.csv`
and runs `validate_submission.py`. Seed 266; uses one 16 GB GPU only.

**Run the evaluation (Task 1.4):** open `notebooks/p1-eval-team16.ipynb` on Kaggle with the same settings, plus
`eval/golden_set.csv` attached as a dataset, then **Run All** (about 40 min).
It prints the results table: baseline (dense only) vs hybrid vs final.

**Models (all open-source, no hosted APIs):**
- `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4, bitsandbytes) – figure descriptions and answers
- `BAAI/bge-small-en-v1.5` – text embeddings
- `BAAI/bge-reranker-base` – reranker
- ChromaDB (vector store), rank_bm25 (keyword search)

**Note:** Kaggle notebooks are not git checkouts, so Cell 1 prints an empty commit; the commit of each tag is listed in `links.txt`.

## Part 2 – LoRA fine-tuning of Stable Diffusion 1.5 (Kaggle)
**Run:** open `notebooks/p2_finetune_team16.ipynb` on Kaggle, **Accelerator: GPU T4**, **Internet: on**, and add a
Kaggle Secret `HF_TOKEN` (a Hugging Face write token). Then **Run All** (about 1 h 40 min). Set `SMOKE = True` in the
training cell for a 10-step test run (about 10 min).

The notebook: loads `lambdalabs/naruto-blip-captions`, removes corrupted/duplicate images, splits 90/10 (seed 266),
resizes to 512×512; trains two LoRA runs with the diffusers `train_text_to_image_lora.py` script
(rank 4 and rank 16, same data, 750 steps, lr 1e-4, batch 1 × 4 gradient accumulation, fp16); generates 2 images
(seeds 0, 1) for each of the 20 prompts in `eval/p2_prompts.csv` for the base model and both runs; computes KID,
CLIP score, LPIPS diversity, GPU-hours and a memorization check; and uploads everything to Hugging Face.

- Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Run 1 (LoRA rank 4): https://huggingface.co/sadaffatimaeee/naruto-sd15-run1-lora-r4
- Run 2 (LoRA rank 16, selected final): https://huggingface.co/sadaffatimaeee/naruto-sd15-run2-lora-r16
- Generated images (prompt, seed, model in `metadata.csv`): https://huggingface.co/datasets/sadaffatimaeee/naruto-lora-generations
- Metrics: CLIP `openai/clip-vit-base-patch16`, KID (torchmetrics, torch-fidelity Inception), LPIPS (AlexNet)

**Hosted APIs used:** none (all models run locally on the Kaggle GPU).
- `requirements.txt` (Part 1) and `requirements-p2.txt` (Part 2) – pinned versions · `CONTRIBUTIONS.md` · `AI_USAGE.md`

## Part 3 – Multimodal agent system

Notebook: `notebooks/p3-agents-team16.ipynb` (Kaggle, 1× T4, seed 266)

**Agents** (they pass one shared, typed `State`, and every step is logged as JSON):
1. **Planner** – picks the route: caption/question, style change, edit, Naruto style, ambiguous, refused or error
2. **Vision** – Qwen2.5-VL-7B (4-bit) describes the image and answers questions
3. **Prompt** – writes the generation prompt; any text inside the image is treated as untrusted
4. **Generation** – SD 1.5 img2img, InstructPix2Pix for edits, and our Part 2 LoRA (`naruto-sd15-run2-lora-r16`) for Naruto style
5. **Critic** – CLIP score + a VLM check; asks for up to 2 revisions

**App:** a Gradio UI shows every agent's output and the full trace.

**Evaluation** (scenarios and red-team cases were committed to `eval/` before running):

| | Without critique | With critique |
|---|---|---|
| Route correct | 14/15 | 14/15 |
| Task success (human) | 8/15 | 10/15 |
| Median seconds / request | 20.3 | 32.8 |
| Mean CLIP of final image | 30.95 | 33.33 |

- Critic agreement with human: 8/9
- Red-team attacks that succeeded: 0/10

**Known failures:** S03 was routed as "ambiguous" by mistake; the safety filter blacked out Naruto-style outputs (S10, S11); snow edits were weak (S07).

**Files:** traces in `traces/`, results in `eval/p3_results.csv`, human judgments in `eval/p3_judging_*.csv`. Inputs, outputs and traces are also on Hugging Face: https://huggingface.co/datasets/sadaffatimaeee/p3-agent-runs
