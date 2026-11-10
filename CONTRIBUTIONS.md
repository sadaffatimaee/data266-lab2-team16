# Contributions – Pair_Programming_Team_16

| Member | GitHub |
|---|---|
| Sadaf Fatima Syeda | sadaffatimaee |
| Poushali Deb Purkayastha | Poushali0202 |

## Part 1 – Multimodal RAG
| Task | Sadaf | Poushali |
|---|---|---|
| 1.1 Ingestion & indexing | Built and ran the ingestion notebook (text chunks, figure/table crops, VLM descriptions, ChromaDB) | Reran the full pipeline in `p1_ablation_safety_poushali.ipynb` |
| 1.2 Retrieval & answers | Hybrid retrieval, reranker, answer generation; Kaggle runs and submissions | Reranker ablation row (hybrid + reranker without cross-references) for the bonus |
| 1.3 Conversation handling | Conversation memory and user-value handling | Reviewed the memory and override rules during the knowledge-transfer session |
| 1.4 Evaluation & safety | Golden set, evaluation notebook, groundedness ratings | Appendix C safety probes and analysis, groundedness ratings, human-evaluation notebook |

## Part 2 – Diffusion fine-tuning
| Task | Sadaf | Poushali |
|---|---|---|
| 2.1 Data preparation | Cleaning, 90/10 split, resizing | Own cleaning and split in the baseline notebook; dataset description and limitations (`p2_analysis_poushali.ipynb`) |
| 2.2 Fine-tuning | Both LoRA runs (rank 4, rank 16), Hugging Face upload, model cards | Own LoRA training loop, rank 8 baseline run (`p2_baseline_poushali.ipynb`) |
| 2.3 Generation | 20 prompts, 120 images, grids | 40 images and grid for the baseline run; blinded pair rating page |
| 2.4 Evaluation & safety | Metrics, memorization check, image-pair ratings | Failure-mode analysis, image-pair ratings, outside rater, win-rate computation |

## Part 3 – Multi-agent system
| Task | Sadaf | Poushali |
|---|---|---|
| 3.1 Agents | Five agents of the team system (`p3-agents-team16.ipynb`) | Own five-agent baseline (`p3_baseline_poushali.ipynb`) |
| 3.2 Orchestration & app | Shared state, traces, failure handling, Gradio app | Own orchestration with a pydantic state, traces, Gradio app |
| 3.3 Evaluation & safety | 15 scenarios, 10 red-team cases, with/without critique runs, judging | Independent judging, agreement and critic-agreement computation, system card |

## Report and submission
| Task | Sadaf | Poushali |
|---|---|---|
| Report | Part sections she led, review | Report build, architecture diagrams, results tables, safety analyses |
| Submission | Tags, Kaggle final submission | Zip, links.txt |

## Meeting recordings
| Date | Topic | Link |
|---|---|---|
| | | |
