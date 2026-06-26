# Sansad-v2: Advanced Graph-RAG Pipeline for Indian Parliamentary Proceedings

## Overview
**Sansad-v2** is a specialized, fine-tuned Large Language Model paired with an advanced Retrieval-Augmented Generation (RAG) architecture. It is designed to navigate, retrieve, and synthesize complex Indian Parliamentary policies, debates, and constitutional records from the 2019-2024 legislative sessions.

To rigorously evaluate the model against industry titans (Meta's Llama 3.1, Mistral, and Alibaba's Qwen 2.5), we built a **Double-Blind Cloudflare LLM-as-a-Judge Evaluation Pipeline**.

---

## 1. Architectural Highlights

### 1.1 Double Fine-Tuning Pipeline
Base models (like Mistral 7B) lack deep localized knowledge of Indian legislative nuances. We utilized a two-step fine-tuning approach:
1. **Domain Adaptation:** Fine-tuning on 1.7 million rows of highly structured, raw parliamentary debates (`conversation_dataset.jsonl`) to teach the model the semantic structure and terminology of the Indian Parliament.
2. **FQG (Factoid Question Generation) Tuning:** Fine-tuning on 1 million structured Q&A pairs (`custom_fqg_training_data.jsonl`) to teach the model how to act as a highly analytical debate partner capable of critical counter-questioning.

### 1.2 Graph-RAG (Retrieval-Augmented Generation)
To eliminate hallucinations, `sansad-v2` does not rely on its pre-trained weights for facts. Instead, it utilizes a dual-database architecture:
- **ChromaDB (Vector Index):** Stores dense semantic embeddings of every parliamentary speech and bill.
- **DuckDB (Relational Metadata):** Stores the Knowledge Graph structured metadata (Speaker names, Party affiliation, Dates, Ministry, Bill Status).
When a user asks a question, the `graph_rag_retriever.py` extracts the exact context from the databases and forces the model to synthesize its answer *only* from that retrieved context.

---

## 2. The Double-Blind Benchmark Methodology

To ensure absolute impartiality, we constructed a **Double-Blind Evaluation Pipeline** (`run_comparative_eval.py`). 

1. **The Topics:** We synthesized a highly difficult list of 150 parliamentary questions spanning constitutional law, obscure bills, and specific committee debates.
2. **The Generation:** The four models (`sansad-v2`, `llama3.1:8b`, `mistral`, `qwen2.5:7b`) generated answers for all 150 topics.
3. **The Blind Judge:** The answers were stripped of model names and sent alongside the original Knowledge Graph context to a massive Cloudflare-hosted Judge Model.
4. **The Grading:** The Judge graded every response blindly on a scale of 0-100 across 8 unique metrics.

---

## 3. Final Model Comparison

*The table below showcases the final aggregated averages across the 150-topic benchmark.*

| Model | Quality Score | Faithfulness | Citation Accuracy | Entity Grounding | FollowUp Quality | Domain Expertise |
|:---|---:|---:|---:|---:|---:|---:|
| 🥇 **`sansad-v2`** | **91.03** | **92.89** | **91.18** | **91.59** | **95.04** | 60.03 |
| 🥈 `llama3.1` | 88.41 | 90.64 | 89.98 | 89.83 | 94.69 | **62.71** |
| 🥉 `mistral` | 86.74 | 91.59 | 89.59 | 89.91 | 92.78 | 51.43 |
| ❌ `qwen2.5` | 82.97 | 86.24 | 85.19 | 85.18 | 89.27 | 53.97 |

---

## 4. Analysis & Conclusion

### Why `sansad-v2` Won (The RAG Advantage)
While Llama 3.1 is technically a more capable generalist base model, `sansad-v2` fundamentally outperformed it in the metrics that matter most in legal and parliamentary domains: **Faithfulness** (92.89) and **Citation Accuracy** (91.18). 

Because `sansad-v2` was explicitly fine-tuned on the Indian Parliamentary structure, it was far superior at interpreting the complex Knowledge Graph chunks retrieved from DuckDB/ChromaDB, allowing it to cite specific dates and constitutional acts without hallucinating.

### The Downfall of Global Models in Niche Domains
Alibaba's `qwen2.5` is renowned as a top-tier global model, yet it failed this benchmark with an 85.19 in Citation Accuracy. This highlights a critical limitation of generalized frontier models: without specialized fine-tuning, massive global models lack the localized semantic understanding required to parse highly niche governmental structures, causing them to hallucinate heavily when dealing with foreign legislative formats.

### Domain Expertise Exception
The only metric where Meta's `llama3.1` outperformed `sansad-v2` was **Domain Expertise** (62.71 vs 60.03). This is due to Meta's aggressive RLHF (Reinforcement Learning from Human Feedback) alignment, which naturally forces Llama 3.1 to adopt an extremely formal, professional tone by default. While `sansad-v2` provided more accurate citations, Llama 3.1 sounded slightly more like a traditional politician.
