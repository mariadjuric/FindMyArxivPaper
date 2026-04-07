# FMAP-RAG Lab reading list

This reading list is for the new astrophysics-first **FMAP-RAG Lab** direction: citation-faithful question answering over scientific papers.

The goal is not to blindly reproduce every paper end-to-end. The goal is to build an incremental research ladder where each stage is grounded in a specific idea from the literature, implemented in a scoped way, and evaluated honestly.

---

## Stage 0 — scientific embedding / retrieval baselines

### 1. Sentence-BERT
- **Paper:** Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- **Link:** https://arxiv.org/abs/1908.10084
- **Why it matters here:** gives the conceptual basis for dense embedding retrieval over paper chunks.
- **What to reproduce in FMAP-RAG Lab:** normalized dense embeddings + cosine or dot-product nearest-neighbour retrieval over chunked scientific text.

### 2. SciBERT
- **Paper:** Beltagy, Lo, & Cohan (2019), *SciBERT: A Pretrained Language Model for Scientific Text*
- **Link:** https://arxiv.org/abs/1903.10676
- **Why it matters here:** domain-adapted scientific language backbone for both classification and later reranking / QA components.
- **What to reproduce in FMAP-RAG Lab:** scientific-text-aware encoder or reranker baselines.

### 3. Dense Passage Retrieval (DPR)
- **Paper:** Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering*
- **Link:** https://arxiv.org/abs/2004.04906
- **Why it matters here:** canonical dense retriever framing for first-stage retrieval.
- **What to reproduce in FMAP-RAG Lab:** query encoder + passage/chunk embeddings as a conceptual baseline, even if the first implementation uses strong off-the-shelf encoders.

---

## Stage 1 — reranking and stronger retrieval

### 4. ColBERT
- **Paper:** Khattab & Zaharia (2020), *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*
- **Link:** https://arxiv.org/abs/2004.12832
- **Why it matters here:** gives a strong academically defensible direction for moving beyond single-vector dense retrieval.
- **What to reproduce in FMAP-RAG Lab:** not full ColBERT training on day one; start with a reranking stage inspired by late interaction, or compare dense retrieval against a stronger second-stage reranker.

### 5. ColBERTv2
- **Paper:** Santhanam et al. (2022), *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*
- **Link:** https://arxiv.org/abs/2112.01488
- **Why it matters here:** more modern late-interaction retrieval reference; directly relevant to the project direction you described.
- **What to reproduce in FMAP-RAG Lab:** late-interaction-inspired retrieval experiment after the first dense baseline and cross-encoder reranker are in place.

### 6. Cross-Encoder reranking (practical bridge)
- **Representative reference:** Nogueira & Cho (2019), *Passage Re-ranking with BERT*
- **Link:** https://arxiv.org/abs/1901.04085
- **Why it matters here:** easiest academically respectable second-stage ranking baseline before attempting ColBERT-style retrieval.
- **What to reproduce in FMAP-RAG Lab:** retrieve top-N chunks with dense retrieval, rerank with a BERT/SciBERT cross-encoder, compare Recall@k / MRR / nDCG and downstream citation faithfulness.

---

## Stage 2 — hierarchical retrieval / document structure

### 7. RAPTOR
- **Paper:** Sarthi et al. (2024), *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*
- **Link:** https://arxiv.org/abs/2401.18059
- **Why it matters here:** motivates hierarchical retrieval over long scientific documents and document collections.
- **What to reproduce in FMAP-RAG Lab:** compare flat chunk retrieval against a lightweight hierarchy such as section summaries or paper-level summaries feeding chunk retrieval.

### 8. Long-context and document-structure-aware retrieval
- **Suggested supporting direction:** use section-aware chunking and metadata preservation even before a full RAPTOR-style hierarchy.
- **Why it matters here:** scientific papers have strong rhetorical structure; section headers and local discourse matter.
- **What to reproduce in FMAP-RAG Lab:** section-aware chunking baseline vs naive fixed windows.

---

## Stage 3 — grounded / corrective generation

### 9. Self-RAG
- **Paper:** Asai et al. (2023), *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- **Link:** https://arxiv.org/abs/2310.11511
- **Why it matters here:** provides a principled reference for self-reflective retrieval-augmented generation.
- **What to reproduce in FMAP-RAG Lab:** not full training initially; instead implement a lightweight reflective/corrective loop that checks whether claims are supported by retrieved evidence and revises or abstains.

### 10. CRAG
- **Paper:** Yan et al. (2024), *Corrective Retrieval Augmented Generation*
- **Link:** https://arxiv.org/abs/2401.15884
- **Why it matters here:** useful reference for retrieval correction and evidence-quality filtering.
- **What to reproduce in FMAP-RAG Lab:** evidence quality checks, fallback retrieval, or answer abstention when support is weak.

---

## Stage 4 — factuality and citation evaluation

### 11. FActScore
- **Paper:** Min et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
- **Link:** https://arxiv.org/abs/2305.14251
- **Why it matters here:** directly relevant to evaluating claim-level support in scientific RAG outputs.
- **What to reproduce in FMAP-RAG Lab:** claim decomposition + support checking over retrieved evidence, reported as a supported-claim rate.

### 12. Retrieval-augmented generation for knowledge-intensive NLP
- **Paper:** Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **Link:** https://arxiv.org/abs/2005.11401
- **Why it matters here:** classic RAG reference for grounding generation on retrieved evidence.
- **What to reproduce in FMAP-RAG Lab:** answer generation conditioned on retrieved chunks, but with stricter citation requirements than generic RAG demos.

---

## Recommended incremental reproduction plan

### Week 1–2: benchmark and retrieval foundations
- Reproduce the **dense retrieval baseline** idea from SBERT / DPR.
- Preserve scientific metadata and chunk structure.
- Evaluate retrieval on a hand-curated astrophysics benchmark.

### Week 2–3: stronger retrieval
- Add a **cross-encoder reranker**.
- Compare dense-only vs dense+rerank.
- Report Recall@k, MRR, nDCG.

### Week 3–4: document structure
- Compare **fixed chunking** vs **section-aware chunking**.
- Optionally add paper-level or section-level summaries inspired by RAPTOR.

### Week 4+: grounded generation
- Add answer generation constrained to retrieved evidence.
- Require inline citations.
- Measure citation precision and claim support.

### Later experiments
- ColBERT-style late interaction.
- Reflective/corrective generation inspired by Self-RAG / CRAG.
- FActScore-style atomic factuality evaluation.

---

## Theoretical spine for the write-up

These are the mathematical objects the project should foreground:

1. **Embedding-based retrieval**
   - query embedding \(q \in \mathbb{R}^d\)
   - chunk embeddings \(c_i \in \mathbb{R}^d\)
   - score \(s_i = q^\top c_i\) for normalized embeddings

2. **Ranking metrics**
   - Recall@k
   - MRR
   - nDCG

3. **Grounded generation**
   - model answer distribution \(p(a \mid q, D_k)\), where \(D_k\) is the retrieved evidence set

4. **Claim support**
   - decompose answers into atomic claims
   - compute supported-claim rate over cited evidence

This framing is enough to make the project feel honestly academic and evaluable, without pretending to reproduce frontier systems in full from day one.
