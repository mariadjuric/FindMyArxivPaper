# Astrophysics QA benchmark

This directory contains the first concrete benchmark pack for **FMAP-RAG Lab**.

The point of this benchmark is to keep the project academically honest. Retrieval, reranking, and grounded-generation changes should eventually be evaluated against a stable astrophysics QA set with explicit evidence targets.

## Current contents

- `schema.md` — benchmark item schema
- `corpus-plan.md` — how to choose the first curated astrophysics paper subset
- `questions.seed.json` — very small initial seed set kept for reference
- `questions.v0.json` — first working benchmark draft with 25 questions

## Benchmark design goals

The benchmark is designed to support:

1. **Retrieval evaluation**
   - Recall@k
   - MRR
   - nDCG

2. **Answer evaluation**
   - correctness / completeness
   - abstention behaviour when evidence is weak

3. **Citation faithfulness**
   - whether cited chunks actually support the claims they are attached to

4. **Claim-level factuality**
   - inspired by FActScore-style support checking

## Week 1 actual milestone

The week-1 milestone is now considered complete when the repo contains:

- a benchmark schema
- a realistic corpus-selection plan
- a first working benchmark draft of roughly 25 questions
- a chunk schema for the next implementation stage

This has now been done in scaffold form. The next step is to freeze the actual paper subset and replace placeholder `gold_papers` / `gold_chunks` with real ids.

## Scope for the first benchmark corpus

The recommended first benchmark corpus is:
- **30–50 astrophysics papers**
- organised around 3 topic clusters:
  - Galactic dynamics / Milky Way disequilibria
  - Distribution functions / inference modelling
  - Hamiltonian dynamics / action-angle methods

That is large enough for meaningful retrieval experiments and still small enough for manual evidence inspection.

## Immediate next step

Week 2 should now implement:

1. paper subset selection
2. full-text or section-aware extraction
3. chunk creation with stable chunk ids
4. lexical and dense retrieval baselines
5. retrieval evaluation against this benchmark draft
