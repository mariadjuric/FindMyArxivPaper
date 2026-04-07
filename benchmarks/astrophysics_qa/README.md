# Astrophysics QA benchmark (week 1 scaffold)

This directory contains the first benchmark scaffold for **FMAP-RAG Lab**.

The point of this benchmark is to stop the project from becoming a vague demo. Every retrieval, reranking, and grounded-generation change should eventually be evaluated against a stable astrophysics QA set.

## Week 1 deliverables

- Define the benchmark schema.
- Draft a first small astrophysics-first evaluation set.
- Keep the benchmark narrow enough that the evidence can be checked manually.

## Planned benchmark fields

Each item should eventually include:

- `id`: stable question id
- `question`: natural-language question
- `topic`: broad astrophysics topic
- `question_type`: e.g. mechanism, method, comparison, definition, result
- `gold_papers`: one or more relevant paper ids
- `gold_chunks`: one or more gold evidence chunk ids
- `reference_answer`: short human-written answer
- `notes`: optional annotation notes

## Intended use

This benchmark will support evaluation of:

1. **Retrieval quality**
   - Recall@k
   - MRR
   - nDCG

2. **Answer quality**
   - correctness / completeness
   - abstention behaviour when evidence is weak

3. **Citation faithfulness**
   - whether cited chunks actually support the claims they are attached to

4. **Claim-level factuality**
   - inspired by FActScore-style support checking

## Scope for the first version

Start with roughly **25 manually curated questions** over a small astrophysics corpus. Expand later to 50–100 when the pipeline is stable.
