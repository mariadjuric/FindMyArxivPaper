# FMAP-RAG Lab — week 1 milestone

## Goal

Refactor FMAP so the repository can evolve from an atlas/search/classification project into an astrophysics-first **citation-faithful scientific RAG lab**.

Week 1 should establish the research direction, benchmark pack, corpus plan, and data model that later retrieval and QA experiments will use.

## Week 1 deliverables

1. **Repository framing updated**
   - README explains the FMAP-RAG Lab direction
   - original FMAP atlas/classification work remains intact
   - new research track is explicit

2. **Benchmark pack created**
   - astrophysics QA benchmark directory
   - benchmark schema description
   - corpus-selection plan
   - first working question set (`questions.v0.json`)

3. **Data model frozen for next stage**
   - processed-data README describing planned paper/chunk schema
   - chunk schema document defining the retrieval evidence unit
   - explicit plan for full-text chunks and evidence tracking

4. **Research reading list created**
   - dense retrieval
   - reranking / late interaction
   - hierarchical retrieval
   - grounded / corrective generation
   - factuality evaluation

## What week 1 still deliberately does not do

- no full PDF ingestion yet
- no reranker yet
- no generation system yet
- no claim-level evaluator yet
- no benchmark scoring yet

That is intentional. The benchmark and schema come before the larger implementation work.

## Success criteria

At the end of week 1, the repo should answer these questions clearly:

- What is FMAP-RAG Lab trying to study?
- What benchmark will it be judged on?
- What evidence units will retrieval operate over?
- How should the first astrophysics paper subset be chosen?
- Which papers provide the theoretical spine for the next milestones?

## Status

This milestone is now complete in planning/scaffolding form:
- benchmark schema exists
- first 25-question draft exists
- corpus plan exists
- chunk schema exists
- reading list exists

## Immediate next milestone after week 1

Week 2 should implement:

1. a small astrophysics paper subset for the benchmark
2. full-text or section-aware extraction
3. chunk generation with metadata
4. lexical and dense retrieval baselines
5. retrieval evaluation on the draft benchmark
