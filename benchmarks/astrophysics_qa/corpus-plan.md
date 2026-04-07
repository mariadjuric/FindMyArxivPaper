# Astrophysics benchmark corpus plan

This file defines the first curated corpus strategy for **FMAP-RAG Lab**.

The aim is to build a corpus that is small enough for manual inspection, but rich enough to support retrieval, reranking, and citation-faithful QA experiments.

---

## Guiding principles

1. **Stay astrophysics-first**
   - preserve continuity with the existing FMAP project
   - build a corpus you can read and reason about

2. **Prefer paper clusters, not random sampling**
   - benchmark quality improves when papers are topically related but not duplicates
   - cluster-based sampling creates realistic retrieval difficulty

3. **Preserve enough domain diversity for retrieval experiments**
   - include nearby subfields so lexical and dense retrieval can differ meaningfully

4. **Keep manual annotation feasible**
   - the first benchmark should be small enough to inspect chunk-level evidence by hand

---

## Recommended first corpus size

### Phase 1 corpus
- **30 to 50 papers** total
- enough for manual evidence inspection
- enough for lexical vs dense retrieval comparisons
- enough for chunk-level relevance judgments

### Phase 2 corpus
- **75 to 150 papers**
- after the retrieval pipeline is stable

---

## Recommended first topic clusters

Build the initial corpus around 3 connected clusters.

### Cluster A — Galactic dynamics / Milky Way disequilibria
Use papers around:
- vertical phase-space spirals
- Galactic disc perturbations
- satellite-driven disequilibrium
- phase mixing / bending waves

### Cluster B — Distribution functions / equilibrium and non-equilibrium modelling
Use papers around:
- Milky Way distribution functions
- equilibrium assumptions
- Fokker–Planck / dynamical modelling
- survey-driven inference challenges

### Cluster C — Hamiltonian dynamics / action-angle methods
Use papers around:
- action-angle coordinates
- orbital structure
- Hamiltonian modelling
- dynamical simplification via canonical coordinates

These clusters are close enough to support meaningful retrieval overlap, but distinct enough to test ranking quality.

---

## Corpus metadata requirements

Each paper in the benchmark subset should eventually have:

- `paper_id`
- `arxiv_id`
- `title`
- `authors`
- `abstract`
- `category`
- `published`
- `url`
- `pdf_path` or local full-text path
- `topic_cluster`
- `included_in_benchmark` boolean

---

## Sampling strategy

### Recommended week-1 sampling method

1. Identify a seed list of papers you already know or can quickly validate.
2. Use FMAP retrieval to surface nearby papers by semantic similarity.
3. Manually prune:
   - duplicates
   - papers with poor text extraction prospects
   - papers too far outside the intended benchmark scope
4. Keep topic balance roughly even across the 3 clusters.

A sensible first target is:
- 10–15 papers in Galactic dynamics
- 10–15 papers in distribution-function / inference modelling
- 10–15 papers in Hamiltonian / action-angle methods

---

## What counts as a good benchmark paper?

Prefer papers that:
- have clear abstracts and structure
- contain mechanistic or methodological explanations, not only result tables
- are likely to produce answerable questions with explicit evidence spans
- are topically related to at least one other included paper

Avoid papers that:
- are impossible to parse cleanly
- are too broad survey articles for the first benchmark
- are too short to generate useful chunk evidence
- require a huge amount of external context just to interpret a basic claim

---

## Immediate annotation plan

Once the first 30–50 papers are selected:

1. assign stable `paper_id`s
2. extract or load full text
3. create chunk ids
4. link each benchmark question to:
   - one or more `gold_papers`
   - one or more `gold_chunks`
5. write or tighten `reference_answer`s based on the real evidence

---

## Why this corpus strategy is academically useful

This avoids two bad extremes:

- **too narrow:** trivial retrieval because all questions point to one obvious paper
- **too broad:** impossible annotation burden with weak evidence quality

A clustered medium-sized astrophysics corpus is the right first step for an honest retrieval-and-grounding benchmark.
