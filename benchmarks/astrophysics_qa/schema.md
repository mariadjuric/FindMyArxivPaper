# Astrophysics QA benchmark schema

This is the concrete benchmark schema for the first **FMAP-RAG Lab** milestone.

The benchmark is designed for **retrieval-first scientific QA** over a curated astrophysics corpus. Each question should be answerable from the benchmark corpus alone, and each answer should be supported by explicit evidence chunks.

---

## Benchmark item schema

Each benchmark item is a JSON object with the following fields.

### Required fields

- `id` — stable unique identifier, e.g. `astro_q001`
- `question` — natural-language question
- `topic` — broad astrophysics topic, e.g. `galactic_dynamics`
- `question_type` — one of:
  - `definition`
  - `mechanism`
  - `comparison`
  - `challenge`
  - `method`
  - `result`
  - `interpretation`
- `difficulty` — one of:
  - `easy`
  - `medium`
  - `hard`
- `gold_papers` — ordered list of relevant paper ids
- `gold_chunks` — ordered list of gold evidence chunk ids
- `reference_answer` — short human-written answer grounded in the intended evidence

### Optional but recommended fields

- `keywords` — terms expected to be useful for lexical retrieval analysis
- `notes` — annotation notes for benchmark maintenance
- `requires_multi_hop` — boolean flag for questions requiring evidence from multiple chunks or papers
- `answer_style` — e.g. `short_paragraph`, `bullet_points`

---

## Example

```json
{
  "id": "astro_q001",
  "question": "What physical mechanisms are commonly proposed to explain vertical phase-space spirals in the Galactic disc?",
  "topic": "galactic_dynamics",
  "question_type": "mechanism",
  "difficulty": "medium",
  "gold_papers": ["paper_placeholder_001"],
  "gold_chunks": ["chunk_placeholder_001"],
  "reference_answer": "Expected answer should mention disequilibrium responses in the Galactic disc such as perturbations from satellites, bending waves, or related phase-mixing dynamics, depending on the chosen benchmark corpus.",
  "keywords": ["vertical phase-space spirals", "Galactic disc", "bending waves", "satellite perturbation"],
  "requires_multi_hop": false,
  "answer_style": "short_paragraph",
  "notes": "Replace placeholder ids once the benchmark corpus is fixed."
}
```

---

## Evaluation targets supported by this schema

### 1. Retrieval

A query is relevant if the retriever returns one or more items from `gold_chunks` or `gold_papers`, depending on the evaluation level.

This supports:
- Recall@k
- MRR
- nDCG

### 2. Answer quality

System outputs can be compared against `reference_answer` with:
- manual judging
- LLM-assisted judging
- rubric-based correctness scoring

### 3. Citation faithfulness

A generated answer is citation-faithful if the claims it makes are actually supported by the retrieved and cited `gold_chunks` or by valid alternative supporting chunks.

---

## Annotation policy for week 1

During week 1 and early week 2:
- `gold_papers` and `gold_chunks` may be empty placeholders while the paper subset is being fixed
- `reference_answer` should still be specific enough to constrain the expected answer
- `keywords` should be added now, since they help later lexical baseline analysis

Once the benchmark corpus is frozen, replace placeholders with stable benchmark paper and chunk ids.
