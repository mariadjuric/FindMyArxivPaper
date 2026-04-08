# Benchmark Annotation Guide

This pack was generated to help upgrade the astrophysics QA benchmark from a tiny heuristic set into a reviewable evidence benchmark.

## Files

- `questions.v1.generated.json`
  - ~100 draft questions with topic, difficulty, keywords, proposed papers, and candidate chunks.
- `questions.annotation.csv`
  - Spreadsheet-friendly review file.
- `questions.annotation.jsonl`
  - Rich version with chunk previews.

## What to review

For each question, check these in order:

### 1. Question quality

- Is the question clear and academically useful?
- Is it too vague / too broad / too trivial?
- Does it actually match the available corpus?
- If not, rewrite or delete it.

### 2. Topic assignment

Check whether the topic is correct:
- `galactic_dynamics`
- `distribution_functions`
- `action_angle_methods`
- `scientific_retrieval`
- `scientific_qa`

### 3. Gold papers

Look at `gold_papers` and `paper_titles`.

Check:
- are these really the best supporting papers?
- are any obviously irrelevant?
- should another paper in the corpus replace one of them?

### 4. Gold chunks

This is the most important part.

Fields:
- `gold_chunks_primary`
- `gold_chunks_acceptable`
- `candidate_chunk_ids`
- `candidate_sections`

What to do:
- keep **primary** chunks that directly answer the question
- move weaker-but-valid evidence to **acceptable**
- remove chunks that are only topically related but not truly supporting
- if none of the proposed candidates are good, mark that in `review_notes`

### 5. Review status

Use `review_status` values like:
- `OK`
- `REWRITE_QUESTION`
- `FIX_GOLD_PAPERS`
- `FIX_GOLD_CHUNKS`
- `DELETE`

## Recommended annotation standard

### Primary chunk
A chunk should be `primary` if it:
- directly supports the answer
- contains the main claim / method / definition / explanation
- would be a good citation target in an answer

### Acceptable chunk
A chunk can be `acceptable` if it:
- partially supports the answer
- gives nearby context or another valid formulation
- would be acceptable evidence, but not the cleanest one

### Bad chunk
Reject a chunk if it:
- is only loosely related
- is mostly background without answering the question
- is front matter / title page / metadata / references
- is mathematically adjacent but not answer-bearing

## Suggested workflow

1. Open `questions.annotation.csv` in Sheets/Excel
2. Use `questions.annotation.jsonl` when you need chunk previews
3. For each row:
   - fix question wording if needed
   - confirm topic
   - confirm best papers
   - mark primary vs acceptable chunks
   - add review notes
4. After review, we can convert the cleaned annotation file back into the benchmark JSON used by eval

## What to tell me after you review

When you're done, I’ll want one of these:

- the edited CSV
- the edited JSONL
- or just a list of rows you changed heavily

Then I can:
- rebuild `questions.linked.json`
- update evaluation code to use `gold_chunks_primary` + `gold_chunks_acceptable`
- rerun retrieval against the curated benchmark
