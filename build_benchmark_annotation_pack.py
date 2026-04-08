from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "processed" / "benchmark_corpus_manifest.csv"
FULLTEXT = ROOT / "data" / "processed" / "paper_section_chunks.csv"
OUT_DIR = ROOT / "benchmarks" / "astrophysics_qa"
QUESTIONS_V1 = OUT_DIR / "questions.v1.generated.json"
ANNOTATION_CSV = OUT_DIR / "questions.annotation.csv"
ANNOTATION_JSONL = OUT_DIR / "questions.annotation.jsonl"

TOPIC_TEMPLATES = {
    "galactic_dynamics": [
        ("definition", "What is {concept} in the context of Galactic dynamics?"),
        ("mechanism", "What physical mechanisms are proposed to explain {concept}?"),
        ("comparison", "How does {concept} compare with {concept2} in models of the Milky Way?"),
        ("challenge", "What are the main uncertainties when interpreting {concept}?"),
    ],
    "distribution_functions": [
        ("definition", "What is the role of {concept} in distribution-function modelling?"),
        ("method", "How is {concept} used when fitting Galactic dynamical models?"),
        ("challenge", "What observational complications affect inference of {concept}?"),
        ("comparison", "How does {concept} differ from {concept2} in Galactic modelling workflows?"),
    ],
    "action_angle_methods": [
        ("definition", "Why are {concept} useful in Hamiltonian dynamics?"),
        ("method", "How are {concept} used to describe orbital structure?"),
        ("comparison", "How does {concept} compare with {concept2} for analysing orbits?"),
        ("challenge", "What makes learning {concept} from trajectory data difficult?"),
    ],
    "scientific_retrieval": [
        ("comparison", "Why might {concept} outperform {concept2} on scientific papers?"),
        ("method", "How does {concept} improve scientific retrieval pipelines?"),
        ("challenge", "What failure modes arise when using {concept} for full-text retrieval?"),
        ("definition", "What does {concept} mean in the setting of scientific document retrieval?"),
    ],
    "scientific_qa": [
        ("definition", "What makes {concept} important for citation-grounded scientific QA?"),
        ("challenge", "Why can {concept} lead to unsupported answers in scientific QA?"),
        ("method", "How should a QA system use {concept} when citing evidence?"),
        ("comparison", "How does {concept} differ from {concept2} in scientific QA evaluation?"),
    ],
}

TOPIC_CONCEPTS = {
    "galactic_dynamics": [
        "phase mixing",
        "bending waves",
        "vertical phase-space spirals",
        "disc disequilibrium",
        "satellite perturbations",
        "orbital structure in the Milky Way",
        "non-equilibrium structure",
    ],
    "distribution_functions": [
        "distribution functions",
        "selection functions",
        "survey incompleteness",
        "uncertainty quantification",
        "Fokker-Planck modelling",
        "phase-space density",
        "likelihood-based dynamical inference",
    ],
    "action_angle_methods": [
        "action-angle coordinates",
        "orbital frequencies",
        "canonical coordinates",
        "symplectic structure",
        "integrable Hamiltonian motion",
        "trajectory-based coordinate learning",
        "Hamiltonian invariants",
    ],
    "scientific_retrieval": [
        "section-aware chunking",
        "dense retrieval",
        "lexical retrieval",
        "paper-first retrieval",
        "reranking",
        "hybrid retrieval",
        "evidence chunks",
    ],
    "scientific_qa": [
        "citation-faithful answering",
        "unsupported claims",
        "abstention",
        "evidence attribution",
        "FActScore-style evaluation",
        "gold evidence chunks",
        "benchmark reproducibility",
    ],
}


def make_question_bank(target_count: int = 100) -> list[dict]:
    rng = random.Random(42)
    questions: list[dict] = []
    qid = 1
    topics = list(TOPIC_TEMPLATES)
    while len(questions) < target_count:
        for topic in topics:
            templates = TOPIC_TEMPLATES[topic]
            concepts = TOPIC_CONCEPTS[topic]
            qtype, template = rng.choice(templates)
            concept = rng.choice(concepts)
            concept2 = rng.choice([c for c in concepts if c != concept])
            question = template.format(concept=concept, concept2=concept2)
            if question in {q['question'] for q in questions}:
                continue
            questions.append(
                {
                    "id": f"astro_q{qid:03d}",
                    "question": question,
                    "topic": topic,
                    "question_type": qtype,
                    "difficulty": rng.choice(["easy", "medium", "hard"]),
                    "keywords": [concept, concept2],
                    "requires_multi_hop": rng.choice([False, False, True]),
                    "answer_style": rng.choice(["short_paragraph", "bullet_points"]),
                    "gold_papers": [],
                    "gold_chunks_primary": [],
                    "gold_chunks_acceptable": [],
                    "notes": "Generated draft question. Review wording and evidence coverage.",
                }
            )
            qid += 1
            if len(questions) >= target_count:
                break
    return questions[:target_count]


def _make_blob(df: pd.DataFrame) -> list[str]:
    return (
        df["title"].fillna("").astype(str)
        + " "
        + df["section_title"].fillna("").astype(str)
        + " "
        + df["chunk_text"].fillna("").astype(str)
    ).tolist()


def attach_candidates(questions: list[dict], chunks: pd.DataFrame, manifest: pd.DataFrame) -> list[dict]:
    blobs = _make_blob(chunks)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(blobs)
    paper_meta = manifest.set_index("benchmark_paper_id").to_dict(orient="index")

    for q in questions:
        query = q["question"] + " " + " ".join(q.get("keywords", [])) + " " + q.get("topic", "")
        qv = vec.transform([query])
        sims = cosine_similarity(qv, X).ravel()
        top_idx = sims.argsort()[::-1][:12]
        candidate_chunks = []
        gold_papers = []
        for idx in top_idx:
            row = chunks.iloc[int(idx)]
            paper_id = str(row["paper_id"])
            if paper_id not in gold_papers:
                gold_papers.append(paper_id)
            candidate_chunks.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "paper_id": paper_id,
                    "title": str(row.get("title", "")),
                    "section_title": str(row.get("section_title", "")),
                    "score": round(float(sims[idx]), 4),
                    "chunk_preview": str(row.get("chunk_text", ""))[:500],
                }
            )
        q["gold_papers"] = gold_papers[:5]
        q["candidate_chunks"] = candidate_chunks
        q["gold_chunks_primary"] = [c["chunk_id"] for c in candidate_chunks[:2]]
        q["gold_chunks_acceptable"] = [c["chunk_id"] for c in candidate_chunks[2:5]]
        q["paper_titles"] = [paper_meta.get(pid, {}).get("title", "") for pid in q["gold_papers"]]
    return questions


def write_outputs(questions: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_V1.write_text(json.dumps(questions, indent=2), encoding="utf-8")

    with ANNOTATION_JSONL.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    fieldnames = [
        "id",
        "topic",
        "question_type",
        "difficulty",
        "question",
        "keywords",
        "gold_papers",
        "paper_titles",
        "gold_chunks_primary",
        "gold_chunks_acceptable",
        "candidate_chunk_ids",
        "candidate_sections",
        "review_status",
        "review_notes",
    ]
    with ANNOTATION_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in questions:
            writer.writerow(
                {
                    "id": q["id"],
                    "topic": q["topic"],
                    "question_type": q["question_type"],
                    "difficulty": q["difficulty"],
                    "question": q["question"],
                    "keywords": " | ".join(q.get("keywords", [])),
                    "gold_papers": " | ".join(q.get("gold_papers", [])),
                    "paper_titles": " | ".join(q.get("paper_titles", [])),
                    "gold_chunks_primary": " | ".join(q.get("gold_chunks_primary", [])),
                    "gold_chunks_acceptable": " | ".join(q.get("gold_chunks_acceptable", [])),
                    "candidate_chunk_ids": " | ".join(c["chunk_id"] for c in q.get("candidate_chunks", [])),
                    "candidate_sections": " | ".join(f"{c['chunk_id']}::{c['section_title']}" for c in q.get("candidate_chunks", [])),
                    "review_status": "TODO",
                    "review_notes": "",
                }
            )


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    chunks = pd.read_csv(FULLTEXT)
    questions = make_question_bank(target_count=100)
    questions = attach_candidates(questions, chunks, manifest)
    write_outputs(questions)
    print(f"Wrote {len(questions)} generated questions to {QUESTIONS_V1}")
    print(f"Wrote annotation CSV to {ANNOTATION_CSV}")
    print(f"Wrote annotation JSONL to {ANNOTATION_JSONL}")


if __name__ == "__main__":
    main()
