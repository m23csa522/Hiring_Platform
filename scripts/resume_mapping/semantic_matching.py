import argparse
import json
import math
from typing import List, Dict, Any
import json, argparse
from pathlib import Path
from .paths import OUTPUT_DIR

import numpy as np
from openai import OpenAI

# ---------- OpenAI client ----------
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from environment


# ---------- Helpers ----------

def load_structured(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of documents.")
    return data


def build_embedding_text(doc: Dict[str, Any]) -> str:
    """
    Build a textual representation from the structured fields for embedding.
    Works for both JDs and resumes.
    """
    s = doc.get("structured") or {}
    title = s.get("title") or ""
    location = s.get("location") or ""
    experience_years = s.get("experience_years")
    skills = s.get("skills") or []
    domains = s.get("domains") or []
    education = s.get("education") or []
    last_company = s.get("last_company") or ""
    summary = s.get("summary") or ""
    projects_summary = s.get("projects_summary") or ""

    exp_str = f"{experience_years} years" if isinstance(experience_years, (int, float)) else ""

    parts = [
        f"Title: {title}",
        f"Location: {location}",
        f"Experience: {exp_str}",
        f"Skills: {', '.join(skills)}",
        f"Domains: {', '.join(domains)}",
        f"Education: {', '.join(education)}",
        f"Last company: {last_company}",
        f"Summary: {summary}",
        f"Projects: {projects_summary}",
    ]

    # Join and strip extra spaces
    text = "\n".join(p for p in parts if p.strip())
    return text.strip()


def compute_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    Call OpenAI embedding API for a list of texts.
    Returns a 2D numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    # OpenAI API supports batching; here we send all at once (fine for small demo).
    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    vectors = [np.array(e.embedding, dtype=np.float32) for e in resp.data]
    return np.vstack(vectors)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1D vectors.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------- Main semantic matching ----------

def semantic_match(
    docs: List[Dict[str, Any]],
    embedding_model: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    For each JD, find top_k resumes using embedding similarity.
    """
    # Filter only docs that have structured info
    valid_docs = [d for d in docs if d.get("structured") is not None]

    jds = [d for d in valid_docs if d.get("doc_type") == "jd"]
    resumes = [d for d in valid_docs if d.get("doc_type") == "resume"]

    if not jds or not resumes:
        print("[WARN] No JDs or no resumes with structured data found.")
        return []

    # Build texts
    jd_texts = [build_embedding_text(d) for d in jds]
    resume_texts = [build_embedding_text(d) for d in resumes]

    # Compute embeddings
    print(f"Computing embeddings for {len(jds)} JDs and {len(resumes)} resumes...")
    jd_emb = compute_embeddings(jd_texts, model=embedding_model)
    resume_emb = compute_embeddings(resume_texts, model=embedding_model)

    results = []

    for i, jd_doc in enumerate(jds):
        jd_vec = jd_emb[i]

        sims = []
        for j, res_doc in enumerate(resumes):
            res_vec = resume_emb[j]
            sim = cosine_sim(jd_vec, res_vec)
            sims.append((sim, res_doc))

        # Sort by similarity desc
        sims.sort(key=lambda x: x[0], reverse=True)

        top_matches = []
        for sim, res_doc in sims[:top_k]:
            s = res_doc.get("structured") or {}
            top_matches.append(
                {
                    "resume_id": res_doc.get("doc_id"),
                    "resume_title": s.get("title"),
                    "semantic_score": round(sim, 4),
                }
            )

        jd_struct = jd_doc.get("structured") or {}
        results.append(
            {
                "jd_id": jd_doc.get("doc_id"),
                "jd_title": jd_struct.get("title"),
                "semantic_matches": top_matches,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic JD-Resume matching using OpenAI embeddings.")
    parser.add_argument("--input",type=str, default=str(OUTPUT_DIR / "structured_docs.json"), required=True, help="Path to structured_docs.json")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "semantic_matches.json"), required=True, help="Path to output semantic_matches.json")
    parser.add_argument(
        "--embedding_model",
        default="text-embedding-3-small",
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top resumes per JD (default: 5)",
    )

    args = parser.parse_args()

    docs = load_structured(args.input)
    results = semantic_match(docs, args.embedding_model, args.top_k)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved semantic matches for {len(results)} JDs to {args.output}")


if __name__ == "__main__":
    main()
