import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import json, argparse, math
from pathlib import Path
from .paths import OUTPUT_DIR
# -----------------------------
# Helpers
# -----------------------------

def safe_structured(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return the 'structured' block or an empty dict."""
    return doc.get("structured") or {}


def to_skill_set(struct: Dict[str, Any]) -> set:
    skills = struct.get("skills") or []
    if not isinstance(skills, list):
        return set()
    return {str(s).strip().lower() for s in skills if str(s).strip()}


def to_domain_set(struct: Dict[str, Any]) -> set:
    domains = struct.get("domains") or []
    if not isinstance(domains, list):
        return set()
    return {str(d).strip().lower() for d in domains if str(d).strip()}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def experience_score(jd_years: Optional[float], cv_years: Optional[float]) -> float:
    """
    Simple experience alignment:
    - 1.0 if exactly equal
    - linearly down to 0 at a gap of 10+ years
    - 0 if any side is missing
    """
    if jd_years is None or cv_years is None:
        return 0.0
    gap = abs(jd_years - cv_years)
    if gap >= 10:
        return 0.0
    return max(0.0, 1.0 - gap / 10.0)


def domain_score(jd_domains: set, cv_domains: set) -> float:
    if not jd_domains or not cv_domains:
        return 0.0
    return 1.0 if (jd_domains & cv_domains) else 0.0


def compute_match_score(
    jd_struct: Dict[str, Any],
    cv_struct: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a composite score and a small explanation payload.
    Score components:
    - 0.6 * skill Jaccard
    - 0.2 * domain overlap
    - 0.2 * experience alignment
    """
    jd_skills = to_skill_set(jd_struct)
    cv_skills = to_skill_set(cv_struct)
    jd_domains = to_domain_set(jd_struct)
    cv_domains = to_domain_set(cv_struct)

    skill_j = jaccard(jd_skills, cv_skills)
    dom_s = domain_score(jd_domains, cv_domains)

    jd_exp = jd_struct.get("experience_years")
    cv_exp = cv_struct.get("experience_years")
    exp_s = experience_score(jd_exp, cv_exp)

    total = 0.6 * skill_j + 0.2 * dom_s + 0.2 * exp_s

    explanation = {
        "matched_skills": sorted(jd_skills & cv_skills),
        "matched_domains": sorted(jd_domains & cv_domains),
        "skill_jaccard": skill_j,
        "domain_score": dom_s,
        "experience_score": exp_s,
        "experience_gap": None if jd_exp is None or cv_exp is None else abs(jd_exp - cv_exp),
    }

    return total, explanation


# -----------------------------
# Main matching logic
# -----------------------------

def run_matching(
    input_path: str,
    output_path: str,
    top_k: int = 5
) -> None:
    # Load structured docs
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Separate JDs and resumes
    jds: List[Dict[str, Any]] = [d for d in docs if d.get("doc_type") == "jd"]
    cvs: List[Dict[str, Any]] = [d for d in docs if d.get("doc_type") == "resume"]

    print(f"Loaded {len(docs)} documents: {len(jds)} JDs, {len(cvs)} resumes.")

    results: List[Dict[str, Any]] = []

    for jd in jds:
        jd_struct = safe_structured(jd)
        jd_title = jd_struct.get("title")
        jd_id = jd.get("doc_id")

        jd_matches: List[Dict[str, Any]] = []

        for cv in cvs:
            cv_struct = safe_structured(cv)
            cv_title = cv_struct.get("title")
            cv_id = cv.get("doc_id")

            score, expl = compute_match_score(jd_struct, cv_struct)

            jd_matches.append({
                "resume_id": cv_id,
                "resume_title": cv_title,
                "score": round(score, 4),
                "matched_skills": expl["matched_skills"],
                "matched_domains": expl["matched_domains"],
                "skill_jaccard": round(expl["skill_jaccard"], 4),
                "domain_score": expl["domain_score"],
                "experience_score": round(expl["experience_score"], 4),
                "experience_years_jd": jd_struct.get("experience_years"),
                "experience_years_resume": cv_struct.get("experience_years"),
                "experience_gap": expl["experience_gap"],
            })

        # Sort resumes by score (desc) and keep top_k
        jd_matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = jd_matches[:top_k]

        results.append({
            "jd_id": jd_id,
            "jd_title": jd_title,
            "matches": top_matches,
        })

    # Save to output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote matches for {len(jds)} JDs to {output_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Simple JD–Resume matching based on Stage A structured docs.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        default=str(OUTPUT_DIR / "structured_docs.json"),
        help="Path to structured_docs.json from Stage A"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        default=str(OUTPUT_DIR / "matches.json"),
        help="Path to write matches.json"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top resumes per JD"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_matching(args.input, args.output, args.top_k)
