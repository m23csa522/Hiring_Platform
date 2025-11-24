import argparse
import json, argparse
from pathlib import Path
from .paths import OUTPUT_DIR
from typing import Dict, Any, List


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_rule_matches(rule_data) -> Dict[str, Dict[str, Any]]:
    """
    Map:
      jd_id -> {
         "jd_title": ...,
         "by_resume": { resume_id -> full_rule_match_obj }
      }
    """
    jd_map = {}
    for jd in rule_data:
        jd_id = jd.get("jd_id")
        if not jd_id:
            continue
        by_resume = {}
        for m in jd.get("matches", []):
            rid = m.get("resume_id")
            if rid:
                by_resume[rid] = m
        jd_map[jd_id] = {
            "jd_title": jd.get("jd_title"),
            "by_resume": by_resume,
        }
    return jd_map


def index_semantic_matches(sem_data) -> Dict[str, Dict[str, Any]]:
    """
    Map:
      jd_id -> {
         "jd_title": ...,
         "by_resume": { resume_id -> full_semantic_match_obj }
      }
    """
    jd_map = {}
    for jd in sem_data:
        jd_id = jd.get("jd_id")
        if not jd_id:
            continue
        by_resume = {}
        for m in jd.get("semantic_matches", []):
            rid = m.get("resume_id")
            if rid:
                by_resume[rid] = m
        jd_map[jd_id] = {
            "jd_title": jd.get("jd_title"),
            "by_resume": by_resume,
        }
    return jd_map


def combine_scores(
    rule_data,
    sem_data,
    semantic_weight: float = 0.6,
    rule_weight: float = 0.4,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Combine rule-based scores and semantic scores:
      combined = semantic_weight * semantic_score + rule_weight * rule_score

    Works over the union of JD IDs and union of resume IDs per JD.
    """
    # Normalize weights just in case
    total_w = semantic_weight + rule_weight
    if total_w <= 0:
        semantic_weight = 0.6
        rule_weight = 0.4
        total_w = 1.0
    semantic_weight /= total_w
    rule_weight /= total_w

    rule_jd_map = index_rule_matches(rule_data)
    sem_jd_map = index_semantic_matches(sem_data)

    all_jd_ids = sorted(set(rule_jd_map.keys()) | set(sem_jd_map.keys()))

    combined_results = []

    for jd_id in all_jd_ids:
        rule_jd = rule_jd_map.get(jd_id)
        sem_jd = sem_jd_map.get(jd_id)

        jd_title = None
        if rule_jd and rule_jd.get("jd_title"):
            jd_title = rule_jd["jd_title"]
        elif sem_jd and sem_jd.get("jd_title"):
            jd_title = sem_jd["jd_title"]

        # Union of resume IDs from both channels
        resume_ids = set()
        if rule_jd:
            resume_ids |= set(rule_jd["by_resume"].keys())
        if sem_jd:
            resume_ids |= set(sem_jd["by_resume"].keys())

        combined_matches = []

        for rid in resume_ids:
            rule_m = rule_jd["by_resume"].get(rid) if rule_jd else None
            sem_m = sem_jd["by_resume"].get(rid) if sem_jd else None

            rule_score = float(rule_m.get("score", 0.0)) if rule_m else 0.0
            semantic_score = float(sem_m.get("semantic_score", 0.0)) if sem_m else 0.0

            combined_score = semantic_weight * semantic_score + rule_weight * rule_score

            # Prefer resume_title from rule, then semantic
            resume_title = None
            if rule_m and rule_m.get("resume_title"):
                resume_title = rule_m["resume_title"]
            elif sem_m and sem_m.get("resume_title"):
                resume_title = sem_m["resume_title"]

            out = {
                "resume_id": rid,
                "resume_title": resume_title,
                "combined_score": round(combined_score, 4),
                "semantic_score": round(semantic_score, 4),
                "rule_score": round(rule_score, 4),
            }

            # If we have rule-based details (skills, domains, experience), keep them
            if rule_m:
                for key in (
                    "matched_skills",
                    "matched_domains",
                    "skill_jaccard",
                    "domain_score",
                    "experience_score",
                    "experience_years_jd",
                    "experience_years_resume",
                    "experience_gap",
                ):
                    if key in rule_m:
                        out[key] = rule_m[key]

            combined_matches.append(out)

        # Sort by combined_score
        combined_matches.sort(key=lambda x: x["combined_score"], reverse=True)

        # Cut to top_k if specified (>0)
        if top_k > 0:
            combined_matches = combined_matches[:top_k]

        combined_results.append(
            {
                "jd_id": jd_id,
                "jd_title": jd_title,
                "matches": combined_matches,
            }
        )

    return combined_results


def main():
    parser = argparse.ArgumentParser(description="Combine rule-based and semantic JD-resume scores.")
    parser.add_argument("--rule_matches",type=str, default=str(OUTPUT_DIR / "matches.json"), required=True, help="Path to rule-based matches.json")
    parser.add_argument("--semantic_matches",type=str, default=str(OUTPUT_DIR / "semantic_matches.json"), required=True, help="Path to semantic_matches.json")
    parser.add_argument("--output",type=str, default=str(OUTPUT_DIR / "combined_matches.json"), required=True, help="Path to output combined_matches.json")
    parser.add_argument("--semantic_weight", type=float, default=0.6, help="Weight for semantic score (default 0.6)")
    parser.add_argument("--rule_weight", type=float, default=0.4, help="Weight for rule score (default 0.4)")
    parser.add_argument("--top_k", type=int, default=5, help="Top K resumes per JD (default 5)")

    args = parser.parse_args()

    rule_data = load_json(args.rule_matches)
    sem_data = load_json(args.semantic_matches)

    combined = combine_scores(
        rule_data,
        sem_data,
        semantic_weight=args.semantic_weight,
        rule_weight=args.rule_weight,
        top_k=args.top_k,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Saved combined matches for {len(combined)} JDs to {args.output}")


if __name__ == "__main__":
    main()
