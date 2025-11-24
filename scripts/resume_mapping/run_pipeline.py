# # run_pipeline.py
# import subprocess
# import sys

# from .paths import RESUME_MAPPING_DIR, OUTPUT_DIR


# def run_step(cmd, description: str):
#     """Run a subprocess step and stop the pipeline if it fails."""
#     print("\n" + "=" * 80)
#     print(f"▶ {description}")
#     print("   Command:", " ".join(str(c) for c in cmd))
#     print("=" * 80)

#     # Run from the resume_mapping folder
#     result = subprocess.run(cmd, cwd=RESUME_MAPPING_DIR, text=True)

#     if result.returncode != 0:
#         print(f"\n✖ Step FAILED: {description}")
#         print(f"Exit code: {result.returncode}")
#         sys.exit(result.returncode)

#     print(f"\n✔ Finished: {description}")


# def main():
#     # You can tweak these defaults as you like
#     model_name = "gpt-5.1"
#     top_k = "5"
#     semantic_weight = "0.6"
#     rule_weight = "0.4"

#     python_exe = sys.executable

#     # 1) Stage A – preprocessing
#     run_step(
#         [python_exe, str(RESUME_MAPPING_DIR / "preprocessing.py")],
#         "Stage A1: Preprocess PDFs → cleaned_docs.json",
#     )

#     # 2) Stage A – LLM structuring
#     run_step(
#         [
#             python_exe,
#             str(RESUME_MAPPING_DIR / "llm_structuring.py"),
#             "--input",
#             str(OUTPUT_DIR / "cleaned_docs.json"),
#             "--output",
#             str(OUTPUT_DIR / "structured_docs.json"),
#             "--model",
#             model_name,
#         ],
#         "Stage A2: LLM structuring → structured_docs.json",
#     )

#     # 3) Stage B – rule-based matching
#     run_step(
#         [
#             python_exe,
#             str(RESUME_MAPPING_DIR / "matching.py"),
#             "--input",
#             str(OUTPUT_DIR / "structured_docs.json"),
#             "--output",
#             str(OUTPUT_DIR / "matches.json"),
#             "--top_k",
#             top_k,
#         ],
#         "Stage B1: Rule-based matching → matches.json",
#     )

#     # 4) Stage B – semantic matching
#     run_step(
#         [
#             python_exe,
#             str(RESUME_MAPPING_DIR / "semantic_matching.py"),
#             "--input",
#             str(OUTPUT_DIR / "structured_docs.json"),
#             "--output",
#             str(OUTPUT_DIR / "semantic_matches.json"),
#             "--top_k",
#             top_k,
#         ],
#         "Stage B2: Semantic (embedding) matching → semantic_matches.json",
#     )

#     # 5) Stage B – combined scoring
#     run_step(
#         [
#             python_exe,
#             str(RESUME_MAPPING_DIR / "combine_matching.py"),
#             "--rule_matches",
#             str(OUTPUT_DIR / "matches.json"),
#             "--semantic_matches",
#             str(OUTPUT_DIR / "semantic_matches.json"),
#             "--output",
#             str(OUTPUT_DIR / "combined_matches.json"),
#             "--semantic_weight",
#             semantic_weight,
#             "--rule_weight",
#             rule_weight,
#             "--top_k",
#             top_k,
#         ],
#         "Stage B3: Combine rule + semantic scores → combined_matches.json",
#     )

#     print("\n🎉 Pipeline finished successfully!")
#     print("Outputs of interest (all inside):")
#     print(f"  {OUTPUT_DIR}")
#     print("  - cleaned_docs.json")
#     print("  - structured_docs.json")
#     print("  - matches.json")
#     print("  - semantic_matches.json")
#     print("  - combined_matches.json")


# if __name__ == "__main__":
#     main()

# run_pipeline.py
import subprocess
import sys

from .paths import RESUME_MAPPING_DIR, OUTPUT_DIR, PROJECT_DIR


def run_step(module_name: str, description: str, extra_args=None):
    """Run a subprocess step (as a module) and stop the pipeline if it fails."""
    if extra_args is None:
        extra_args = []

    print("\n" + "=" * 80)
    print(f"▶ {description}")
    cmd = [sys.executable, "-m", module_name, *map(str, extra_args)]
    print("   Command:", " ".join(cmd))
    print("=" * 80)

    # IMPORTANT: run from project root so `scripts` is importable
    result = subprocess.run(cmd, cwd=PROJECT_DIR, text=True)

    if result.returncode != 0:
        print(f"\n✖ Step FAILED: {description}")
        print(f"Exit code: {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✔ Finished: {description}")


def main():
    # You can tweak these defaults as you like
    model_name = "gpt-5.1"
    top_k = "5"
    semantic_weight = "0.6"
    rule_weight = "0.4"

    print(f"PROJECT_DIR = {PROJECT_DIR}")
    print(f"RESUME_MAPPING_DIR = {RESUME_MAPPING_DIR}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")

    # 1) Stage A – preprocessing
    run_step(
        "scripts.resume_mapping.preprocessing",
        "Stage A1: Preprocess PDFs → cleaned_docs.json",
    )

    # 2) Stage A – LLM structuring
    run_step(
        "scripts.resume_mapping.llm_structuring",
        "Stage A2: LLM structuring → structured_docs.json",
        [
            "--input",
            OUTPUT_DIR / "cleaned_docs.json",
            "--output",
            OUTPUT_DIR / "structured_docs.json",
            "--model",
            model_name,
        ],
    )

    # 3) Stage B – rule-based matching
    run_step(
        "scripts.resume_mapping.matching",
        "Stage B1: Rule-based matching → matches.json",
        [
            "--input",
            OUTPUT_DIR / "structured_docs.json",
            "--output",
            OUTPUT_DIR / "matches.json",
            "--top_k",
            top_k,
        ],
    )

    # 4) Stage B – semantic matching
    run_step(
        "scripts.resume_mapping.semantic_matching",
        "Stage B2: Semantic (embedding) matching → semantic_matches.json",
        [
            "--input",
            OUTPUT_DIR / "structured_docs.json",
            "--output",
            OUTPUT_DIR / "semantic_matches.json",
            "--top_k",
            top_k,
        ],
    )

    # 5) Stage B – combined scoring
    run_step(
        "scripts.resume_mapping.combine_matching",
        "Stage B3: Combine rule + semantic scores → combined_matches.json",
        [
            "--rule_matches",
            OUTPUT_DIR / "matches.json",
            "--semantic_matches",
            OUTPUT_DIR / "semantic_matches.json",
            "--output",
            OUTPUT_DIR / "combined_matches.json",
            "--semantic_weight",
            semantic_weight,
            "--rule_weight",
            rule_weight,
            "--top_k",
            top_k,
        ],
    )

    print("\n🎉 Pipeline finished successfully!")
    print("Outputs of interest (all inside):")
    print(f"  {OUTPUT_DIR}")
    print("  - cleaned_docs.json")
    print("  - structured_docs.json")
    print("  - matches.json")
    print("  - semantic_matches.json")
    print("  - combined_matches.json")


if __name__ == "__main__":
    main()
