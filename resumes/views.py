
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import json
from pathlib import Path

from scripts.resume_mapping.run_pipeline import main as run_pipeline_main
from scripts.resume_mapping.paths import OUTPUT_DIR


def match_jd_and_resumes(jd_path: str | None = None):
    """
    Read combined_matches.json and return a simple list of
    {name, score} dicts for the given JD (or the first JD if none).
    """
    combined_path = OUTPUT_DIR / "combined_matches.json"

    if not combined_path.exists():
        # Pipeline didn't produce file yet
        return []

    with combined_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return []

    # Figure out which JD block to use
    target_block = None
    jd_name = None

    if jd_path:
        jd_name = Path(jd_path).name  # e.g. JD-1.pdf

    if jd_name:
        # Try to match on jd_id / jd_name field
        for block in data:
            if (
                block.get("jd_id") == jd_name
                or block.get("jd_name") == jd_name
            ):
                target_block = block
                break

    # Fallback: just use the first JD block
    if target_block is None:
        target_block = data[0]

    resume_matches = target_block.get("matches", [])

    # Convert to the simple [{name, score}, ...] structure
    resp_len = []
    for item in resume_matches:
        resp_len.append(
            {
                "name": item.get("resume_id") or item.get("resume_name"),
                "score": item.get("combined_score"),
            }
        )

    return resp_len


def screening(request):
    context = {}

    if request.method == "POST":
        action = request.POST.get("action")

        # ----------------------------
        # OPTION 1: UPLOAD RESUME
        # ----------------------------
        if action == "upload_resume" and request.FILES.get("resume"):
            resume = request.FILES["resume"]
            fs = FileSystemStorage(location="media/resumes/")
            filename = fs.save(resume.name, resume)

            context["resume_msg"] = f"Resume '{filename}' uploaded & stored successfully!"

        # ----------------------------
        # OPTION 2: MATCH JD → GET MATCHING PROFILES
        # ----------------------------
        elif action == "match_jd" and request.FILES.get("jd"):
            jd_file = request.FILES["jd"]
            fs = FileSystemStorage(location="media/jd/")
            jd_name = fs.save(jd_file.name, jd_file)
            jd_path = f"media/jd/{jd_name}"

            # 1) Run the pipeline – this updates the JSONs under OUTPUT_DIR
            run_pipeline_main()

            # 2) Read the combined_matches.json and pull matches for this JD
            matches = match_jd_and_resumes(jd_path)

            context["matches"] = matches

    return render(request, "screening.html", context)
