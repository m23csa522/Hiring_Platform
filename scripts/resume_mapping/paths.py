# from pathlib import Path

# # This file lives in: Project/scripts/resume_mapping/paths.py
# RESUME_MAPPING_DIR = Path(__file__).resolve().parent     # .../Project/scripts/resume_mapping
# SCRIPTS_DIR = RESUME_MAPPING_DIR.parent                  # .../Project/scripts
# PROJECT_DIR = SCRIPTS_DIR.parent                         # .../Project

# DATA_DIR = PROJECT_DIR / "ai_interview_platform" / "media"
# JD_DIR = DATA_DIR / "jd"
# RESUME_DIR = DATA_DIR / "resume"

# OUTPUT_DIR = PROJECT_DIR / "output"
# OUTPUT_DIR.mkdir(exist_ok=True)

from pathlib import Path

# This file now lives in: Project/ai_interview_platform/scripts/resume_mapping/paths.py

RESUME_MAPPING_DIR = Path(__file__).resolve().parent      # .../Project/ai_interview_platform/scripts/resume_mapping
SCRIPTS_DIR = RESUME_MAPPING_DIR.parent                   # .../Project/ai_interview_platform/scripts
PROJECT_DIR = SCRIPTS_DIR.parent                          # .../Project/ai_interview_platform

# 👇 media is directly under the outer ai_interview_platform now
DATA_DIR = PROJECT_DIR / "media"
JD_DIR = DATA_DIR / "jd"
RESUME_DIR = DATA_DIR / "resumes"

# 👇 output is also directly under the outer ai_interview_platform
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
