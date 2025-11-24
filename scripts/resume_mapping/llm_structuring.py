import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json, argparse, time, re
from openai import OpenAI
from .paths import OUTPUT_DIR
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def extract_json_block(text: str) -> str:
    """
    Take the LLM raw text and try to slice out the first JSON object.
    We assume there's a top-level {...} object.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    return text[start:end + 1]


def build_primary_prompt(doc: Dict[str, Any]) -> str:
    """
    Main prompt used for the first LLM call.
    """
    doc_type = doc.get("doc_type", "unknown")
    text = doc.get("cleaned_text") or doc.get("raw_text") or ""

    # You can tweak / extend this schema later if you add more fields.
    schema_description = """
You MUST return a single JSON object with these exact top-level fields:

- "doc_id": string                      # e.g., filename or identifier
- "doc_type": "jd" or "resume"

- "title": string or null               # job title or candidate title
- "location": string or null            # main location mentioned

- "experience_years": number or null    # approx total years of experience (for resumes) or required experience (for JDs)
- "experience_range": {                 # optional; for JDs if range is clearly stated
    "min": number or null,
    "max": number or null
  }

- "skills": array of strings            # canonical skill names, lowercase, e.g. ["python", "pandas", "sql"]
- "domains": array of strings           # high-level domains, like ["finance", "ecommerce", "healthcare", "supply_chain", "consulting", "education", "research", "cloud", "data_platforms"]

- "education": array of strings         # degrees/lines describing education
- "last_company": string or null        # for resumes (most recent job/company); null for JDs

- "seniority_level": string or null     # e.g. "junior", "mid", "senior", "lead", "principal"
- "company": string or null             # for JDs: hiring company; for resumes: current or most recent employer

- "project_summaries": array of strings # 2–5 short bullet-like summaries of important projects / responsibilities
- "profile_summary": string or null     # 2–4 sentences summary of the role (JD) or candidate (resume)
"""

    instructions = f"""
You are an information extraction assistant for a hiring platform.

Your task:
- Read the following document (type: "{doc_type}").
- Extract the fields according to the schema below.
- If something is unknown or missing, use null or an empty list (for arrays).
- Normalize skills to lowercase.
- Respond with a SINGLE JSON object ONLY (no extra text, no explanations, no comments).

{schema_description}

Now here is the document text (may be messy; it's okay):

\"\"\"{text}\"\"\"
"""

    return instructions.strip()


def build_strict_retry_prompt(doc: Dict[str, Any], last_error: str) -> str:
    """
    Simpler, stricter prompt for the retry when JSON parsing fails.
    """
    doc_type = doc.get("doc_type", "unknown")
    text = doc.get("cleaned_text") or doc.get("raw_text") or ""

    schema_short = """
Return ONLY valid JSON, minified or pretty-printed, with these top-level keys:

{
  "doc_id": string,
  "doc_type": "jd" or "resume",
  "title": string or null,
  "location": string or null,
  "experience_years": number or null,
  "experience_range": { "min": number or null, "max": number or null },
  "skills": string[],
  "domains": string[],
  "education": string[],
  "last_company": string or null,
  "seniority_level": string or null,
  "company": string or null,
  "project_summaries": string[],
  "profile_summary": string or null
}
"""

    instructions = f"""
Your previous output could not be parsed as JSON. Error was:

{last_error}

Try again.

IMPORTANT RULES:
- Return ONLY a JSON object. No comments, no backticks, no explanations.
- Make sure all strings are double-quoted.
- Do NOT leave trailing commas.

Document type: "{doc_type}".

Document text:

\"\"\"{text}\"\"\"

{schema_short}
"""

    return instructions.strip()


def call_llm_with_retry(
    doc: Dict[str, Any],
    model: str,
    max_retries: int = 2
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Call the LLM up to max_retries times.
    Returns (parsed_json_or_none, raw_text_or_none, error_or_none).
    """
    raw_response_text: Optional[str] = None
    last_error: Optional[str] = None

    for attempt in range(max_retries):
        if attempt == 0:
            prompt = build_primary_prompt(doc)
        else:
            # On retry: use stricter prompt and pass the last error message
            prompt = build_strict_retry_prompt(doc, last_error or "Unknown JSON error")

        try:
            # ❌ FIXED: removed the invalid beta header
            resp = client.responses.create(
                model=model,
                input=prompt,
            )

            # Be robust to different SDK object shapes
            first_output = resp.output[0].content[0].text
            raw_response_text = str(first_output)

            # Extract JSON region and parse
            json_str = extract_json_block(raw_response_text)
            parsed = json.loads(json_str)
            return parsed, raw_response_text, None

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    # If we reach here, all attempts failed
    return None, raw_response_text, last_error



# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        default=str(OUTPUT_DIR / "cleaned_docs.json"),
        help="Path to cleaned_docs.json (output of preprocessing.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        default=str(OUTPUT_DIR / "structured_docs.json"),
        help="Where to save structured_docs.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model to use (e.g. gpt-4.1-mini, gpt-5.1, etc.)",
    )

    args = parser.parse_args()

    # Load input docs
    with open(args.input, "r", encoding="utf-8") as f:
        docs: List[Dict[str, Any]] = json.load(f)

    structured_docs: List[Dict[str, Any]] = []

    for idx, doc in enumerate(docs, start=1):
        doc_id = doc.get("doc_id") or f"doc_{idx}"
        doc_type = doc.get("doc_type", "unknown")
        print(f"[{idx}/{len(docs)}] Structuring doc_id={doc_id} (type={doc_type}) ...")

        parsed, raw_text, error = call_llm_with_retry(doc, model=args.model, max_retries=2)

        if parsed is not None:
            # Ensure doc_id and doc_type are present
            parsed.setdefault("doc_id", doc_id)
            parsed.setdefault("doc_type", doc_type)

            structured_docs.append(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "structured": parsed,
                    "_raw_llm_output": raw_text,
                    "_error": None,
                }
            )
        else:
            print(f"  -> FAILED to parse JSON for doc_id={doc_id}. Error: {error}")
            structured_docs.append(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "structured": None,
                    "_raw_llm_output": raw_text,
                    "_error": error,
                }
            )

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(structured_docs, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(structured_docs)} structured records to {args.output}")


if __name__ == "__main__":
    main()
