import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from pathlib import Path
import os, re, json

from pathlib import Path
from pypdf import PdfReader
from unidecode import unidecode
import spacy
from spacy.cli import download as spacy_download
from .paths import JD_DIR, RESUME_DIR, OUTPUT_DIR


def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Ensure that the spaCy model is installed and loadable.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model: {model_name} ...")
        spacy_download(model_name)
        return spacy.load(model_name)


# -------------------------------------------------------------------
# Helpers to collect files
# -------------------------------------------------------------------

def collect_pdf_files(folder: str | Path, doc_type: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (path, doc_type) for all .pdf files in a folder.
    doc_type is "jd" or "resume".
    """
    folder = str(folder)
    files: List[Tuple[str, str]] = []

    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return files

    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            full_path = os.path.join(folder, name)
            files.append((full_path, doc_type))

    return files


# -------------------------------------------------------------------
# Data structure
# -------------------------------------------------------------------

@dataclass
class CleanedDoc:
    doc_id: str
    doc_type: str        # "jd" or "resume"
    file_path: str
    raw_text: str
    cleaned_text: str          # generic cleaned version for LLMs / embeddings
    content_words_text: str    # optional: noun/verb/adjective-heavy version


# -------------------------------------------------------------------
# Generic Preprocessor
# -------------------------------------------------------------------

class GenericPreprocessor:
    def __init__(self):
        # spaCy model for POS tagging to drop prepositions etc.
        self.nlp = ensure_spacy_model("en_core_web_sm")
        self.multiple_space_pattern = re.compile(r"[ \t]{2,}")
        self.multiple_newline_pattern = re.compile(r"\n{3,}")

    # -------- PDF text extraction -------- #

    def extract_text_from_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages)

    # -------- Generic cleaning (LLM-friendly) -------- #

    def basic_clean(self, text: str) -> str:
            # 0. Fix section-sign bullets *before* unidecode, otherwise '§' -> 'SS'
            text = text.replace("§", "* ")
            # 1. Normalize accents etc.
            text = unidecode(text)
            # 2. Normalize bullet characters to "* " so they are consistent
            bullet_chars = ["\u2022", "•", "", "●", "★", "■"]
            for b in bullet_chars:
                text = text.replace(b, "* ")
            # 3. Replace tabs with single space
            text = text.replace("\t", " ")
            # 4. Remove very noisy repeated punctuation lines
            text = re.sub(r"[=]{3,}", " ", text)
            text = re.sub(r"[_]{3,}", " ", text)
            text = re.sub(r"[-]{5,}", " ", text)
            # 5. Fix typical PDF “run-on” patterns BEFORE we collapse spaces
            #    a) lowerCase followed by UpperCase: "DataScientist" -> "Data Scientist"
            text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
            #    b) letter followed by digit: "with4.5" -> "with 4.5"
            text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
            #    c) digit followed by letter: "2021Microsoft" -> "2021 Microsoft"
            text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
            # 6. Normalize newlines
            text = text.replace("\r", "\n")
            text = self.multiple_newline_pattern.sub("\n\n", text)
            # 7. Collapse multiple spaces
            text = self.multiple_space_pattern.sub(" ", text)
            # 8. Keep only printable characters + newlines
            text = "".join(ch for ch in text if ch.isprintable() or ch in "\n")
            return text.strip()


    # -------- Content-words-only version -------- #

    def keep_content_words(self, text: str) -> str:
        """
        Build a version that keeps mainly:
        - NOUN, PROPN, VERB, ADJ, NUM
        Drops:
        - prepositions, conjunctions, determiners, pronouns, etc.
        This is useful if you want a more "semantic core" of the text.
        """
        doc = self.nlp(text)

        content_tokens: list[str] = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue

            if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB", "NUM"}:
                content_tokens.append(token.text)

        return " ".join(content_tokens)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    pre = GenericPreprocessor()
    cleaned_docs: List[CleanedDoc] = []

    # Collect all JD + resume files
    all_files: List[Tuple[str, str]] = []
    all_files.extend(collect_pdf_files(JD_DIR, "jd"))
    all_files.extend(collect_pdf_files(RESUME_DIR, "resume"))

    print(f"Found {len(all_files)} PDF files in total.")

    for path, doc_type in all_files:
        print(f"Processing ({doc_type}): {path}")
        raw_text = pre.extract_text_from_pdf(path)
        cleaned_text = pre.basic_clean(raw_text)
        content_words_text = pre.keep_content_words(cleaned_text)

        cleaned_docs.append(
            CleanedDoc(
                doc_id=os.path.basename(path),
                doc_type=doc_type,
                file_path=path,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                content_words_text=content_words_text,
            )
        )

    # Save results
    output = [asdict(doc) for doc in cleaned_docs]
    out_path = OUTPUT_DIR / "cleaned_docs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(cleaned_docs)} documents to {out_path}")


if __name__ == "__main__":
    main()
