"""Bundle the final deliverables into one submission archive.

The project description asks for the report, slides and code compressed into a
single file. This collects the deliverables + everything needed to run the code
into ``final/GroupXX_final_submission.zip`` and prints a manifest. Heavy / regen
artifacts (HF cache, scratch files) are excluded.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

FINAL = Path(__file__).resolve().parent
REPO = FINAL.parent
ZIP = FINAL / "GroupXX_final_submission.zip"

# Deliverables (graded artifacts).
DELIVERABLES = [
    "GroupXX_final_report.md",
    "GroupXX_final_report.pdf",
    "GroupXX_final_report.docx",
    "Group24_final_presentation.pptx",
    "GroupXX_final_notebook.ipynb",
    "README.md",
    "RUN_ON_GPU.md",
    "requirements_final.txt",
    "results.json",
]
# Source + helpers needed to re-run.
CODE_GLOBS = ["src/*.py", "_build_*.py", "_make_figures.py",
              "_check_submission.py", "_make_archive.py", "_finalize.py"]
# Generated outputs worth shipping.
OUT_GLOBS = ["figures/*.png", "predictions/*.jsonl", "_solution_dev.jsonl",
             "_run_all.log"]
# Repo files required for the official scorer to run from the archive.
REPO_FILES = ["scoring.py", "format_check.py", "requirements.txt",
              "data/train.json", "data/dev.json", "data/test.json"]

EXCLUDE_DIRS = {".hf_cache", "__pycache__"}


def _add(zf: zipfile.ZipFile, path: Path, arcname: str) -> bool:
    if not path.exists() or any(p in EXCLUDE_DIRS for p in path.parts):
        return False
    zf.write(path, arcname)
    return True


def main():
    added, missing = [], []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in DELIVERABLES:
            (added if _add(zf, FINAL / name, f"final/{name}") else missing
             ).append(name)
        for pat in CODE_GLOBS + OUT_GLOBS:
            for p in sorted(FINAL.glob(pat)):
                if _add(zf, p, f"final/{p.relative_to(FINAL)}"):
                    added.append(str(p.relative_to(FINAL)))
        for rf in REPO_FILES:
            if _add(zf, REPO / rf, rf):
                added.append(rf)

    print(f"Wrote {ZIP}  ({ZIP.stat().st_size/1e6:.2f} MB, {len(added)} files)")
    if missing:
        print("WARNING — not yet built (run the build steps in README first):")
        for m in missing:
            print("   -", m)
    else:
        print("All graded deliverables present.")


if __name__ == "__main__":
    main()
