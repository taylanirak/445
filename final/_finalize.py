"""One-command post-run finalisation.

After ``python final/src/run_all.py`` has produced ``results.json`` +
``predictions/``, this regenerates every deliverable from those artifacts and
validates the submission:

    figures → PDF → DOCX → PPTX → executed notebook → submission check → zip

Each step is independent; a failure is reported but does not abort the rest, so
you always see the full picture. Run from the repo root or from ``final/``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FINAL = Path(__file__).resolve().parent

STEPS = [
    ("figures", [sys.executable, str(FINAL / "_make_figures.py")]),
    ("report PDF", [sys.executable, str(FINAL / "_build_pdf.py")]),
    ("report DOCX", [sys.executable, str(FINAL / "_build_docx.py")]),
    ("slides PPTX", [sys.executable, str(FINAL / "_build_pptx.py")]),
    ("notebook (executed)", [sys.executable, str(FINAL / "_build_notebook.py")]),
    ("submission check",
     [sys.executable, str(FINAL / "_check_submission.py"),
      str(FINAL / "predictions" / "ensemble_test.jsonl")]),
    ("archive", [sys.executable, str(FINAL / "_make_archive.py")]),
]


def main():
    if not (FINAL / "results.json").exists():
        print("results.json not found — run `python final/src/run_all.py` first.")
        return 1
    failures = []
    for name, cmd in STEPS:
        print(f"\n{'=' * 70}\n[finalize] {name}\n{'=' * 70}", flush=True)
        rc = subprocess.run(cmd, cwd=str(FINAL.parent)).returncode
        if rc != 0:
            failures.append((name, rc))
            print(f"[finalize] !! {name} returned {rc}")
    print(f"\n{'=' * 70}")
    if failures:
        print("[finalize] completed WITH ISSUES:")
        for n, rc in failures:
            print(f"   - {n}: rc={rc}")
        return 1
    print("[finalize] all deliverables built and submission validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
