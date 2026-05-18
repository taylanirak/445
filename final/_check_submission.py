"""Validate a predictions JSONL before zipping / CodaBench upload.

Checks, against the matching gold split, that:

* every line is valid JSON with ``id`` and ``prediction`` keys,
* ids are strings and cover the gold split *exactly once* (no missing/extra),
* predictions parse as numbers within [1, 5] (continuous is allowed — the
  official scorer consumes the raw value; this only warns, like
  ``format_check.py``).

It reuses the repository's own ``format_check.check_formatting`` so a file that
passes here passes the grader's parser too.

Usage::

    python final/_check_submission.py final/predictions/ensemble_test.jsonl
    python final/_check_submission.py final/predictions/ensemble_dev.jsonl dev
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from format_check import check_formatting  # the grader's own parser


def _gold_for(pred_path: Path) -> str:
    """Infer which split a predictions file targets from its name."""
    stem = pred_path.stem.lower()
    if "test" in stem:
        return "test"
    if "dev" in stem:
        return "dev"
    return "dev"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python final/_check_submission.py <predictions.jsonl> [split]")
        return 2
    pred_path = Path(argv[1])
    split = argv[2] if len(argv) > 2 else _gold_for(pred_path)

    with open(REPO / "data" / f"{split}.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    gold = [{"id": str(k)} for k in raw]  # check_formatting only needs ids

    n_lines = sum(1 for _ in open(pred_path, encoding="utf-8"))
    print(f"Validating {pred_path} against split '{split}' "
          f"({len(gold)} gold ids, {n_lines} prediction lines)")

    ok = check_formatting(str(pred_path), gold)
    if not ok:
        print("FAILED: file does not satisfy the official format_check.")
        return 1
    if n_lines != len(gold):
        print(f"FAILED: line count {n_lines} != gold size {len(gold)}.")
        return 1
    print(f"OK: {pred_path.name} is a valid submission for split '{split}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
