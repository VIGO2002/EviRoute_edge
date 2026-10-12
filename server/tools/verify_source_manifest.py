#!/usr/bin/env python3
"""Verify every archived source-file SHA256 without importing ML libraries.

Historical code is unchanged. One protocol README moved to docs/ as part of
the repository layout cleanup. The historical manifest is preserved verbatim.
"""
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "SOURCE_FILE_MANIFEST.csv"
MOVED = {"README_MATCHED_PAIR_20K.md": "docs/MATCHED_PAIR_20K.md"}


def main() -> int:
    failures = []
    checked = 0
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            original = row["path"]
            relative = MOVED.get(original, original)
            actual = ROOT / relative
            if not actual.is_file():
                failures.append(f"MISSING {relative}")
                continue
            blob = actual.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != row["sha256"]:
                failures.append(f"HASH MISMATCH {relative}")
            if len(blob) != int(row["bytes"]):
                failures.append(f"SIZE MISMATCH {relative}")
            checked += 1
    for entry in failures:
        print(entry, file=sys.stderr)
    if failures:
        print(f"Source identity FAILED ({checked} files scanned)", file=sys.stderr)
        return 1
    print(f"Source identity PASS ({checked} archived files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
