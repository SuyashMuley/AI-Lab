"""Tiny snapshot helper: JSON files under tests/snapshots, reviewed in git diffs.

A missing or different snapshot FAILS. Regenerate deliberately with UPDATE_SNAPSHOTS=1 and
review the diff.
"""

import json
import os
from pathlib import Path
from typing import Any

SNAP_DIR = Path(__file__).parent / "snapshots"


def assert_matches_snapshot(name: str, data: Any) -> None:
    path = SNAP_DIR / f"{name}.json"
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAP_DIR.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"missing snapshot {path.name}; run with UPDATE_SNAPSHOTS=1 and review"
    assert text == path.read_text(), (
        f"snapshot {path.name} differs; review, then UPDATE_SNAPSHOTS=1"
    )
