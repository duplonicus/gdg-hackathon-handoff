"""The decision log is UTF-8 on every platform (Windows would otherwise read it as cp1252)."""

import json

from handoff.logbook import read_decisions


def test_read_decisions_reads_utf8(tmp_path):
    row = {"question": "finish", "text": "tilted 15° — scratched"}
    path = tmp_path / "decisions.jsonl"
    path.write_bytes((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    assert read_decisions(path) == [row]
