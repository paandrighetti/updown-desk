"""The passive quoting note and the README summary match the raw outputs they are built from."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research" / "passive_note.py"


def test_note_and_readme_summary_are_up_to_date():
    spec = importlib.util.spec_from_file_location("passive_note", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0
