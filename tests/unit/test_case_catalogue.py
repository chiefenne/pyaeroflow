"""Every example case loads, and docs/cases.md lists exactly the example cases."""

import re
from pathlib import Path

import pytest

from pyaeroflow.config import load_case

ROOT = Path(__file__).parents[2]
CASES = sorted((ROOT / "examples" / "cases").glob("*.toml"))


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.stem)
def test_example_case_is_valid(case):
    """The checks of 'pyaeroflow check' pass for every example case file."""
    assert load_case(case).name


def test_catalogue_lists_every_example_case():
    text = (ROOT / "docs" / "cases.md").read_text()
    listed = set(re.findall(r"`([A-Za-z0-9_]+\.toml)`", text))
    present = {p.name for p in CASES}
    assert listed == present, (
        f"not in docs/cases.md: {sorted(present - listed)}; "
        f"in docs/cases.md but no file: {sorted(listed - present)}"
    )


def test_catalogue_template_names_exist():
    from pyaeroflow.mesh.library import LIBRARY
    from pyaeroflow.templates import TEMPLATES

    text = (ROOT / "docs" / "cases.md").read_text()
    for name in re.findall(r"template `([a-z0-9-]+)`", text):
        assert name in TEMPLATES, name
    for name in re.findall(r"library `([a-z0-9-]+)`", text):
        assert name in LIBRARY, name
    listed = set(re.findall(r"template `([a-z0-9-]+)`", text))
    missing = {n for n, t in TEMPLATES.items() if t.case_file} - listed
    assert not missing, f"templates not in docs/cases.md: {sorted(missing)}"
