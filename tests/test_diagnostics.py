"""The model-diagnostics document never shows a count from 1 to 9."""
import re

from mtop import diagnostics as D


def test_small_counts_are_hidden():
    assert D.cnt(0) == "0" and D.cnt(4) == "<10" and D.cnt(10) == "10" and D.cnt(1234) == "1,234"
    assert D.events_cell(5, 40) == "·"          # the count itself under 10
    assert D.events_cell(35, 40) == "·"         # its complement under 10
    assert D.events_cell(20, 40) == "20"
    assert D.events_cell(0, 0) == "—"


def test_generated_document_has_no_small_counts():
    if not D.DOC.exists():
        return
    text = D.DOC.read_text(encoding="utf-8")
    for section in ("## Missing values", "## Rare categories"):
        if section not in text:
            continue
        body = text.split(section, 1)[1].split("\n## ", 1)[0]
        for line in body.splitlines():
            if not line.startswith("| ") or line.startswith("| Stage") or set(line) <= set("|- "):
                continue
            for cell in line.strip("|").split("|")[2:]:
                cell = cell.strip()
                if re.fullmatch(r"[\d,]+", cell):
                    assert int(cell.replace(",", "")) == 0 or int(cell.replace(",", "")) >= 10, line
