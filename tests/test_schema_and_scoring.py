from ade_ft.evaluate import comparison_table, score
from ade_ft.schema import ADE, Extraction, parse_output

GOLD = Extraction(events=[ADE(drug="Metformin", reaction="Diarrhea", severity="mild")])


def test_parse_valid_and_normalised():
    out = parse_output('{"events": [{"drug": " Metformin", "reaction": "diarrhea ", "severity": "MILD"}]}')
    assert out == GOLD


def test_parse_tolerates_prose_and_trailing_text():
    txt = 'Here you go: {"events": []} and {"extra": 1}'
    assert parse_output(txt) == Extraction(events=[])


def test_parse_rejects_bad_schema():
    assert parse_output('{"events": [{"drug": "x", "reaction": "y", "severity": "fatal"}]}') is None
    assert parse_output("no json here") is None
    assert parse_output('{"events": [') is None


def test_score_perfect_and_partial():
    perfect = score([GOLD], [GOLD])
    assert perfect.f1 == 1.0 and perfect.severity_accuracy == 1.0 and perfect.json_valid_rate == 1.0

    wrong_sev = Extraction(events=[ADE(drug="metformin", reaction="diarrhea", severity="severe")])
    extra = Extraction(events=[*wrong_sev.events, ADE(drug="metformin", reaction="nausea", severity="mild")])
    s = score([extra], [GOLD])
    assert s.recall == 1.0 and s.precision == 0.5 and s.severity_accuracy == 0.0 and s.exact_match_rate == 0.0


def test_score_invalid_json_counts_as_misses():
    s = score([None], [GOLD])
    assert s.json_valid_rate == 0.0 and s.recall == 0.0


def test_empty_note_accuracy():
    empty = Extraction(events=[])
    assert score([empty], [empty]).empty_note_accuracy == 1.0
    assert score([GOLD], [empty]).empty_note_accuracy == 0.0


def test_comparison_table_has_deltas():
    a, b = score([None], [GOLD]), score([GOLD], [GOLD])
    assert "+1.000" in comparison_table(a, b)
