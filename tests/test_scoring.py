import pytest

from benchmarking.scoring import (
    correct_index,
    exact_match,
    extract_option,
    is_hallucination,
    option_match,
    score_predictions,
)

CHOICES = ["Ampicillin", "Ceftriaxone", "Vancomycin", "Ceftriaxone and vancomycin"]


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("B", 1),
        ("B.", 1),
        ("(C) Vancomycin", 2),
        ("D) Ceftriaxone and vancomycin", 3),
        ("A: Ampicillin is first line", 0),
        ("The answer is C.", 2),
        ("Answer: (B)", 1),
        ("the correct option D", 3),
        ("Ampicillin", 0),
        # Longest-choice wins when one option contains another.
        ("Start ceftriaxone and vancomycin empirically", 3),
    ],
)
def test_extract_option_reads_committed_option(prediction, expected):
    assert extract_option(prediction, CHOICES) == expected


@pytest.mark.parametrize(
    "prediction",
    [
        # Regression: old scorer accepted any "a" in the first 10 characters.
        "The patient needs further workup.",
        # Leading article must not be read as option A.
        "A lumbar puncture should be done first.",
        # Lowercase "a" after "answer is" is prose, not a letter.
        "The answer is a broad-spectrum antibiotic.",
        "",
    ],
)
def test_extract_option_returns_none_without_commitment(prediction):
    assert extract_option(prediction, CHOICES) is None


def test_extract_option_ignores_letters_beyond_choice_count():
    assert extract_option("E.", CHOICES) is None


def test_extract_option_no_choices():
    assert extract_option("A", []) is None


def test_correct_index_requires_exact_choice_text():
    # Old scorer used substring, so "Ceftriaxone" also matched option D.
    assert correct_index("Ceftriaxone", CHOICES) == 1
    assert correct_index("  ceftriaxone AND vancomycin ", CHOICES) == 3
    assert correct_index("Penicillin", CHOICES) is None


def test_option_match():
    assert option_match("B. Ceftriaxone", "Ceftriaxone", CHOICES)
    assert not option_match("D", "Ceftriaxone", CHOICES)
    assert not option_match("I am not sure.", "Ceftriaxone", CHOICES)
    assert not option_match("A", "Not an option", CHOICES)


def test_exact_match_normalizes_case_and_whitespace():
    assert exact_match("Give  CEFTRIAXONE now", "ceftriaxone")
    assert not exact_match("Give vancomycin", "ceftriaxone")


def test_hallucination_with_choices():
    assert is_hallucination("A. Ampicillin", "Ceftriaxone", CHOICES)
    assert not is_hallucination("B. Ceftriaxone", "Ceftriaxone", CHOICES)
    assert not is_hallucination("It might be A, but I am uncertain.", "Ceftriaxone", CHOICES)
    assert not is_hallucination("I don't know.", "Ceftriaxone", CHOICES)


def test_hallucination_without_choices_uses_legacy_heuristic():
    assert is_hallucination("It is definitely lupus.", "sarcoidosis")
    assert not is_hallucination("It is sarcoidosis.", "sarcoidosis")


def test_score_predictions_reports_coverage():
    records = [
        {"output": "Ceftriaxone", "choices": CHOICES},
        {"output": "Vancomycin", "choices": CHOICES},
        {"output": "Vancomycin"},  # legacy record without choices
    ]
    preds = ["B", "A. Ampicillin", "Vancomycin"]
    r = score_predictions(records, preds)
    assert r["n_samples"] == 3
    assert r["n_with_choices"] == 2
    assert r["option_accuracy"] == 0.5
    assert r["accuracy"] == pytest.approx(1 / 3)
    assert r["hallucination_count"] == 1


def test_score_predictions_option_accuracy_none_without_choices():
    r = score_predictions([{"output": "x"}], ["x"])
    assert r["option_accuracy"] is None
    assert r["n_with_choices"] == 0


def test_score_predictions_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_predictions([{"output": "x"}], [])
