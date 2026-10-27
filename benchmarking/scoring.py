"""
Pure-Python scoring functions for the accuracy benchmark.

Kept free of torch/transformers imports so they can be unit-tested in CI
without a GPU.

Metrics:
  - exact_match: ground-truth answer text appears in the output (case-insensitive)
  - option_match: the option the model committed to is the correct one
  - is_hallucination: the model committed to a wrong option without hedging
"""

import re

OPTION_LABELS = "ABCDEFGHIJ"

# Phrases that signal the model is hedging rather than asserting an answer.
HEDGE_PATTERNS = [
    r"\b(not sure|uncertain|unclear|cannot be determined|can't be determined)\b",
    r"\b(may|might|possibly|perhaps|likely|unlikely|probably)\b",
]

# Legacy heuristic, used only for free-text records that have no choices.
CONFIDENCE_PATTERNS = [
    r"\b(definitely|certainly|absolutely|always|never|clearly|obviously)\b",
    r"\b(the answer is|it is|this is|that is)\b",
]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    return _norm(ground_truth) in _norm(prediction)


def correct_index(answer: str, choices: list[str]) -> int | None:
    """Index of the choice equal to the answer text, or None if not found."""
    target = _norm(answer)
    for i, choice in enumerate(choices):
        if _norm(choice) == target:
            return i
    return None


def extract_option(prediction: str, choices: list[str]) -> int | None:
    """Return the index of the option the model committed to, or None.

    Tries, in order:
      1. A leading letter: "B", "B.", "(B)", "B) ...", "B: ..." (not "B cells")
      2. An explicit statement: "answer is B", "answer: (B)", "option B"
      3. Exactly one choice's full text appearing in the output
    """
    n = len(choices)
    if n == 0:
        return None
    letters = OPTION_LABELS[:n]

    # A bare leading capital needs punctuation after it, so a reply starting
    # with the article "A ..." is not read as option A.
    lead = re.match(rf"^\s*(?:\(([{letters}])\)|([{letters}])(?:[\).:,]|\s*$))", prediction)
    if lead:
        return letters.index(lead.group(1) or lead.group(2))

    # Keyword is case-insensitive, the letter is not ("answer is a ..." is prose).
    stated = re.search(
        rf"(?i:\b(?:the answer is|answer|option|choice))\s*[:\-]?\s*\(?([{letters}])(?![\w-])",
        prediction,
    )
    if stated:
        return letters.index(stated.group(1))

    pred = _norm(prediction)
    hits = [i for i, c in enumerate(choices) if _norm(c) and _norm(c) in pred]
    # Drop choices that only match because they are contained in a longer hit.
    hits = [
        i for i in hits
        if not any(j != i and _norm(choices[i]) in _norm(choices[j]) for j in hits)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def option_match(prediction: str, answer: str, choices: list[str]) -> bool:
    gold = correct_index(answer, choices)
    if gold is None:
        return False
    return extract_option(prediction, choices) == gold


def is_hedged(prediction: str) -> bool:
    return any(re.search(p, prediction, re.IGNORECASE) for p in HEDGE_PATTERNS)


def is_hallucination(prediction: str, answer: str, choices: list[str] | None = None) -> bool:
    """A confidently wrong answer.

    With choices: the model picked a definite option, it is wrong, and the
    output does not hedge. Without choices: falls back to the legacy
    confident-phrasing heuristic on a non-matching answer.
    """
    if choices:
        picked = extract_option(prediction, choices)
        gold = correct_index(answer, choices)
        if picked is None or gold is None or picked == gold:
            return False
        return not is_hedged(prediction)

    if exact_match(prediction, answer):
        return False
    return any(re.search(p, prediction, re.IGNORECASE) for p in CONFIDENCE_PATTERNS)


def score_predictions(records: list[dict], predictions: list[str]) -> dict:
    exact_hits = 0
    option_hits = 0
    hallucination_count = 0
    n_with_choices = 0

    for rec, prediction in zip(records, predictions, strict=True):
        answer = rec.get("output", rec.get("answer", ""))
        choices = rec.get("choices") or []

        if exact_match(prediction, answer):
            exact_hits += 1
        if choices:
            n_with_choices += 1
            if option_match(prediction, answer, choices):
                option_hits += 1
        if is_hallucination(prediction, answer, choices):
            hallucination_count += 1

    n = len(records)
    return {
        "accuracy": exact_hits / n if n else 0.0,
        "option_accuracy": option_hits / n_with_choices if n_with_choices else None,
        "hallucination_rate": hallucination_count / n if n else 0.0,
        "n_samples": n,
        # Coverage: how many records the option scorer could actually grade.
        "n_with_choices": n_with_choices,
        "exact_hits": exact_hits,
        "option_hits": option_hits,
        "hallucination_count": hallucination_count,
    }
