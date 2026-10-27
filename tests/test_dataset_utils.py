from data.dataset_utils import format_alpaca, split_dataset, validate_record

RECORD = {
    "question": "Which antibiotic is first line for this patient?",
    "choices": ["Ampicillin", "Ceftriaxone", "Vancomycin", "Doxycycline"],
    "answer": "Ceftriaxone",
}


def test_format_alpaca_persists_choices():
    # Regression: choices were dropped, so option_accuracy was always None.
    out = format_alpaca(RECORD)
    assert out["choices"] == RECORD["choices"]
    assert out["output"] == "Ceftriaxone"
    assert "B. Ceftriaxone" in out["input"]
    assert out["text"].endswith("### Response:\nCeftriaxone")


def test_validate_record():
    assert validate_record(RECORD)
    assert not validate_record({**RECORD, "question": "short"})
    assert not validate_record({**RECORD, "answer": ""})


def test_split_dataset_is_disjoint_and_deterministic():
    records = [{"i": i} for i in range(11451)]
    train, val, test = split_dataset(records, seed=42)
    assert (len(train), len(val), len(test)) == (9733, 1145, 573)
    ids = [r["i"] for r in train + val + test]
    assert len(set(ids)) == len(records)
    assert split_dataset(records, seed=42)[2] == test
