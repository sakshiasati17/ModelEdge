import json
import subprocess

import pytest

from benchmarking import run_benchmark


def _fake_worker(results_by_key, fail=()):
    """Stand-in for the per-state subprocess: writes {key}_results.json."""

    def fake_run(cmd, check=False):
        key = cmd[cmd.index("--worker") + 1]
        out = cmd[cmd.index("--output") + 1]
        if key in fail:
            return subprocess.CompletedProcess(cmd, 1)
        with open(f"{out}/{key}_results.json", "w") as f:
            json.dump(results_by_key[key], f)
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def test_each_state_runs_in_its_own_worker(tmp_path, monkeypatch):
    calls = []
    fake = _fake_worker({"base": {"vram_gb": 6.0}, "int4": {"vram_gb": 2.5}})
    monkeypatch.setattr(run_benchmark.subprocess, "run", lambda cmd, check: calls.append(cmd) or fake(cmd))

    run_benchmark.run_all(["base", "int4"], "test.jsonl", tmp_path, max_samples=5)

    assert [c[c.index("--worker") + 1] for c in calls] == ["base", "int4"]
    assert all(c[1:3] == ["-m", "benchmarking.run_benchmark"] for c in calls)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary == {"base": {"vram_gb": 6.0}, "int4": {"vram_gb": 2.5}}


def test_failed_state_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    fake = _fake_worker({"base": {"vram_gb": 6.0}}, fail={"int8"})
    monkeypatch.setattr(run_benchmark.subprocess, "run", fake)

    with pytest.raises(SystemExit, match="int8"):
        run_benchmark.run_all(["base", "int8"], "test.jsonl", tmp_path, max_samples=5)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert list(summary) == ["base"]


def test_stale_result_file_is_not_reused(tmp_path, monkeypatch):
    (tmp_path / "int4_results.json").write_text(json.dumps({"vram_gb": 99}))
    monkeypatch.setattr(run_benchmark.subprocess, "run", _fake_worker({}, fail={"int4"}))

    with pytest.raises(SystemExit):
        run_benchmark.run_all(["int4"], "test.jsonl", tmp_path, max_samples=5)
    assert json.loads((tmp_path / "summary.json").read_text()) == {}


def test_quantized_states_load_from_the_adapter():
    paths = run_benchmark.MODEL_PATHS
    assert paths["int8"] == paths["int4"] == paths["fp16"]
    assert run_benchmark.QUANT_BITS == {"fp16": None, "int8": 8, "int4": 4, "base": None}
