import json

import httpx
import pytest
from fastapi.testclient import TestClient

from data.dataset_utils import format_alpaca
from serving import api

CHOICES = ["Ampicillin", "Ceftriaxone", "Vancomycin", "Doxycycline"]
QUESTION = "Which antibiotic is first line for this patient?"


def test_prompt_matches_training_format():
    # Serving must send exactly the training prompt, up to the response.
    trained = format_alpaca({"question": QUESTION, "choices": CHOICES, "answer": "Ceftriaxone"})
    assert api._build_prompt(QUESTION, CHOICES) + "Ceftriaxone" == trained["text"]


def test_prompt_without_choices_matches_training_format():
    trained = format_alpaca({"question": QUESTION, "choices": [], "answer": "x"})
    assert api._build_prompt(QUESTION, None) + "x" == trained["text"]


@pytest.fixture
def client(monkeypatch):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "modeledge-medqa"}]})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "modeledge-medqa", "choices": [{"text": " B. Ceftriaxone\n"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        api.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(api, "_model_name", "unknown")
    with TestClient(api.app) as c:
        yield c, sent


def test_infer_uses_completions_with_training_prompt(client):
    c, sent = client
    r = c.post("/infer", json={"question": QUESTION, "choices": CHOICES})
    assert r.status_code == 200
    assert r.json()["answer"] == "B. Ceftriaxone"
    assert r.json()["model"] == "modeledge-medqa"
    assert sent[0]["prompt"] == api._build_prompt(QUESTION, CHOICES)
    assert "messages" not in sent[0]
    assert sent[0]["stop"] == api.STOP_SEQUENCES


def test_infer_rejects_too_many_choices(client):
    c, _ = client
    r = c.post("/infer", json={"question": QUESTION, "choices": [str(i) for i in range(11)]})
    assert r.status_code == 422


def test_infer_vllm_unreachable_returns_503(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        api.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    with TestClient(api.app) as c:
        r = c.post("/infer", json={"question": QUESTION})
        assert r.status_code == 503
        assert c.get("/health").json() == {"status": "ok", "vllm_reachable": False}
