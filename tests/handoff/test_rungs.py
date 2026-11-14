"""The Decider rung speaks the TypeSafe /v1/systemone shape to the Strands server and, with a model name, to Ollama."""

import pytest

import handoff.rungs as rungs
from handoff.decider import QUESTIONS
from handoff.rungs import DeciderRung


def capture_posts(monkeypatch):
    posts = []

    def fake_post(url, body, timeout):
        posts.append((url, body))
        qid = next(iter(body["questions"]))
        options = list(body["questions"][qid]["criteria"])
        return {"answers": {qid: {"probabilities": {o: (0.7 if i == 0 else 0.3 / (len(options) - 1)) for i, o in enumerate(options)}}}, "latency_ms": 5}

    monkeypatch.setattr(rungs, "_post", fake_post)
    return posts


def test_strands_server_request_has_no_model_field_and_raw_base64_image(monkeypatch):
    posts = capture_posts(monkeypatch)
    rung = DeciderRung("decider-2b-vision", "http://127.0.0.1:8765", with_image=True)
    answer = rung.answer(QUESTIONS["finish"], "a part in the station", "AAAA")
    url, body = posts[0]
    assert url == "http://127.0.0.1:8765/v1/systemone"
    assert "model" not in body
    assert body["images"] == ["AAAA"]
    assert body["questions"]["finish"]["type"] == "choice"
    assert set(body["questions"]["finish"]["criteria"]) == set(QUESTIONS["finish"].options)
    assert answer.error is None and abs(sum(answer.probs.values()) - 1.0) < 1e-6


def test_ollama_request_names_the_model_and_text_arms_send_no_image(monkeypatch):
    posts = capture_posts(monkeypatch)
    rung = DeciderRung("nimble-text", "http://127.0.0.1:11434", with_image=False, model="nimble:9b")
    rung.answer(QUESTIONS["in_bin"], "the hand released the part", "AAAA")
    url, body = posts[0]
    assert url == "http://127.0.0.1:11434/v1/systemone"
    assert body["model"] == "nimble:9b"
    assert "images" not in body


def test_ollama_health_means_the_model_is_pulled(monkeypatch):
    monkeypatch.setattr(rungs, "_get", lambda url, timeout=3.0: {"models": [{"name": "clef-flash:9b"}, {"name": "gemma4:e4b"}]})
    assert DeciderRung("clef-flash-vision", "http://127.0.0.1:11434", model="clef-flash:9b").healthy()
    assert not DeciderRung("nimble-text", "http://127.0.0.1:11434", model="nimble:9b").healthy()


def test_ollama_and_decider_urls_come_from_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("HANDOFF_OLLAMA_URL", "http://127.0.0.1:11435")
    monkeypatch.delenv("HANDOFF_DECIDER_URL", raising=False)
    try:
        fresh = importlib.reload(rungs)
        assert fresh.OLLAMA_URL == "http://127.0.0.1:11435"
        assert fresh.DECIDER_URL == "http://127.0.0.1:8765"
    finally:
        monkeypatch.delenv("HANDOFF_OLLAMA_URL")
        importlib.reload(rungs)


def test_unload_ollama_models_posts_keep_alive_zero_for_each_loaded_model(monkeypatch):
    posts = []
    monkeypatch.setattr(rungs, "_get", lambda url, timeout=3.0: {"models": [{"name": "gemma4:e4b"}, {"name": "gemma4:e2b"}]})
    monkeypatch.setattr(rungs, "_post", lambda url, body, timeout: posts.append((url, body)) or {})
    rungs.unload_ollama_models("http://127.0.0.1:11435")
    assert [(u, b["model"], b["keep_alive"]) for u, b in posts] == [
        ("http://127.0.0.1:11435/api/generate", "gemma4:e4b", 0),
        ("http://127.0.0.1:11435/api/generate", "gemma4:e2b", 0),
    ]


def test_d3_server_gets_a_data_url_image_and_its_ready_health_counts(monkeypatch):
    posts = capture_posts(monkeypatch)
    rung = DeciderRung("d3-nano-vision", "http://127.0.0.1:8768", with_image=True, data_url=True)
    rung.answer(QUESTIONS["finish"], "a part in the station", "AAAA")
    assert posts[0][1]["images"] == ["data:image/png;base64,AAAA"]
    monkeypatch.setattr(rungs, "_get", lambda url, timeout=3.0: {"status": "ready"})
    assert rung.healthy()
    monkeypatch.setattr(rungs, "_get", lambda url, timeout=3.0: {"status": "loading"})
    assert not rung.healthy()


def fake_routes(monkeypatch, routes):
    def fake_get(url, timeout=3.0):
        for path, body in routes.items():
            if url.endswith(path):
                return body
        raise OSError("404")

    monkeypatch.setattr(rungs, "_get", fake_get)


def test_served_model_reads_llama_server_qev_and_startlux_names(monkeypatch):
    fake_routes(monkeypatch, {"/health": {"status": "ok"}, "/v1/models": {"data": [{"id": "d3-27B-Artemis-IQ4_XS.gguf"}]}, "/props": {"model_path": "D:\\Models\\gguf\\d3-27b\\d3-27B-Artemis-IQ4_XS.gguf"}})
    assert "d3-27b-artemis" in rungs.served_model("http://127.0.0.1:8774")
    fake_routes(monkeypatch, {"/health": {"status": "ok", "model": "OneJev1.6-27B.IQ4_XS"}, "/v1/models": {"models": [{"name": "OneJev1.6-27B.IQ4_XS"}, {"name": "jev-latest"}]}})
    assert rungs.served_model("http://127.0.0.1:8774").startswith("onejev1.6-27b.iq4_xs")
    fake_routes(monkeypatch, {})
    assert rungs.served_model("http://127.0.0.1:8774") == ""


def test_an_arm_on_a_shared_port_refuses_another_arms_server(monkeypatch):
    fake_routes(monkeypatch, {"/v1/models": {"data": [{"id": "Clef-Artemis-IQ4_XS.gguf"}]}})
    d3 = DeciderRung("d3-27b-vision", "http://127.0.0.1:8774", expect=("d3-27b",))
    with pytest.raises(RuntimeError, match="expected d3-27b"):
        d3.check_model()
    assert DeciderRung("clef-27b-vision", "http://127.0.0.1:8774", expect=("clef",)).check_model() == "clef-artemis-iq4_xs.gguf"
    fake_routes(monkeypatch, {"/health": {"model": "OneJev1.6-27B.IQ4_XS"}})
    with pytest.raises(RuntimeError):
        DeciderRung("onejev-9b-vision", "http://127.0.0.1:8774", expect=("onejev", "9b")).check_model()
    fake_routes(monkeypatch, {})
    with pytest.raises(RuntimeError, match="an unnamed model"):
        d3.check_model()
