from __future__ import annotations

import os
from types import SimpleNamespace


def test_llm_client_cloud_monkeypatch(monkeypatch):
    os.environ["APP_MODE"] = "cloud"
    os.environ.pop("OPENAI_API_KEY", None)

    from ai_module.core import llm_client

    # No API key should trigger stub path but still return a string
    out = llm_client.generate("Say hi", sys_prompt="sys")
    assert isinstance(out, str)
    assert out.startswith("[openai-stub]")


def test_llm_client_local_mock_requests(monkeypatch):
    os.environ["APP_MODE"] = "local"

    class MockResp:
        status_code = 200

        def json(self):
            return {"response": "hello from ollama"}

    def mock_post(url, json, timeout):  # noqa: A002 - shadow json ok in test
        return MockResp()

    monkeypatch.setattr("requests.post", mock_post, raising=False)

    from ai_module.core import llm_client

    out = llm_client.generate("test", sys_prompt=None)
    assert isinstance(out, str)
    assert "ollama" in out
