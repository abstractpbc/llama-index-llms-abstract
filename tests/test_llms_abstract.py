import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import ChatMessage

from llama_index.llms.abstract import Abstract
from llama_index.llms.abstract.base import CONTEXT_WINDOW, DEFAULT_API_BASE


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in Abstract.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABSTRACT_API_BASE", raising=False)
    llm = Abstract(model="kimi-k3:superfast", api_key="test-key")
    assert llm.api_base == DEFAULT_API_BASE
    assert llm.metadata.context_window == CONTEXT_WINDOW
    assert llm.metadata.is_function_calling_model
    assert llm.metadata.model_name == "kimi-k3:superfast"
    assert llm._tokenizer is None
    assert llm.class_name() == "Abstract"


def test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABSTRACT_API_KEY", "env-key")
    monkeypatch.setenv("ABSTRACT_API_BASE", "https://example.test/v1")
    llm = Abstract()
    assert llm.api_key == "env-key"
    assert llm.api_base == "https://example.test/v1"


def test_missing_key_never_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ABSTRACT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(ValueError, match="ABSTRACT_API_KEY"):
        Abstract()


def test_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = Abstract(model="glm-5.3:superfast", api_key="test-key")
    captured: dict = {}

    class Done(Exception):
        pass

    def create(**kwargs: object) -> None:
        captured.update(kwargs)
        raise Done

    monkeypatch.setattr(llm._client.responses, "create", create)
    with pytest.raises(Done):
        llm.chat([ChatMessage(role="user", content="hi")])
    assert captured["model"] == "glm-5.3:superfast"
    assert captured["stream"] is False
    assert str(llm._client.base_url).rstrip("/") == DEFAULT_API_BASE
    assert "previous_response_id" not in captured
