import os
from typing import Any

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.openai import OpenAIResponses
from llama_index.llms.openai.responses import Tokenizer

DEFAULT_API_BASE = "https://api.abstract.inc/v1"
DEFAULT_MODEL = "glm-5.3:superfast"
CONTEXT_WINDOW = 262144
MAX_OUTPUT_TOKENS = 65536


class Abstract(OpenAIResponses):
    """
    Abstract LLM.

    Abstract serves `glm-5.3:superfast`, `deepseek-v4-flash:superfast` and
    `kimi-k3:superfast` through an OpenAI Responses-compatible API. This class is
    `OpenAIResponses` pointed at that API.

    Args:
        model: Abstract model id, for example `glm-5.3:superfast`.
        api_key: Abstract API key. Read from `ABSTRACT_API_KEY` if not passed.
            Create one with `abstract login` or at https://platform.abstract.inc/keys.
        api_base: API base URL. Read from `ABSTRACT_API_BASE` if not passed,
            otherwise `https://api.abstract.inc/v1`.
        context_window: Context window to report in the LLM metadata.
        **kwargs: Other `OpenAIResponses` arguments, such as `max_output_tokens`
            (at most 65,536) or `temperature`.

    Examples:
        `pip install llama-index-llms-abstract`

        ```python
        from llama_index.llms.abstract import Abstract

        llm = Abstract(model="glm-5.3:superfast")
        print(llm.complete("What is the capital of France?"))
        ```

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        context_window: int | None = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("ABSTRACT_API_KEY")
        if not api_key:
            raise ValueError(
                "Abstract API key not found. Set the ABSTRACT_API_KEY environment "
                "variable or pass api_key."
            )
        api_base = api_base or os.environ.get("ABSTRACT_API_BASE") or DEFAULT_API_BASE
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            context_window=context_window or CONTEXT_WINDOW,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Abstract"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window or CONTEXT_WINDOW,
            num_output=self.max_output_tokens or MAX_OUTPUT_TOKENS,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        # Send only the fields that are set: Abstract refuses an explicit
        # `previous_response_id: null`.
        model_kwargs = super()._get_model_kwargs(**kwargs)
        return {k: v for k, v in model_kwargs.items() if v is not None}

    @property
    def _tokenizer(self) -> Tokenizer | None:
        # tiktoken has no encoding for these models.
        return None
