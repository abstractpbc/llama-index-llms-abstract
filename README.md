# LlamaIndex LLMs integration: Abstract

`llama-index-llms-abstract` connects LlamaIndex to [Abstract](https://abstract.inc).
Abstract serves `glm-5.3:superfast`, `deepseek-v4-flash:superfast` and
`kimi-k3:superfast` through an OpenAI Responses-compatible API at
`https://api.abstract.inc/v1`. The `Abstract` class is LlamaIndex's
`OpenAIResponses` pointed at that API.

## Install

```bash
pip install llama-index-llms-abstract
```

Create an API key with `abstract login` or at
<https://platform.abstract.inc/keys>, then set it:

```bash
export ABSTRACT_API_KEY="your-api-key"
```

`Abstract` reads only `ABSTRACT_API_KEY` (or `api_key=`). It never falls back
to `OPENAI_API_KEY`.

## Use

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.abstract import Abstract

llm = Abstract(model="glm-5.3:superfast")

print(llm.complete("What is the capital of France?"))

for r in llm.stream_chat([ChatMessage(role="user", content="Count from 1 to 5.")]):
    print(r.delta, end="")
```

## Models

| Model ID                      | Input | Context window | Max output tokens |
| ----------------------------- | ----- | -------------- | ----------------- |
| `glm-5.3:superfast`           | Text  | 262,144        | 65,536            |
| `deepseek-v4-flash:superfast` | Text  | 262,144        | 65,536            |
| `kimi-k3:superfast`           | Text  | 262,144        | 65,536            |

Prices: <https://abstract.inc/pricing>. API docs: <https://abstract.inc/docs>.

## Development

```bash
uv sync
uv run pytest
```
