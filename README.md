# yolorag

Provider-agnostic RAG orchestrator prototype for the Ultralytics LLM Engineer take-home challenge.

The first implementation slice focuses on the architecture surfaces that matter most:

- Provider classes normalize vendor-specific responses.
- Usage extractors convert raw provider usage into one `TokenUsage` shape.
- Cost calculation is owned by this project, with local pricing overrides and `genai-prices` as the fallback pricing backend.
- The orchestrator decides whether retrieval/review are needed based on response mode and request intent.
- The CLI uses real provider adapters only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run Locally

Set API credentials and choose an explicit model:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m yolorag.cli "What is YOLO and should we retrieve docs?" \
  --provider openai \
  --mode fast
```

Deep mode triggers richer orchestration. DeepSeek uses its OpenAI-compatible API:

```bash
export DEEPSEEK_API_KEY=...
PYTHONPATH=src python3 -m yolorag.cli "Debug a YOLO export error" \
  --provider deepseek \
  --mode deep
```

## Model Defaults

The CLI picks models by provider and mode. The override order is:

1. `--model`
2. Mode-specific env vars, such as `YOLORAG_OPENAI_FAST_MODEL`
3. `models.json`, or another file passed with `--models-config`
4. Built-in defaults

| Provider | Fast mode | Deep/thinking mode |
| --- | --- | --- |
| OpenAI | `gpt-5.4-mini` | `gpt-5.5` |
| DeepSeek | `deepseek-v4-flash` | `deepseek-v4-pro` |

Swap models by editing `models.json`:

```json
{
  "openai": {
    "fast": { "model": "gpt-5.4-mini" },
    "deep": { "model": "gpt-5.5" }
  },
  "deepseek": {
    "fast": { "model": "deepseek-v4-flash" },
    "deep": { "model": "deepseek-v4-pro" }
  }
}
```

Or use environment overrides:

```env
YOLORAG_OPENAI_FAST_MODEL=gpt-5.4-mini
YOLORAG_OPENAI_THINKING_MODEL=gpt-5.5
YOLORAG_DEEPSEEK_FAST_MODEL=deepseek-v4-flash
YOLORAG_DEEPSEEK_THINKING_MODEL=deepseek-v4-pro
```

To try an alternate profile:

```bash
PYTHONPATH=src python3 -m yolorag.cli "Debug a YOLO export error" \
  --provider deepseek \
  --mode deep \
  --models-config ./models.experimental.json
```

DeepSeek fast mode explicitly disables thinking. DeepSeek deep mode enables thinking with high reasoning effort.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Architecture

`providers/` owns model API calls and normalizes responses into `LLMResponse`.

`usage/` owns provider-specific token extraction and cost calculation.

`retrieval/` owns knowledge-source access.

`review/` owns answer verification and confidence scoring.

`core/` owns routing, conversation state, orchestration, and traces.

The orchestrator should not read raw provider response shapes directly. OpenAI and DeepSeek provider adapters normalize those details before returning.
