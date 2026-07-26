"""Unified model identity mappings (top-level; no registry package).

Single source of truth for mapping API/OpenRouter model names to
HuggingFace repo IDs (tokenizer loading) and context lengths (prompt
truncation). GenZ simulator model names live next to their consumer in
``rag_stack.cost_model.llm_sim.genz_model_map``; the component → sub-cost-
model bindings are code in ``rag_stack.cost_model.dispatch``.

To add a new model, add an entry to each relevant dict below.
"""

import logging

logger = logging.getLogger("RAG-Stack")

# ---------------------------------------------------------------------------
# API/OpenRouter name → HuggingFace repo (for tokenizer loading)
# ---------------------------------------------------------------------------
TOKENIZER_MAP = {
    # OpenAI — tiktoken handles these directly, but map anyway for AutoTokenizer fallback
    "gpt-4o-mini": "Xenova/gpt-4o",
    "gpt-4o": "Xenova/gpt-4o",
    "gpt-4": "Xenova/gpt-4",
    "gpt-3.5-turbo": "Xenova/gpt-3.5-turbo",
    # DeepSeek
    "deepseek/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "deepseek/deepseek-r1": "deepseek-ai/DeepSeek-R1",
    # Qwen
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-14b": "Qwen/Qwen3-14B",
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "qwen/qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen/qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    # Google
    "google/gemma-3-4b-it": "google/gemma-3-4b-it",
    "google/gemma-3-27b-it": "google/gemma-3-27b-it",
    # Llama
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    # Ollama short names
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
}

# ---------------------------------------------------------------------------
# Model → max context length in tokens (for prompt truncation in OpenAILLM)
# ---------------------------------------------------------------------------
CONTEXT_LENGTH_MAP = {
    # GPT-5.x
    "gpt-5.1-2025-11-13": 272_000,
    "gpt-5.1": 272_000,
    "gpt-5": 272_000,
    "gpt-5-pro": 272_000,
    "gpt-5-2025-08-07": 272_000,
    "gpt-5-chat-latest": 272_000,
    "gpt-5-mini-2025-08-07": 272_000,
    "gpt-5-mini": 272_000,
    "gpt-5-nano-2025-08-07": 272_000,
    # GPT-4.1
    "gpt-4.1": 1_000_000,
    "gpt-4.1-2025-04-14": 1_000_000,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-mini-2025-04-14": 1_047_576,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4.1-nano-2025-04-14": 1_000_000,
    # Reasoning models
    "o1": 200_000,
    "o1-preview": 128_000,
    "o1-preview-2024-09-12": 128_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "o1-pro": 200_000,
    "o1-pro-2025-03-19": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o3-mini-2025-01-31": 200_000,
    "o4-mini": 200_000,
    "o4-mini-2025-04-16": 200_000,
    # GPT-4o
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "chatgpt-4o-latest": 128_000,
    # GPT-4
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    "gpt-4-1106-preview": 128_000,
    "gpt-4-vision-preview": 128_000,
    "gpt-4-1106-vision-preview": 128_000,
    "gpt-4": 8_192,
    "gpt-4-0613": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-32k-0613": 32_768,
    # GPT-3.5
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-3.5-turbo-instruct": 4_096,
    "gpt-3.5-turbo-16k": 16_385,
    "gpt-3.5-turbo-0613": 4_096,
    "gpt-3.5-turbo-16k-0613": 16_385,
    # OpenRouter / open-source models
    "deepseek/deepseek-v3.2": 128_000,
    "deepseek/deepseek-r1": 128_000,
    "qwen/qwen3-8b": 128_000,
    "qwen/qwen3-14b": 128_000,
    "qwen/qwen3-32b": 128_000,
    "google/gemma-3-4b-it": 128_000,
    "google/gemma-3-27b-it": 128_000,
}

_DEFAULT_CONTEXT_LENGTH = 128_000

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def resolve_tokenizer_name(model_name: str) -> str:
    """Resolve API model name to HuggingFace repo for tokenizer loading."""
    key = model_name.lower()
    mapped = TOKENIZER_MAP.get(key)
    if mapped:
        return mapped
    # Try stripping openai/ prefix (openai_llm prepends it)
    if key.startswith("openai/"):
        mapped = TOKENIZER_MAP.get(key[len("openai/"):])
        if mapped:
            return mapped
    # Return as-is — may work if it's already a valid HF repo
    return model_name



def get_context_length(model_name: str, default: int = _DEFAULT_CONTEXT_LENGTH) -> int:
    """Get max context length for a model. Returns default (128k) if unknown."""
    length = CONTEXT_LENGTH_MAP.get(model_name)
    if length is not None:
        return length
    logger.warning(
        f"Model '{model_name}' not in CONTEXT_LENGTH_MAP, "
        f"defaulting to {default:,} tokens."
    )
    return default
