"""RAG-Stack component aliases for GenZ cost-model simulation.

LLM architecture metadata and architecture-identical checkpoint aliases belong
to ``GenZ.Models.ModelConfig``.  This map is only for rag-stack component names
that GenZ cannot know about (embedding/reranker components, API shorthands).
Value can be:
  - str: GenZ built-in model name (e.g. "Qwen/Qwen3-8B")
  - ModelConfig: custom model architecture (e.g. SBERT for reranker)

Add a new LLM or checkpoint alias in GenZ; add only rag-stack-specific
component aliases below.
"""

import logging
from typing import Union

from GenZ.Models.default_models import ModelConfig

logger = logging.getLogger("RAG-Stack")

# ---------------------------------------------------------------------------
# Unified GenZ model map
# Key: lowercase API/config model name
# Value: str (GenZ built-in name) or ModelConfig (custom architecture)
# ---------------------------------------------------------------------------
GENZ_MODEL_MAP: dict[str, Union[str, ModelConfig]] = {
    # ── OpenAI ──
    "gpt-4o-mini": "openai/GPT-4",
    "gpt-4o": "openai/GPT-4",
    "gpt-4": "openai/GPT-4",
    "gpt-3.5-turbo": "openai/gpt2",
    # ── Llama (ollama short names) ──
    "llama3.2": "meta-llama/Llama-3.2-3B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama-3-8b": "llama3_8b",
    "llama-2-7b": "llama2_7b",
    "llama-2-13b": "llama2_13b",
    "llama-2-70b": "llama2_70b",
    # ── Mistral ──
    "mistral-7b": "mistral_7b",
    # ── Qwen ──
    # Qwen2.5 base/Instruct aliases are architecture metadata owned by GenZ's
    # ModelConfig registry.  Do not duplicate them here: resolve_genz_model()
    # consults that registry before this rag-stack-only component alias map.
    "qwen/qwen3-8b": "Qwen/Qwen3-8B",
    "qwen/qwen3-14b": "Qwen/Qwen3-14B",
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    # ── Google ──
    "google/gemma-3-4b-it": "google/gemma-3-4B",
    "google/gemma-3-27b-it": "google/gemma-3-27B",
    # ── Encoder / Reranker models (custom ModelConfig) ──
    "sbert_120m": ModelConfig(
        model="sbert_120M",
        hidden_size=384,
        num_attention_heads=12,
        num_ffi=2,
        intermediate_size=1536,
        num_decoder_layers=12,
        vocab_size=50272,
        hidden_act="relu",
        max_model_len=2 * 1024,
    ),
    # NOTE on vocab_size for cross-encoder rerankers:
    # GenZ appends a "classifier" layer at the end that does
    #   (batch, seq, hidden) × (hidden, vocab_size) → (batch, seq, vocab_size)
    # which is correct for LLMs with an LM head, but WRONG for cross-encoder
    # classifiers that only output a single scalar score per (query, passage).
    # For these models we set vocab_size=1 so the classifier contributes
    # negligible FLOPS. The real (tokenizer-side) vocab is irrelevant to
    # the forward-pass compute; input embedding is a hash lookup, not a
    # matmul. Real HF tokenizer vocab kept as a comment for reference.
    #
    # BAAI/bge-reranker-large — XLM-RoBERTa-based, real vocab=250002.
    # Verified architecture against HF config.json.
    "bge_reranker_large": ModelConfig(
        model="bge_reranker_large",
        hidden_size=1024,
        num_attention_heads=16,
        num_ffi=1,
        intermediate_size=4096,
        num_decoder_layers=24,
        vocab_size=1,          # cross-encoder: only a scalar classifier head
        hidden_act="gelu",
        max_model_len=512,
    ),
    # cross-encoder/ms-marco-MiniLM-L-2-v2 — 2-layer BERT, real vocab=30522.
    "ms_marco_minilm_l2": ModelConfig(
        model="ms_marco_minilm_l2",
        hidden_size=384,
        num_attention_heads=12,
        num_ffi=1,
        intermediate_size=1536,
        num_decoder_layers=2,
        vocab_size=1,          # cross-encoder: only a scalar classifier head
        hidden_act="gelu",
        max_model_len=512,
    ),
    # colbert-ir/colbertv2.0 — BERT-base backbone, real vocab=30522.
    # Outputs per-token hidden states (no classifier at all); MaxSim
    # scoring runs in numpy post-processing, not in the torch graph.
    "colbertv2": ModelConfig(
        model="colbertv2",
        hidden_size=768,
        num_attention_heads=12,
        num_ffi=1,
        intermediate_size=3072,
        num_decoder_layers=12,
        vocab_size=1,          # no classifier, MaxSim is numpy post-processing
        hidden_act="gelu",
        max_model_len=512,
    ),
    # microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank is a
    # BERT-base token classifier.  Strict pricing still uses the dedicated
    # aggregate CPU+GPU LLMLingua2 profile because the production wrapper is
    # mostly host/Python work.  This architecture entry is used only by the
    # explicit uncalibrated_spec hardware-roofline lower bound when that exact
    # aggregate profile is absent; it is not a fitted replacement profile.
    "llmlingua2_bert_base": ModelConfig(
        model="llmlingua2_bert_base_multilingual_cased",
        hidden_size=768,
        num_attention_heads=12,
        num_ffi=1,
        intermediate_size=3072,
        num_decoder_layers=12,
        vocab_size=2,          # per-token keep/drop classifier
        hidden_act="gelu",
        max_model_len=512,
    ),
    # ── Retrieval embedding (bi-encoder) models ──
    # Consumed by the `encode` (query-embedding) cost-model stage
    # (rag_stack/cost_model/retrieval_encode.py). Keyed by the rag-stack
    # embedding registry name (static_rag_evaluator/embedding/base.py) — what
    # vectordb configs carry in ``embedding_model:``. Pooled-vector output, no
    # LM head → vocab_size=1 (same rationale as the cross-encoders above).
    #
    # sentence-transformers/all-mpnet-base-v2 — MPNet-base: 12 layers, hidden
    # 768, heads 12, FFN 3072 (plain GELU, num_ffi=1), ~110M params, real
    # vocab=30527, max position 512. Verified against the HF config.json.
    "huggingface_all_mpnet_base_v2": ModelConfig(
        model="all_mpnet_base_v2",
        hidden_size=768,
        num_attention_heads=12,
        num_ffi=1,
        intermediate_size=3072,
        num_decoder_layers=12,
        vocab_size=1,          # bi-encoder: pooled embedding, no LM head
        hidden_act="gelu",
        max_model_len=512,
    ),
    # BAAI/bge-small-en-v1.5 — BERT-small backbone: 12 layers, hidden 384,
    # heads 12, FFN 1536 (plain GELU, num_ffi=1), ~33M params, real
    # vocab=30522, max position 512. Bi-encoder → pooled embedding, vocab=1.
    "huggingface_baai_bge_small": ModelConfig(
        model="bge_small_en_v1_5",
        hidden_size=384,
        num_attention_heads=12,
        num_ffi=1,
        intermediate_size=1536,
        num_decoder_layers=12,
        vocab_size=1,          # bi-encoder: pooled embedding, no LM head
        hidden_act="gelu",
        max_model_len=512,
    ),
    # BAAI/bge-m3 — XLM-RoBERTa-large backbone (same arch family as
    # bge_reranker_large): 24 layers, hidden 1024, heads 16, FFN 4096 (GELU,
    # num_ffi=1), ~560M params, real vocab=250002, max position 8194 (8192
    # usable). Bi-encoder dense output → pooled embedding, vocab=1.
    "huggingface_bge_m3": ModelConfig(
        model="bge_m3",
        hidden_size=1024,
        num_attention_heads=16,
        num_ffi=1,
        intermediate_size=4096,
        num_decoder_layers=24,
        vocab_size=1,          # bi-encoder: pooled embedding, no LM head
        hidden_act="gelu",
        max_model_len=8192,
    ),
    # ── Reranker default models ──
    "bge-reranker-v2-gemma": "gemma2_2b",       # flag_embedding_llm_reranker (~2B gemma)
    # "bge-reranker-large" → use "bge_reranker_large" key directly (avoid double lookup)
    "t5-large": ModelConfig(                       # t5-large (~770M, used by UPR)
        model="t5_large",
        hidden_size=1024,
        num_attention_heads=16,
        num_ffi=2,
        intermediate_size=2816,
        num_decoder_layers=24,
        num_encoder_layers=24,
        vocab_size=32128,
        hidden_act="relu",
        max_model_len=512,
    ),
    # castorini/monot5-3b-msmarco-10k — T5-3B encoder-decoder (~2.85B params).
    # Verified against the HF config.json on the hub:
    #   d_model=1024, d_ff=16384, num_heads=32, num_layers=24 (encoder
    #   and decoder each), d_kv=128, feed_forward_proj="relu",
    #   vocab_size=32128.
    # MonoT5 scores each (query, passage) via a full encoder forward + 1
    # decoder step reading logits over ("▁false", "▁true"). The decoder
    # contribution is negligible at realistic seq_len (~0.2% of encoder
    # FLOPS for seq=512), so PassageRerankerCostModel models this as prefill-
    # only; the 24 encoder layers go into num_decoder_layers because GenZ
    # treats that field as "transformer layers computed in prefill".
    # The num_encoder_layers field is kept informational (GenZ ignores it).
    # T5 uses a plain ReLU FFN (x → W1 → ReLU → W2), not gated —
    # num_ffi=1. GPT-style gated variants use num_ffi=2.
    "monot5-3b": ModelConfig(
        model="monot5_3b",
        hidden_size=1024,
        num_attention_heads=32,
        num_ffi=1,
        intermediate_size=16384,
        num_decoder_layers=24,
        num_encoder_layers=24,
        vocab_size=32128,
        hidden_act="relu",
        max_model_len=512,
    ),
    # facebook/tart-full-flan-t5-xl — TART reranker built on the encoder
    # stack of an EncT5ForSequenceClassification model. The HF repo is
    # named "*-flan-t5-xl" but the actual checkpoint is **Flan-T5-Large
    # sized** (verified by inspecting `model.config` on the loaded weights:
    # d_model=1024, d_ff=2816, num_heads=16, total 341M params — NOT
    # 3B-XL specs). TART only runs ONE encoder forward (no decoder, no
    # autoregressive step). GenZ treats ``num_decoder_layers`` as
    # "transformer layers to compute in prefill" regardless of whether
    # the original model is encoder or decoder, so we put the 24 encoder
    # layers into num_decoder_layers. The num_encoder_layers field is
    # kept at 0 because GenZ never reads it.
    # Verified loaded config:
    #   d_model=1024, d_ff=2816, num_heads=16, num_layers=24,
    #   feed_forward_proj="gated-gelu", vocab_size=32128, params=0.341B.
    # gated-gelu has two parallel up projections (gate + up), so num_ffi=2.
    "tart_flan_t5_xl": ModelConfig(
        model="tart_flan_t5_xl",
        hidden_size=1024,
        num_attention_heads=16,
        num_ffi=2,                      # gated-gelu: gate + up = 2 parallel ups
        intermediate_size=2816,
        num_decoder_layers=24,          # 24 encoder layers → modeled as prefill
        num_encoder_layers=0,           # unused field; keep 0 to avoid misleading
        vocab_size=32128,
        hidden_act="gelu",
        max_model_len=1024,
    ),
}


def resolve_genz_model(model_name: str) -> Union[str, ModelConfig]:
    """Resolve API model name to GenZ model (str or ModelConfig).

    Tries GenZ's model/alias registry first, then rag-stack component aliases,
    then strips the historical ``openai/`` prefix.
    Returns as-is (str) if not found in map.
    """
    from GenZ.Models.default_models import MODEL_DICT

    registered = MODEL_DICT.get_model(model_name)
    if registered is not None:
        return registered.model

    key = model_name.lower()
    mapped = GENZ_MODEL_MAP.get(key)
    if mapped:
        return mapped
    if key.startswith("openai/"):
        mapped = GENZ_MODEL_MAP.get(key[len("openai/"):])
        if mapped:
            return mapped
    logger.warning(
        "Model %r is absent from the GenZ registry and rag-stack alias map; "
        "using it as-is.",
        model_name,
    )
    return model_name
