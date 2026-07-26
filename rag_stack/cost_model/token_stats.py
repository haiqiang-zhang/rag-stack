"""Centralized token statistics for cost model estimation.

Computes base token stats from QA/corpus data and derives per-stage
token parameters used by GenZ simulation cost models.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger("RAG-Stack")


class TokenStats:
    """Token statistics computed from dataset and pipeline config.

    Base stats (computed from data):
        avg_chunk_tokens, avg_query_tokens, avg_output_tokens, prompt_template_tokens

    Per-stage derived params (for cost model _get_params):
        get_llm_input_tokens, get_query_expansion_input_tokens, etc.
    """

    def __init__(
        self,
        qa_data: pd.DataFrame,
        corpus_data: pd.DataFrame,
        config: dict,
    ):
        tokenizer = self._load_tokenizer(config)

        # avg_chunk_tokens
        self.avg_chunk_tokens: int = self._compute_avg_chunk_tokens(
            corpus_data, tokenizer,
        )

        # Chunk-length DISTRIBUTION (r26): heavy rerankers run every forward
        # at the PADDED sequence length (the max pair in the padding group),
        # so the CM needs order statistics of the chunk token counts, not
        # just their mean. Its own (larger) sample: a padding group of B*k
        # pairs is set by the ~1/(B*k) upper tail, which a mean-sized sample
        # cannot resolve. avg_chunk_tokens keeps its original 1k sample so
        # every existing mean-shape price is bit-identical.
        self.chunk_token_quantiles: list | None = (
            self._compute_chunk_token_quantiles(corpus_data, tokenizer)
        )

        # prompt_template_tokens
        prompt_template = self._find_prompt_template(config)
        fixed_text = re.sub(r"\{[^}]+\}", "", prompt_template)
        self.prompt_template_tokens: int = len(tokenizer.encode(fixed_text))

        # avg_output_tokens
        self.avg_output_tokens: int = int(
            qa_data["generation_gt"].apply(
                lambda x: len(tokenizer.encode(
                    x[0] if hasattr(x, '__len__') and not isinstance(x, str) else x
                ))
            ).mean()
        )

        # avg_query_tokens
        self.avg_query_tokens: int = int(
            qa_data["query"].apply(lambda t: len(tokenizer.encode(t))).mean()
        )

        # bytes_per_token: dataset-derived UTF-8 bytes per token for THIS
        # dataset under THIS tokenizer (English ~4, CJK ~1.5-3, code ~3).
        # Consumed by the communication model to convert token-counted text
        # payloads into bytes — never assume a language.
        self.bytes_per_token: float = self._compute_bytes_per_token(
            qa_data, corpus_data, tokenizer,
        )

        # Compressor-shape stats: llmlingua2's runtime splits every passage
        # on "\n\n" and pays per ELEM (nodes/passagecompressor/llmlingua2.py
        # line 232/266 — no dedup, no empty-drop), so the CM must price
        # elems, not whole passages (pricing passages-as-chunks was 3.4x
        # optimistic on compressor-bound points — r12). Pure corpus
        # statistics, same family as avg_chunk_tokens; only computed when
        # the config can select llmlingua2 (BERT-tokenizer cost is skipped
        # for every other pipeline).
        self.avg_compressor_elems_per_chunk: float | None = None
        self.avg_compressor_elem_tokens: int | None = None
        self._compressor_tokenizer = None
        if self._config_has_llmlingua2(config):
            self._compressor_tokenizer = self._load_compressor_tokenizer()
            if self._compressor_tokenizer is not None:
                (
                    self.avg_compressor_elems_per_chunk,
                    self.avg_compressor_elem_tokens,
                ) = self._compute_compressor_elem_stats(
                    corpus_data, self._compressor_tokenizer,
                )

        # Keep the tokenizer around so we can recompute avg_chunk_tokens
        # when the chunker config changes mid-sweep.
        self._tokenizer = tokenizer
        self.avg_llm_input_tokens: int | None = None
        self.avg_query_expansion_input_tokens: int | None = None
        self.avg_query_expansion_output_tokens: int | None = None
        self.avg_passage_reranker_input_tokens: int | None = None

        logger.info(
            f"Auto-computed token stats: avg_chunk_tokens={self.avg_chunk_tokens}, "
            f"prompt_template_tokens={self.prompt_template_tokens}, "
            f"avg_output_tokens={self.avg_output_tokens}, "
            f"avg_query_tokens={self.avg_query_tokens}"
        )

    # ------------------------------------------------------------------
    # Tokenizer / chunk-stats helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tokenizer(config: dict):
        """Load the generator's tokenizer, falling back to gpt2 on failure."""
        from transformers import AutoTokenizer
        from rag_stack.model_map import resolve_tokenizer_name

        model_name = TokenStats._find_generator_model(config)
        hf_name = resolve_tokenizer_name(model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            logger.info(f"Loaded tokenizer from '{hf_name}' for token stats")
        except (OSError, ValueError):
            logger.warning(
                f"Cannot load tokenizer for '{hf_name}'. "
                f"Falling back to 'gpt2' tokenizer for token stats."
            )
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tokenizer

    @staticmethod
    def _compute_bytes_per_token(
        qa_data: pd.DataFrame, corpus_data: pd.DataFrame, tokenizer
    ) -> float:
        """UTF-8 bytes per token over the corpus chunks + queries."""
        sample = corpus_data["contents"]
        if len(sample) > 1000:
            sample = sample.sample(1000, random_state=42)
        texts = [t for t in list(sample) + list(qa_data["query"]) if isinstance(t, str)]
        total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
        if total_tokens <= 0:
            return 4.0  # empty dataset — English UTF-8 average as last resort
        return sum(len(t.encode("utf-8")) for t in texts) / total_tokens

    @staticmethod
    def _chunk_token_counts(
        corpus_data: pd.DataFrame, tokenizer, n_sample: int = 1000
    ) -> list:
        """Token counts of up to ``n_sample`` sampled corpus chunks."""
        sample = corpus_data["contents"]
        if len(sample) > n_sample:
            sample = sample.sample(n_sample, random_state=42)
        return sample.apply(lambda t: len(tokenizer.encode(t))).tolist()

    # Quantile grid for the chunk-length distribution: uniform percentiles
    # plus a REFINED upper tail (P99.1..P99.9, P99.95, P99.99, P100). The
    # padding excess integrates ``Q(p) d(p^k)``, whose mass for a group of k
    # pairs sits at p ≈ 1-1/k; a uniform grid both truncates the tail for
    # large k (top-bin midpoint) and over-weights it for small k (one fat
    # 1%-bin interpolated linearly across a skewed tail). The grid is a
    # RESOLUTION choice of a measured distribution — not a fitted value.
    _QUANTILE_GRID_P: list = (
        [i / 100 for i in range(100)]
        + [0.991 + i / 1000 for i in range(9)]
        + [0.9995, 0.9999, 1.0]
    )

    @classmethod
    def _quantiles_from_counts(cls, counts: list) -> list | None:
        """Self-describing empirical quantile grid ``[[p, q], ...]`` of the
        chunk token counts — the order-statistics helpers integrate over it.
        Pure corpus statistics, same family as ``avg_chunk_tokens``."""
        if not counts:
            return None
        import numpy as np
        qs = np.quantile(
            np.asarray(counts, dtype=float), np.asarray(cls._QUANTILE_GRID_P)
        )
        return [[float(p), int(round(v))] for p, v in zip(cls._QUANTILE_GRID_P, qs)]

    # Sample size for the quantile grid: 10k draws resolve ~P99.99 — enough
    # for the largest real padding group (request batch x fanout ~ 10^3..10^4).
    # Statistical fidelity of a measured distribution, not a model constant.
    _QUANTILE_SAMPLE_ROWS = 10_000

    @classmethod
    def _compute_chunk_token_quantiles(
        cls, corpus_data: pd.DataFrame, tokenizer
    ) -> list | None:
        return cls._quantiles_from_counts(
            cls._chunk_token_counts(
                corpus_data, tokenizer, n_sample=cls._QUANTILE_SAMPLE_ROWS,
            )
        )

    @staticmethod
    def _compute_avg_chunk_tokens(corpus_data: pd.DataFrame, tokenizer) -> int:
        """Sample up to 1k rows from corpus and average their token counts."""
        counts = TokenStats._chunk_token_counts(corpus_data, tokenizer)
        if not counts:
            return 0
        return int(sum(counts) / len(counts))

    @staticmethod
    def _config_has_llmlingua2(config: dict) -> bool:
        for _block, node_line in TokenStats._iter_node_lines(config):
            for node in node_line.get("nodes", []) or []:
                if node.get("stage") != "passage_compressor":
                    continue
                for module in node.get("modules", []) or []:
                    if str(module.get("component") or "") == "llmlingua2":
                        return True
        return False

    @staticmethod
    def _load_compressor_tokenizer():
        """llmlingua2's own BERT tokenizer — the per-token coefficient of the
        compressor cost model is calibrated in ITS token units, not the
        generator tokenizer's."""
        from transformers import AutoTokenizer
        name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        try:
            return AutoTokenizer.from_pretrained(name)
        except (OSError, ValueError) as e:
            logger.warning(
                f"Cannot load compressor tokenizer '{name}' ({e}); "
                "compressor elem stats unavailable — CM falls back to "
                "passage-as-chunk pricing."
            )
            return None

    @staticmethod
    def _compute_compressor_elem_stats(
        corpus_data: pd.DataFrame, tokenizer
    ) -> tuple:
        """Replicate llmlingua2's runtime batch shape on the chunked corpus:
        each chunk splits on "\\n\\n" into elems (empties kept — the runtime
        pays per-elem overhead for them too); an elem longer than the BERT
        window (510) is internally re-chunked into ceil(tokens/510) units.
        Returns (avg units per corpus chunk, avg BERT tokens per unit)."""
        import math
        sample = corpus_data["contents"]
        if len(sample) > 1000:
            sample = sample.sample(1000, random_state=42)
        n_chunks = 0
        total_units = 0
        total_tokens = 0
        for text in sample:
            if not isinstance(text, str):
                continue
            n_chunks += 1
            for elem in text.split("\n\n"):
                n_tok = len(tokenizer.encode(elem, add_special_tokens=False))
                total_units += max(1, math.ceil(n_tok / 510))
                total_tokens += n_tok
        if n_chunks == 0 or total_units == 0:
            return None, None
        avg_units = total_units / n_chunks
        avg_tok = int(min(510, round(total_tokens / total_units)))
        return avg_units, max(1, avg_tok)

    @classmethod
    def inject_compressor_elem_stats(
        cls,
        pipeline_config: dict,
        raw_corpus_path: str,
        memo: dict | None = None,
    ) -> bool:
        """Offline-repricing helper: stored ``_token_stats`` dicts predate
        the compressor elem stats, so CM repricing of archived evals would
        silently fall back to the (3.4x-optimistic) passage-as-chunk shape.
        Recompute the stats from the eval's resolved chunker
        (``corpus_runtime.chunker``) + the raw corpus and inject them into
        the config's ``_token_stats`` dict. No-op when already present, no
        llmlingua2 in the config, or no resolved chunker. Pass a shared
        ``memo`` dict to amortize the chunk+tokenize across evals.
        Returns True when stats were injected."""
        ts_dict = pipeline_config.get("_token_stats")
        if not isinstance(ts_dict, dict):
            return False
        if ts_dict.get("avg_compressor_elems_per_chunk") is not None:
            return False
        if not cls._config_has_llmlingua2(pipeline_config):
            return False
        chunker = (
            (pipeline_config.get("corpus_runtime") or {}).get("chunker") or {}
        )
        method = chunker.get("component")
        size = chunker.get("chunk_size")
        if not method or size is None:
            return False
        overlap = int(chunker.get("chunk_overlap") or 0)
        key = (str(method), int(size), overlap, str(raw_corpus_path))
        memo = memo if memo is not None else {}
        if key not in memo:
            from rag_stack_evaluator.static_rag_evaluator.chunk_cache import _run_chunker
            raw_df = pd.read_parquet(raw_corpus_path)
            if "contents" not in raw_df.columns and "texts" in raw_df.columns:
                raw_df = raw_df.rename(columns={"texts": "contents"})
            chunked = _run_chunker(raw_df, str(method), int(size), overlap)
            tok = cls._load_compressor_tokenizer()
            if tok is None:
                return False
            memo[key] = cls._compute_compressor_elem_stats(chunked, tok)
        epc, ept = memo[key]
        if epc is None:
            return False
        ts_dict["avg_compressor_elems_per_chunk"] = epc
        ts_dict["avg_compressor_elem_tokens"] = ept
        return True

    def with_chunked_corpus(self, corpus_data: pd.DataFrame) -> "TokenStats":
        """Return a new TokenStats with avg_chunk_tokens recomputed for a
        different chunked corpus, preserving query/output/prompt-template
        stats (those depend on QA data + config, not on chunking).

        Requires the original tokenizer to still be attached to ``self``
        (i.e. this TokenStats was built via ``__init__``, not ``from_dict``).
        """
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            raise RuntimeError(
                "with_chunked_corpus requires a TokenStats built via __init__ "
                "(tokenizer is not preserved through from_dict round-trips)."
            )
        new = object.__new__(TokenStats)
        new.avg_chunk_tokens = self._compute_avg_chunk_tokens(corpus_data, self._tokenizer)
        new.chunk_token_quantiles = self._compute_chunk_token_quantiles(
            corpus_data, self._tokenizer,
        )
        # bytes/token is a language/tokenizer property — chunk size doesn't
        # change it, so the original dataset's ratio carries over.
        new.bytes_per_token = getattr(self, "bytes_per_token", 4.0)
        # Compressor elem stats are chunker-DEPENDENT (a 2048-char chunk
        # holds ~2x the "\n\n" elems of a 1024-char chunk) — recompute for
        # the new corpus whenever the original TokenStats carried them.
        new._compressor_tokenizer = getattr(self, "_compressor_tokenizer", None)
        if new._compressor_tokenizer is not None:
            (
                new.avg_compressor_elems_per_chunk,
                new.avg_compressor_elem_tokens,
            ) = self._compute_compressor_elem_stats(
                corpus_data, new._compressor_tokenizer,
            )
        else:
            new.avg_compressor_elems_per_chunk = getattr(
                self, "avg_compressor_elems_per_chunk", None
            )
            new.avg_compressor_elem_tokens = getattr(
                self, "avg_compressor_elem_tokens", None
            )
        new.avg_query_tokens = self.avg_query_tokens
        new.avg_output_tokens = self.avg_output_tokens
        new.prompt_template_tokens = self.prompt_template_tokens
        new.avg_llm_input_tokens = getattr(self, "avg_llm_input_tokens", None)
        new.avg_query_expansion_input_tokens = getattr(
            self, "avg_query_expansion_input_tokens", None
        )
        new.avg_query_expansion_output_tokens = getattr(
            self, "avg_query_expansion_output_tokens", None
        )
        new.avg_passage_reranker_input_tokens = getattr(
            self, "avg_passage_reranker_input_tokens", None
        )
        new._tokenizer = self._tokenizer
        return new

    # ------------------------------------------------------------------
    # Per-stage derived params
    # ------------------------------------------------------------------

    def get_llm_input_tokens(self, top_k: int) -> int:
        """Main LLM prefill input = prompt template + user query + top_k passages.

        The prompt_maker fstring substitutes BOTH ``{query}`` and
        ``{retrieved_contents}`` into the template, so the query's tokens are
        part of every real prefill (they were silently dropped before —
        a ~25% under-count on small-context configs).
        """
        override = getattr(self, "avg_llm_input_tokens", None)
        if override is not None:
            return int(override)
        return (
            self.prompt_template_tokens
            + self.avg_query_tokens
            + top_k * self.avg_chunk_tokens
        )

    def get_query_expansion_input_tokens(self) -> int:
        """Query expansion input = user query."""
        override = getattr(self, "avg_query_expansion_input_tokens", None)
        if override is not None:
            return int(override)
        return self.avg_query_tokens

    # How many query-length generations each expansion method emits. HyDE writes
    # one passage (handled separately). multi_query_expansion's default prompt asks
    # for "3 different versions"; query_decompose breaks a question into a handful
    # of self-contained sub-questions. Modeling these as ONE query (the old bug)
    # under-counted their decode work N-fold and made query-expansion look ~free.
    _QUERY_EXPANSION_FANOUT = {
        "multi_query_expansion": 3,   # default prompt: generate 3 versions
        "query_decompose": 3,         # typical sub-question count (variable)
    }

    def get_query_expansion_output_tokens(self, component: str) -> int:
        """Tokens generated by a query-expansion method.

        HyDE generates a passage (chunk-length). multi_query_expansion /
        query_decompose each emit MULTIPLE query-length generations, so the
        output is ``fanout * avg_query_tokens`` — not one query. (Per-generation
        length is approximated by the query length; a verbose generator may write
        longer reformulations, the same generation-length approximation used for
        ``avg_output_tokens``.)
        """
        override = getattr(self, "avg_query_expansion_output_tokens", None)
        if override is not None:
            return int(override)
        if component == "hyde":
            return self.avg_chunk_tokens
        fanout = self._QUERY_EXPANSION_FANOUT.get(component, 1)
        return fanout * self.avg_query_tokens

    def get_passage_reranker_input_tokens(self) -> int:
        """Cross-encoder passage reranker input = query + passage concatenated."""
        override = getattr(self, "avg_passage_reranker_input_tokens", None)
        if override is not None:
            return int(override)
        return self.avg_query_tokens + self.avg_chunk_tokens

    # ------------------------------------------------------------------
    # Padded-shape order statistics (r26)
    # ------------------------------------------------------------------

    @staticmethod
    def _order_stat_mean(quantile_grid: list, k: float) -> float:
        """E[max of k iid draws] from the empirical quantile grid
        (``[[p, q], ...]``, p ascending, p[0]=0, p[-1]=1).

        With Q the quantile function, ``E[max_k] = ∫ Q(p) d(p^k)``. Q is
        interpolated LINEARLY within each grid bin and the integral is taken
        exactly:

            ∫[pL,pU] (a + m·(p - pL)) d(p^k)
              = (a - m·pL)·(pU^k - pL^k) + m·k/(k+1)·(pU^{k+1} - pL^{k+1})

        (a = Q(pL), m = bin slope). Pure order statistics on the stored
        distribution — no fitted values; ``k=1`` reproduces the trapezoid
        sample mean, so differences against ``k=1`` cancel the grid bias, and
        the refined tail bins give the large-k mass the resolution it needs.
        """
        if not quantile_grid:
            return 0.0
        if len(quantile_grid) == 1:
            return float(quantile_grid[0][1])
        total = 0.0
        for (p_lo, q_lo), (p_hi, q_hi) in zip(quantile_grid, quantile_grid[1:]):
            if p_hi <= p_lo:
                continue
            a = float(q_lo)
            m = (float(q_hi) - a) / (p_hi - p_lo)
            d_cdf = p_hi ** k - p_lo ** k
            d_next = p_hi ** (k + 1) - p_lo ** (k + 1)
            total += (a - m * p_lo) * d_cdf + m * (k / (k + 1)) * d_next
        return total

    def chunk_pad_excess_tokens(self, group_pairs: int) -> int | None:
        """Expected padding excess per pair when ``group_pairs`` pairs are
        padded together: ``E[max over group] - E[chunk]`` of the corpus chunk
        token distribution. The query side of a pair is (near-)constant
        within a call, so the chunk length carries the padding variability.

        Returns ``None`` when the distribution is unavailable (legacy
        ``_token_stats`` dicts) — callers keep their pre-r26 behavior.
        """
        q = getattr(self, "chunk_token_quantiles", None)
        if not q or len(q) < 2:
            return None
        k = max(1, int(group_pairs))
        if k == 1:
            return 0
        excess = self._order_stat_mean(q, float(k)) - self._order_stat_mean(q, 1.0)
        return max(0, int(round(excess)))

    @classmethod
    def inject_chunk_token_quantiles(
        cls,
        pipeline_config: dict,
        raw_corpus_path: str,
        memo: dict | None = None,
    ) -> bool:
        """Offline-repricing helper (same family as
        :meth:`inject_compressor_elem_stats`): stored ``_token_stats`` dicts
        predate the chunk-length distribution, so CM repricing of archived
        evals would silently fall back to mean-shape reranker pricing.
        Recompute the percentile grid from the eval's resolved chunker
        (``corpus_runtime.chunker``) + the raw corpus and inject it. No-op
        when already present or no resolved chunker. Pass a shared ``memo``
        dict to amortize the chunk+tokenize across evals. Returns True when
        stats were injected."""
        ts_dict = pipeline_config.get("_token_stats")
        if not isinstance(ts_dict, dict):
            return False
        if ts_dict.get("chunk_token_quantiles") is not None:
            return False
        chunker = (
            (pipeline_config.get("corpus_runtime") or {}).get("chunker") or {}
        )
        method = chunker.get("component")
        size = chunker.get("chunk_size")
        if not method or size is None:
            return False
        overlap = int(chunker.get("chunk_overlap") or 0)
        try:
            gen_model = cls._find_generator_model(pipeline_config)
        except ValueError:
            return False
        key = (
            "chunk_token_quantiles",
            str(method), int(size), overlap, str(raw_corpus_path), str(gen_model),
        )
        memo = memo if memo is not None else {}
        if key not in memo:
            from rag_stack_evaluator.static_rag_evaluator.chunk_cache import _run_chunker
            raw_df = pd.read_parquet(raw_corpus_path)
            if "contents" not in raw_df.columns and "texts" in raw_df.columns:
                raw_df = raw_df.rename(columns={"texts": "contents"})
            chunked = _run_chunker(raw_df, str(method), int(size), overlap)
            tokenizer = cls._load_tokenizer(pipeline_config)
            memo[key] = cls._compute_chunk_token_quantiles(chunked, tokenizer)
        if not memo[key]:
            return False
        ts_dict["chunk_token_quantiles"] = memo[key]
        return True

    def get_passage_compressor_input_tokens(self, top_k: int) -> int:
        """Passage compressor input = all retrieved passages."""
        return top_k * self.avg_chunk_tokens

    def get_passage_compressor_output_tokens(self) -> int:
        """Passage compressor output ≈ one compressed chunk."""
        return self.avg_chunk_tokens

    def get_passage_filter_input_tokens(self) -> int:
        """Passage filter input = query + passage (re-embed for similarity)."""
        return self.avg_query_tokens + self.avg_chunk_tokens

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return base stats dict for config['_token_stats']."""
        out = {
            "avg_chunk_tokens": self.avg_chunk_tokens,
            "prompt_template_tokens": self.prompt_template_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_query_tokens": self.avg_query_tokens,
            "bytes_per_token": getattr(self, "bytes_per_token", 4.0),
        }
        for key in (
            "avg_llm_input_tokens",
            "avg_query_expansion_input_tokens",
            "avg_query_expansion_output_tokens",
            "avg_passage_reranker_input_tokens",
            "avg_compressor_elems_per_chunk",
            "avg_compressor_elem_tokens",
            "chunk_token_quantiles",
        ):
            value = getattr(self, key, None)
            if value is not None:
                out[key] = value
        return out

    # ------------------------------------------------------------------
    # Alternative constructor from pre-computed dict
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "TokenStats":
        """Reconstruct from a to_dict() output (no tokenizer needed)."""
        instance = object.__new__(cls)
        instance.avg_chunk_tokens = d["avg_chunk_tokens"]
        instance.prompt_template_tokens = d["prompt_template_tokens"]
        instance.avg_output_tokens = d["avg_output_tokens"]
        instance.avg_query_tokens = d["avg_query_tokens"]
        instance.bytes_per_token = float(d.get("bytes_per_token", 4.0))
        instance.avg_llm_input_tokens = d.get("avg_llm_input_tokens")
        instance.avg_query_expansion_input_tokens = d.get(
            "avg_query_expansion_input_tokens"
        )
        instance.avg_query_expansion_output_tokens = d.get(
            "avg_query_expansion_output_tokens"
        )
        instance.avg_passage_reranker_input_tokens = d.get(
            "avg_passage_reranker_input_tokens"
        )
        instance.avg_compressor_elems_per_chunk = d.get(
            "avg_compressor_elems_per_chunk"
        )
        instance.avg_compressor_elem_tokens = d.get("avg_compressor_elem_tokens")
        instance.chunk_token_quantiles = d.get("chunk_token_quantiles")
        return instance

    @staticmethod
    def resolve(pipeline_config: dict) -> "TokenStats":
        """Get TokenStats from pipeline_config['_token_stats'] dict."""
        d = pipeline_config.get("_token_stats")
        if d is None:
            raise ValueError(
                "pipeline_config missing '_token_stats'. "
                "Controller must create TokenStats first."
            )
        return TokenStats.from_dict(d)

    # ------------------------------------------------------------------
    # Config parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_node_lines(config: dict):
        """Yield every (block, node_line) pair from the pipeline list.

        After Phase 2, ``node_lines`` lives inside each ``pipeline[i]``
        method block. Use this helper everywhere instead of reaching for
        top-level ``config["node_lines"]`` (gone).
        """
        from rag_stack.search_space.rag_search_space import pipeline_method_blocks
        for block in pipeline_method_blocks(config):
            if not isinstance(block, dict):
                continue
            for node_line in block.get("node_lines") or []:
                yield block, node_line

    @staticmethod
    def _find_generator_model(config: dict) -> str:
        for _block, node_line in TokenStats._iter_node_lines(config):
            for node in node_line.get("nodes", []):
                if node.get("stage") == "generator":
                    modules = node.get("modules", [])
                    if modules:
                        from rag_stack.search_space.rag_search_space import (
                            sweep_values,
                        )
                        # shape-agnostic: scalar / list / {ordered: [...]}
                        vals = sweep_values(modules[0].get("model", ""))
                        if vals:
                            return vals[0]
        raise ValueError("No generator module with 'model' field found in config.")

    @staticmethod
    def _find_prompt_template(config: dict) -> str:
        for _block, node_line in TokenStats._iter_node_lines(config):
            for node in node_line.get("nodes", []):
                if node.get("stage") == "prompt_maker":
                    modules = node.get("modules", [])
                    if modules:
                        prompt = modules[0].get("prompt")
                        if prompt:
                            return prompt
        # No prompt_maker node found. Sequential static_gt is GUARANTEED one by
        # the config validator, so reaching here means a pipeline that owns its
        # own prompt assembly: the flashrag backend, or an agentic static_gt
        # method (e.g. react). Both are priced by the trace-driven cost model,
        # not these aggregate stats, so a base template keeps the ballpark
        # token estimate sane.
        return (
            "Answer the question based on the given document. "
            "Only give me the answer and do not output any other words. "
            "The following are given documents.\n\n{reference}\n\n"
            "Question: {question}"
        )
