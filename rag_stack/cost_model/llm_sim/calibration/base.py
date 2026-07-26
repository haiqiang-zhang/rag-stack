"""Base interfaces for per-module benchmark runners.

Framework convention
--------------------
**Every benchmark runner MUST drive the real implementation that lives
under ``RAG-Stack-Evaluator/rag_stack_evaluator/static_rag_evaluator/nodes/...``** — never re-implement
model loading, tokenization, batching, or inference logic inside the
calibrator. Reasons:

1. **Single source of truth.** Production GT evaluation and calibration
   share one model-loading / tokenizer / forward-pass code path. If a
   future commit changes how passages are tokenized, the calibration
   numbers automatically reflect that change — no risk of the cost model
   silently drifting away from production behaviour.
2. **End-to-end measurement.** What we want to calibrate is the gap
   between GenZ's roofline prediction and what the *production code path*
   actually takes — including Python tokenizer overhead, DataFrame
   marshalling, framework dispatch, etc. Re-implementing a "pure tensor"
   benchmark would underestimate this gap.
3. **No drift between benchmark code and production code.** A module
   author only needs to touch the static_rag_evaluator implementation;
   the calibrator picks the changes up automatically.

Two equally-valid base classes
------------------------------
Pick whichever matches how the production code is *shaped*. Neither is
"the default" — they're peers serving different module families.

**A. ``StaticRAGEvaluatorBenchmarkRunner``** — for AutoRAG-style
nodes whose production entry point is ``BaseModule.pure()``
(rerankers, filters, compressors, query expanders, generators, …).
You declare *which* node-module class to instantiate and *how* to
build a synthetic ``previous_result`` DataFrame for a given
(batch, seq); the shape-grid loop, lazy load, CUDA-event timing, OOM
handling, and median are inherited. ``TartBenchmarkRunner``
([reranker/tart.py](reranker/tart.py)) is the reference example.

**B. ``BaseBenchmarkRunner``** — subclass this directly and implement
``run()`` yourself when the production code path is not a
``BaseModule.pure()`` call. Examples of modules that live here:

  * API-backed modules (Cohere / Jina / OpenAI / …): latency is
    network-bound; time the production SDK client's real method.
  * ``RetrievalEncodeCostModel`` (cost_model/retrieval_encode.py) — the
    embedding model has no ``BaseModule`` wrapper; load
    ``SentenceTransformer`` directly and time ``.encode()``.
  * Any future production path that isn't AutoRAG-shaped.

Minimum skeleton for path B::

      class MyCustomRunner(BaseBenchmarkRunner):
          def run(self, batch_sizes, seq_lens, warmup, measurements):
              results: dict[tuple[int, int], float] = {}
              # instantiate the real production object ONCE here
              for bs in batch_sizes:
                  for seq in seq_lens:
                      # build inputs, warmup, measure median latency
                      results[(bs, seq)] = median_seconds
              return results

The invariant that applies to **both** paths: **call the production
code path** — don't re-implement model loading or inference inside the
runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import median


def _build_torch_dtype_to_genz_bits() -> dict:
    """Lazy-import torch to avoid hard dep at module load time."""
    try:
        import torch
    except ImportError:
        return {}
    return {
        torch.float32: "f32",
        torch.float64: "f32",   # GenZ has no f64; f32 is the closest
        torch.bfloat16: "bf16",
        torch.float16: "bf16",  # GenZ lumps fp16 under bf16 (same bit-width)
        torch.int8: "int8",
        torch.uint8: "int8",
    }


_TORCH_DTYPE_TO_GENZ_BITS: dict = {}


def _ensure_dtype_map() -> dict:
    global _TORCH_DTYPE_TO_GENZ_BITS
    if not _TORCH_DTYPE_TO_GENZ_BITS:
        _TORCH_DTYPE_TO_GENZ_BITS.update(_build_torch_dtype_to_genz_bits())
    return _TORCH_DTYPE_TO_GENZ_BITS


class BaseBenchmarkRunner(ABC):
    """Measure the real module on a shape grid and return latencies.

    ``run(batch_sizes, seq_lens, warmup, measurements)`` returns
    ``{(batch_size, seq_len): median_latency_seconds}``. Missing shapes
    (OOM, ``supports_shape`` returned False) should be omitted, not
    reported as zero.

    See module-level docstring for the framework convention every
    implementation must follow.
    """

    @abstractmethod
    def run(
        self,
        batch_sizes: list[int],
        seq_lens: list[int],
        warmup: int,
        measurements: int,
    ) -> dict[tuple[int, int], float]:
        ...

    def supports_shape(self, batch_size: int, seq_len: int) -> bool:
        """Optional pre-check to skip shapes the module can't handle."""
        return True


# ---------------------------------------------------------------------------
# Standard pattern: instantiate a static_rag_evaluator BaseModule once,
# call .pure() many times, time each call with CUDA events.
# ---------------------------------------------------------------------------

class StaticRAGEvaluatorBenchmarkRunner(BaseBenchmarkRunner):
    """Run the production-path ``BaseModule.pure()`` on a (bs, seq) grid.

    This is the standard recipe for any node-module benchmark runner.
    Subclasses provide three things and inherit the timing/warmup/OOM
    handling:

    1. ``module_class()``  — return the production class to instantiate
       (the SAME class used by ``static_rag_evaluator/nodes/.../run.py``).
       Must subclass ``BaseModule`` so it has a ``pure()`` method.
    2. ``build_input(batch_size, seq_len)`` — return a synthetic
       ``previous_result`` ``pd.DataFrame`` shaped like what the
       production node would receive at this point in the pipeline.
       For rerankers this means: query column + retrieved_contents /
       retrieved_ids / retrieve_scores columns from the upstream
       retrieval node.
    3. ``pure_kwargs(batch_size, seq_len)`` — return the kwargs to pass
       to ``module.pure(previous_result, **kwargs)`` (e.g. ``top_k``,
       ``batch``, model-specific options).

    The runner instantiates the module ONCE (not once per shape — that
    would re-load weights each call), then loops the shape grid issuing
    real ``pure()`` calls timed via CUDA events.
    """

    # ---- subclass contract ----------------------------------------------

    @abstractmethod
    def module_class(self) -> type:
        """Return the BaseModule subclass to benchmark — MUST be the same
        class used in production (under ``static_rag_evaluator/nodes/``)."""
        ...

    def module_init_kwargs(self) -> dict:
        """Optional extra kwargs for the module's ``__init__`` (besides
        ``project_dir``). Override if your module needs custom args."""
        return {}

    @abstractmethod
    def build_input(self, batch_size: int, seq_len: int) -> "pd.DataFrame":
        """Synthesize a ``previous_result`` DataFrame for one benchmark
        point. Should match what the production upstream node would
        produce."""
        ...

    @abstractmethod
    def pure_kwargs(self, batch_size: int, seq_len: int) -> dict:
        """Return the kwargs for ``module.pure(previous_result, **kwargs)``."""
        ...

    # ---- shared infrastructure ------------------------------------------

    def __init__(self, project_dir: str = "."):
        self._project_dir = project_dir
        self._module = None

    def _get_module(self):
        if self._module is None:
            cls = self.module_class()
            self._module = cls(self._project_dir, **self.module_init_kwargs())
        return self._module

    # ---- per-module hooks (override when the production class hides
    # tokenizer / torch model behind a library wrapper like CrossEncoder
    # or FlagReranker; defaults match the project-wide convention used by
    # static_rag_evaluator/nodes/passagereranker/*) ---------------

    def tokenizer(self):
        """Return the HuggingFace tokenizer the production module uses.

        Default contract: production classes under
        ``static_rag_evaluator/nodes/passagereranker/`` expose it
        as ``module.tokenizer``. Override when the production class holds
        a library wrapper (e.g. ``module.model.tokenizer`` for
        sentence_transformers / FlagEmbedding).
        """
        return self._get_module().tokenizer

    def torch_module(self):
        """Return the underlying ``torch.nn.Module`` (for dtype detection).

        Default contract: production classes expose it as ``module.model``.
        Override when wrapped (e.g. ``module.model.model`` for CrossEncoder).
        """
        return self._get_module().model

    # ---- shared helpers -------------------------------------------------

    def _synthesize_passage(self, target_tokens: int) -> str:
        """Produce passage text that tokenizes to ~``target_tokens`` tokens
        via the production module's tokenizer.

        Calibrates for the tokenizer's subword ratio by tokenizing a probe
        string and measuring tokens-per-word, then scales the ``"token "``
        repetition count.
        """
        tok = self.tokenizer()
        probe_words = 100
        probe_text = "token " * probe_words
        n = len(tok(probe_text, add_special_tokens=False)["input_ids"])
        tokens_per_word = max(n / probe_words, 0.5)
        # Leave ~10 tokens of headroom for special tokens / query concat
        target = max(target_tokens - 10, 1)
        words = max(int(target / tokens_per_word), 1)
        return "token " * words

    def detect_bits(self) -> str:
        """Return the GenZ ``bits`` string matching the model's weight dtype."""
        nn_module = self.torch_module()
        dtype = next(nn_module.parameters()).dtype
        return _ensure_dtype_map().get(dtype, "bf16")

    def run(
        self,
        batch_sizes: list[int],
        seq_lens: list[int],
        warmup: int,
        measurements: int,
    ) -> dict[tuple[int, int], float]:
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"{type(self).__name__} requires PyTorch with CUDA for timing."
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available — calibration must be run on the "
                "target GPU."
            )

        module = self._get_module()
        results: dict[tuple[int, int], float] = {}
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        for bs in batch_sizes:
            for seq in seq_lens:
                if not self.supports_shape(bs, seq):
                    continue
                df = self.build_input(bs, seq)
                kwargs = self.pure_kwargs(bs, seq)

                # Warmup
                for _ in range(warmup):
                    module.pure(df, **kwargs)

                # Measure with CUDA events (more accurate than time.perf_counter
                # for short kernels)
                torch.cuda.synchronize()
                latencies_ms: list[float] = []
                for _ in range(measurements):
                    start_evt.record()
                    module.pure(df, **kwargs)
                    end_evt.record()
                    torch.cuda.synchronize()
                    latencies_ms.append(start_evt.elapsed_time(end_evt))

                median_ms = median(latencies_ms)
                results[(bs, seq)] = median_ms / 1000.0
                self._on_point_done(bs, seq, median_ms, len(latencies_ms))

                # Reset CUDA memory between shape points so fragmentation from
                # the previous (bs, seq) doesn't cause a spurious OOM at the
                # next one. If an OOM still fires, it's a genuine capacity
                # limit — let it propagate instead of silently dropping shapes.
                torch.cuda.empty_cache()

        return results

    # ---- hooks for subclass overrides ----------------------------------

    def _on_point_done(
        self, batch_size: int, seq_len: int, median_ms: float, n_runs: int
    ) -> None:
        import logging
        logging.getLogger("RAG-Stack").info(
            f"{type(self).__name__} (bs={batch_size}, seq={seq_len}): "
            f"median={median_ms:.2f} ms over {n_runs} runs"
        )
