"""Isolated preview lifecycle and public evaluator boundaries, without models."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag_stack import cm_preview as preview
from rag_stack.evaluation import EvaluationResult
from rag_stack.rag_ir import make_quality_trace_envelope


def manager_stub():
    return SimpleNamespace(resolve_corpus=lambda params: object(),
                           apply_corpus_view=lambda config, view: None)


def observed(ids=("q0", "q1", "q2")):
    trace = make_quality_trace_envelope(
        [[{"stage": "generator", "model_id": "fixture/model", "input_tokens": 3,
           "output_tokens": 2, "step_idx": 0}] for _ in ids], question_ids=list(ids))
    trace["provenance"] = {"trace_origin": "observed", "producer": "fixture"}
    return trace


@pytest.fixture
def project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "rag_stack_config.yaml").write_text(yaml.safe_dump({
        "global": {"eval_backend": "static_gt"}, "dataset": {"dataset_name": "fixture"}}))
    (project / "summary.json").write_text('{"status":"completed","gt_count":2}')
    (project / "all_evaluations.csv").write_text('eval_id,performance\n0,10\n')
    (project / "pareto_frontier.csv").write_text('eval_id,performance\n0,10\n')
    evaluation = project / "evaluations" / "eval_0000"
    evaluation.mkdir(parents=True)
    algo = {"node_lines": [{"nodes": [{"stage": "generator", "modules": [
        {"component": "vllm_api", "model": "fixture/model"}]}]}],
        "_token_stats": {"avg_chunk_tokens": 7}, "dataset": {"qa": "obsolete"},
        "system": {"gpu_model": "test", "cm_search_space": {"max_batch_size_request": 16}},
        "eval_backend_setting": {"performance_only": True, "metrics": ["correctness"]}}
    (evaluation / "pipeline_config.json").write_text(json.dumps(algo))
    (evaluation / "system_config.json").write_text(json.dumps({"layout": {"available_devices": ["cuda:0"]}}))
    (evaluation / "execution_dag.json").write_text(json.dumps(observed()))
    return project


def create(project, **request):
    return preview.create_preview(project, {"eval_id": 0, "reuse_trace": True,
                                           "sample_count": 2, "skip_judge": True, **request}, launch=False)


def test_reused_trace_preserves_selected_invocations_and_never_mutates_optimizer(project, monkeypatch):
    before = {str(p.relative_to(project)): p.read_bytes() for p in project.rglob('*') if p.is_file()}
    state = create(project, question_ids=["q2", "q0"])
    directory = preview.preview_directory(project, state["preview_id"])
    seen = {}
    def compile_plan(directory, request, algo, system, trace):
        seen.update(trace=deepcopy(trace), algo=deepcopy(algo), system=deepcopy(system))
        return {"status": "compiled", "prediction": None}
    monkeypatch.setattr(preview, "_compile", compile_plan)
    assert preview.run_worker(directory, state["run_id"]) == 0
    result = preview.detail(directory)
    assert result["status"] == "succeeded"
    assert result["compiled_path_status"] == "available"
    assert result["compiled_plan"] == {"status": "compiled", "prediction": None}
    assert not (directory / "compiled_plan.json").exists()
    assert preview.read_json(directory / "compiled_description.json") == result["compiled_plan"]
    assert result["cohort"]["question_ids"] == ["q2", "q0"]
    assert result["cohort"]["requested_count"] == 2
    assert [q["question_id"] for q in seen["trace"]["queries"]] == ["q2", "q0"]
    original = observed()
    assert seen["trace"]["queries"] == [original["queries"][2], original["queries"][0]]
    assert seen["algo"]["_token_stats"] == {"avg_chunk_tokens": 7}
    algo = preview.read_json(directory / "algo_config.json")
    assert "dataset" not in algo and "system" not in algo and "_token_stats" not in algo
    assert "cm_search_space" not in preview.read_json(directory / "system_config.json")
    assert all((project / path).read_bytes() == content for path, content in before.items())
    assert "sha256" not in (directory / "state.json").read_text()


def test_legacy_executable_plan_is_never_read_downloaded_or_recompiled(project, monkeypatch):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    preview.update_state(directory, status="succeeded", phase="complete")
    preview.write_json(directory / "compiled_plan.json", {"status": "compiled", "old": True})
    with pytest.raises(preview.PreviewError, match="Unknown preview artifact"):
        preview.artifact_path(directory, "compiled_plan.json")
    monkeypatch.setattr(preview, "_compile", lambda *args: pytest.fail("Viewing a result must not compile"))
    result = preview.detail(directory)
    assert result["status"] == "succeeded"
    assert result["compiled_plan"] is None
    assert result["compiled_path_status"] == "unavailable"
    assert not any(item["name"].startswith("compiled_") for item in result["artifacts"])


def test_worker_saves_only_display_projection_and_another_process_can_reopen_it(project, monkeypatch):
    import subprocess
    import sys
    import textwrap

    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    description = {"status": "compiled", "prediction": None, "sentinel": "display-only-cm-result"}
    script = (
        "from rag_stack import cm_preview as p; "
        f"p._compile = lambda *args: {description!r}; p.main()"
    )
    real_popen = subprocess.Popen
    processes = []
    def fixture_worker(cmd, **kwargs):
        # Run the real isolated worker lifecycle, with a CPU compiler fixture.
        proc = real_popen([sys.executable, "-u", "-c", script, *cmd[4:]], **kwargs)
        processes.append(proc)
        return proc
    monkeypatch.setattr(preview.subprocess, "Popen", fixture_worker)
    preview.launch_worker(directory)
    try:
        assert processes[0].wait(timeout=15) == 0
        result = preview.detail(directory)
        assert result["status"] == "succeeded", result
        assert result["compiled_path_status"] == "available"
        assert result["compiled_plan"] == description
        projection_path = directory / "compiled_description.json"
        assert preview.read_json(projection_path) == description
        assert preview.artifact_path(directory, projection_path.name) == projection_path
        for path in directory.rglob("*"):
            if path.is_file() and path != projection_path:
                assert b"display-only-cm-result" not in path.read_bytes(), path
        assert not (directory / "compiled_plan.json").exists()

        # A fresh process has no worker objects or result cache; viewing is file I/O only.
        monkeypatch.setattr(preview.subprocess, "Popen", real_popen)
        reopen_script = textwrap.dedent("""
            import json, sys
            from pathlib import Path
            from rag_stack import cm_preview as p
            def forbidden(*args, **kwargs):
                raise AssertionError('Viewing must not evaluate or compile')
            p.run_worker = p._fresh_trace = p._compile = forbidden
            directory = Path(sys.argv[1])
            detail = p.detail(directory)
            print(json.dumps({
                'compiled_path_status': detail['compiled_path_status'],
                'compiled_plan': detail['compiled_plan'],
                'download': p.read_json(p.artifact_path(directory, 'compiled_description.json')),
                'listed': p.list_previews(directory.parent.parent)[0]['compiled_path_status'],
            }))
        """)
        reopened = subprocess.run([sys.executable, "-c", reopen_script, str(directory)],
                                  capture_output=True, text=True, timeout=15, check=True)
        assert json.loads(reopened.stdout) == {
            "compiled_path_status": "available", "compiled_plan": description,
            "download": description, "listed": "available",
        }
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_cancellation_during_compilation_cannot_publish_a_success_or_description(project, monkeypatch):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    def compile_then_cancel(*args):
        preview.update_state(directory, status="cancelled", phase="cancelled")
        return {"status": "compiled"}
    monkeypatch.setattr(preview, "_compile", compile_then_cancel)
    assert preview.run_worker(directory, state["run_id"]) == 1
    result = preview.detail(directory)
    assert result["status"] == "cancelled"
    assert result["compiled_plan"] is None
    assert result["compiled_path_status"] is None
    assert not (directory / "compiled_description.json").exists()
    with pytest.raises(FileNotFoundError):
        preview.artifact_path(directory, "compiled_description.json")


@pytest.mark.parametrize("failed_file", ["compiled_description.json", "state.json"])
def test_failed_publication_never_exposes_an_incomplete_display_result(project, monkeypatch, failed_file):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    write_json = preview.write_json
    def failing_write(path, value):
        if path.name == failed_file and (failed_file != "state.json" or value["status"] == "succeeded"):
            raise OSError("fixture output failure")
        write_json(path, value)
    monkeypatch.setattr(preview, "write_json", failing_write)
    monkeypatch.setattr(preview, "_compile", lambda *args: {"status": "compiled"})
    assert preview.run_worker(directory, state["run_id"]) == 1
    result = preview.detail(directory)
    assert result["status"] == "failed"
    assert "fixture output failure" in result["error"]["message"]
    assert result["compiled_plan"] is None
    assert result["compiled_path_status"] is None
    assert not any(item["name"] == "compiled_description.json" for item in result["artifacts"])
    with pytest.raises(FileNotFoundError):
        preview.artifact_path(directory, "compiled_description.json")


def test_reaper_launch_failure_reaps_the_just_started_worker(project, monkeypatch):
    import subprocess
    import sys
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    processes = []
    real_popen = subprocess.Popen
    def spawn(cmd, **kwargs):
        process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process
    def cannot_start(*args):
        raise RuntimeError("reaper thread unavailable")
    monkeypatch.setattr(preview.subprocess, "Popen", spawn)
    monkeypatch.setattr(preview.threading.Thread, "start", cannot_start)
    try:
        preview.launch_worker(directory)
        assert processes[0].poll() is not None
        result = preview.detail(directory)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "worker_launch_failed"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_cm_block_retains_real_trace_and_actionable_reason(project, monkeypatch):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    def blocked(*args):
        raise preview.PreviewError("unsupported_bm25", "BM25 has no CM calibration", phase="compiling")
    monkeypatch.setattr(preview, "_compile", blocked)
    assert preview.run_worker(directory, state["run_id"]) == 1
    result = preview.detail(directory)
    assert result["status"] == "blocked"
    assert result["error"]["code"] == "unsupported_bm25"
    assert len(result["rag_ir"]["queries"]) == 2
    assert result["compiled_plan"] is None
    assert preview.list_previews(project)[0]["preview_id"] == state["preview_id"]


def test_trace_sampling_rejects_ambiguous_or_unknown_ids():
    for ids, request in [(["a", "a"], {}), (["a"], {"question_ids": ["b"]}),
                         (["a"], {"question_ids": ["a", "a"]}), ([], {})]:
        with pytest.raises(preview.PreviewError):
            preview.select_question_ids(ids, request)
    assert preview.select_question_ids(list(range(10)), {}) == list(range(8))
    assert preview.select_question_ids([10, 11], {"question_ids": ["11"]}) == [11]


def test_reused_pid_is_never_signalled_and_orphan_can_reopen(project, monkeypatch):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    preview.update_state(directory, status="running", worker={"pid": 123, "pgid": 123, "start_ticks": "old"})
    monkeypatch.setattr(preview, "_process_identity", lambda pid: {"pid": pid, "pgid": pid, "start_ticks": "new"})
    monkeypatch.setattr(preview.os, "killpg", lambda *args: pytest.fail("Never signal a reused PID"))
    assert preview.detail(directory)["status"] == "interrupted"
    assert preview.cancel_preview(directory)["status"] == "interrupted"


def test_cancel_only_owned_process_group_and_never_restarts_terminal_job(project, monkeypatch):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    preview.update_state(directory, worker={"pid": 123, "pgid": 123, "start_ticks": "start"})
    monkeypatch.setattr(preview, "_owns_worker", lambda state: True)
    killed = []
    monkeypatch.setattr(preview.os, "killpg", lambda *args: killed.append(args))
    assert preview.cancel_preview(directory)["status"] == "cancelled"
    assert len(killed) == 1 and killed[0][0] == 123
    assert preview.run_worker(directory, state["run_id"]) == 1
    assert preview.update_state(directory, status="succeeded")["status"] == "cancelled"
    assert preview.cancel_preview(directory)["status"] == "cancelled"
    assert len(killed) == 1


def test_artifact_allowlist_and_symlinks_cannot_escape_preview(project, tmp_path):
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    with pytest.raises(preview.PreviewError):
        preview.artifact_path(directory, "../summary.json")
    secret = tmp_path / "outside.json"
    secret.write_text('{}')
    (directory / "rag_ir.json").symlink_to(secret)
    with pytest.raises(preview.PreviewError):
        preview.artifact_path(directory, "rag_ir.json")
    assert preview.detail(directory)["rag_ir"] is None
    with pytest.raises(preview.PreviewError):
        preview.preview_directory(project, "../outside")


@pytest.mark.parametrize("skip_judge", [True, False])
def test_fresh_trace_uses_public_evaluator_and_explicit_judge_choice(project, monkeypatch, skip_judge):
    from rag_stack import replay_cli
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    algo = preview.read_json(directory / "algo_config.json")
    system = preview.read_json(directory / "system_config.json")
    trace = observed(["q1"])
    calls = {}
    class Evaluator:
        def evaluate(self, config, **kwargs):
            calls.update(config=deepcopy(config), **kwargs)
            kwargs["on_trace_ready"](trace)
            return EvaluationResult(metrics={}, trace=trace)
    def runtime(config, project, **kwargs):
        calls["manager_args"] = kwargs
        return manager_stub(), Evaluator(), "static_gt"
    monkeypatch.setattr(replay_cli, "_runtime", runtime)
    output = preview._fresh_trace(directory, {"skip_judge": skip_judge, "corpus_path": "/explicit/corpus.parquet"},
                                  algo, system, Path("/sample/qa.parquet"), ["q1"])
    assert output is trace
    assert calls["metrics_override"] == ([] if skip_judge else None)
    assert calls["config"]["eval_backend_setting"].get("performance_only", False) == skip_judge
    assert calls["config"]["system"] == system
    assert calls["manager_args"]["qa_path"] == Path("/sample/qa.parquet")
    assert "dataset" not in calls["config"]
    assert algo["eval_backend_setting"]["performance_only"] is True


def test_static_fallback_trace_is_not_presented_as_observed(project, monkeypatch):
    from rag_stack import replay_cli
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    evaluator = SimpleNamespace(evaluate=lambda *args, **kwargs: EvaluationResult(metrics={}, trace=observed(["q1"])))
    monkeypatch.setattr(replay_cli, "_runtime", lambda *args, **kwargs: (manager_stub(), evaluator, "static_gt"))
    with pytest.raises(preview.PreviewError, match="observed serving trace"):
        preview._fresh_trace(directory, {"corpus_path": "/corpus.parquet"},
                             preview.read_json(directory / "algo_config.json"), {}, Path("/qa.parquet"), ["q1"])
    assert not (directory / "rag_ir.json").exists()


def test_early_trace_survives_evaluator_failure(project, monkeypatch):
    from rag_stack import replay_cli
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    class Evaluator:
        def evaluate(self, config, **kwargs):
            kwargs["on_trace_ready"](observed(["q1"]))
            raise RuntimeError("judge failed after serving")
    monkeypatch.setattr(replay_cli, "_runtime", lambda *args, **kwargs: (manager_stub(), Evaluator(), "static_gt"))
    with pytest.raises(RuntimeError, match="judge failed"):
        preview._fresh_trace(directory, {"skip_judge": False, "corpus_path": "/corpus.parquet"},
                             preview.read_json(directory / "algo_config.json"), {}, Path("/qa.parquet"), ["q1"])
    assert preview.read_json(directory / "rag_ir.json")["queries"][0]["question_id"] == "q1"


def test_sampled_parquet_contains_only_requested_ids_in_requested_order(project):
    import pandas as pd
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    source = project / "qa.parquet"
    frame = pd.DataFrame({"qid": [10, 11, 12], "query": ["a", "b", "c"]})
    frame.to_parquet(source, index=False)
    original = source.read_bytes()
    sampled, ids = preview._sample_qa(directory, {"qa_path": str(source), "question_ids": ["12", 10]})
    assert ids == [12, 10]
    assert pd.read_parquet(sampled).to_dict("list") == {"qid": [12, 10], "query": ["c", "a"]}
    assert source.read_bytes() == original


def test_formal_parameter_resolver_requires_every_active_choice(project, monkeypatch):
    from rag_stack import performance_context
    from rag_stack.search_space import RagSearchSpace
    config = {"global": {"eval_backend": "static_gt"}, "system": {"performance_source": "cost_model"},
              "algo_search_space": {"pipeline": {"node_lines": [{"nodes": [{"stage": "generator", "modules": [
                  {"component": "vllm_api", "model": ["m1", "m2"], "temperature": [0.1, 0.7]},
                  {"component": "vllm", "model": ["m3", "m4"], "max_tokens": [3, 9]},
              ]}]}]}}}
    adapter = RagSearchSpace.from_yaml(config, check_performance_support=False)
    seen = {}
    class Context:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            self._ssops, self.search_space, self.system_space_owner = adapter, adapter.ss, "cost_model"
        def resolve_config(self, decoded):
            runtime, lines = adapter.resolve_pipeline_block(config, decoded)
            return {"pipeline_runtime": runtime, "node_lines": lines}
    monkeypatch.setattr(performance_context, "PerformanceContext", Context)
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    params = {"pipeline__sequential__generator__component": "vllm_api",
              "pipeline__sequential__generator__vllm_api__model": "m2",
              "pipeline__sequential__generator__vllm_api__temperature": 0.7}
    request = {"parameters": params, "corpus_path": "/explicit/corpus.parquet"}
    algo, system = preview._resolve_parameters(directory, request, Path("/sampled/qa.parquet"))
    assert system is None  # CM deployment sweep was not silently sampled.
    assert algo["node_lines"][0]["nodes"][0]["modules"] == [{"component": "vllm_api", "model": "m2", "temperature": 0.7}]
    assert seen["check_performance_support"] is False
    assert seen["qa_data_path"] == "/sampled/qa.parquet"
    del params["pipeline__sequential__generator__vllm_api__model"]
    with pytest.raises(preview.PreviewError) as error:
        preview._resolve_parameters(directory, request, Path("/sampled/qa.parquet"))
    assert error.value.code == "invalid_parameters"


def test_callback_trace_is_kept_when_returned_trace_identity_is_wrong(project, monkeypatch):
    from rag_stack import replay_cli
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    real = observed(["q1"])
    class Evaluator:
        def evaluate(self, config, **kwargs):
            kwargs["on_trace_ready"](real)
            return EvaluationResult(metrics={}, trace=deepcopy(real))
    monkeypatch.setattr(replay_cli, "_runtime", lambda *args, **kwargs: (manager_stub(), Evaluator(), "static_gt"))
    with pytest.raises(preview.PreviewError) as error:
        preview._fresh_trace(directory, {"corpus_path": "/corpus.parquet"},
                             preview.read_json(directory / "algo_config.json"), {}, Path("/qa.parquet"), ["q1"])
    assert error.value.code == "trace_identity_mismatch"
    assert preview.read_json(directory / "rag_ir.json") == real


@pytest.mark.parametrize("runtime", [{"mode": "parallel"}, {"mode": "sequential", "rag_dataflow": "parallel"}])
def test_explicit_algorithm_rejects_unsupported_runtime_modes(project, runtime):
    with pytest.raises(preview.PreviewError, match="sequential or react"):
        preview.create_preview(project, {"algo_config": {"node_lines": [], "pipeline_runtime": runtime},
                                         "system_config": {"layout": {"available_devices": ["cuda:0"]}}}, launch=False)
    assert not (project / "cm_previews").exists()


def test_explicit_algorithm_accepts_public_default_sequential_runtime(project):
    original = preview.read_json(project / "evaluations/eval_0000/pipeline_config.json")
    algo, system = preview.split_config_pair(original, {"layout": {"available_devices": ["cuda:0"]}})
    assert "pipeline_runtime" not in algo
    result = preview.create_preview(project, {"algo_config": algo, "system_config": system}, launch=False)
    assert result["status"] == "queued"


@pytest.mark.parametrize("field, message", [
    ("dataset", "must not bind a dataset"),
    ("system", "belong in system_config"),
    ("algo_search_space", "not optimizer settings"),
    ("cm_search_space", "not optimizer settings"),
    ("multiple_modules", "exactly one module"),
])
def test_explicit_algorithm_errors_are_rejected_before_snapshot(project, field, message):
    original = preview.read_json(project / "evaluations/eval_0000/pipeline_config.json")
    algo, system = preview.split_config_pair(original, {"layout": {"available_devices": ["cuda:0"]}})
    algo["pipeline_runtime"] = {"mode": "sequential"}
    if field == "multiple_modules":
        algo["node_lines"][0]["nodes"][0]["modules"].append({"component": "vllm"})
    else:
        algo[field] = {}
    with pytest.raises(ValueError, match=message):
        preview.create_preview(project, {"algo_config": algo, "system_config": system}, launch=False)
    assert not (project / "cm_previews").exists()


@pytest.mark.parametrize("field", ["dataset", "cm_search_space"])
def test_explicit_system_errors_are_not_silently_discarded(project, field):
    with pytest.raises(ValueError):
        create(project, system_config={"layout": {"available_devices": ["cuda:0"]}, field: {}})
    assert not (project / "cm_previews").exists()


def test_orphan_recovery_cannot_overwrite_concurrent_terminal_result(project, monkeypatch):
    from contextlib import contextmanager
    import threading
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    preview.update_state(directory, status="running", worker={"pid": 123, "pgid": 123, "start_ticks": "old"})
    monkeypatch.setattr(preview, "_owns_worker", lambda state: False)
    original_lock = preview._state_lock
    waiting = threading.Event()
    @contextmanager
    def announced_lock(path):
        waiting.set()
        with original_lock(path):
            yield
    monkeypatch.setattr(preview, "_state_lock", announced_lock)
    results = []
    with original_lock(directory):
        thread = threading.Thread(target=lambda: results.append(preview.recover_state(directory)))
        thread.start()
        assert waiting.wait(2)
        final = preview.read_json(directory / "state.json")
        final.update(status="succeeded", phase="complete")
        preview.write_json(directory / "state.json", final)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert results[0]["status"] == "succeeded"
    assert preview.read_json(directory / "state.json")["status"] == "succeeded"


@pytest.mark.parametrize("origin", [None, "reconstructed"])
def test_trace_reuse_requires_observed_origin_even_when_envelope_is_valid(project, monkeypatch, origin):
    path = project / "evaluations/eval_0000/execution_dag.json"
    trace = json.loads(path.read_text())
    if origin is None:
        trace.pop("provenance", None)
    else:
        trace["provenance"]["trace_origin"] = origin
    path.write_text(json.dumps(trace))
    state = create(project)
    directory = preview.preview_directory(project, state["preview_id"])
    monkeypatch.setattr(preview, "_compile", lambda *args: pytest.fail("Unobserved trace must not compile"))
    assert preview.run_worker(directory, state["run_id"]) == 1
    result = preview.detail(directory)
    assert result["status"] == "blocked"
    assert result["error"]["code"] == "trace_origin_unverified"
    assert result["error"]["details"]["fresh_run_required"] is True
    assert result["rag_ir"] is None
    assert preview.read_json(directory / "source_trace.json") == trace
