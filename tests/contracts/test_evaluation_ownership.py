"""The host extension boundary works without installing a built-in backend."""

from tests import REPO_ROOT

from pathlib import Path
import subprocess
import sys
import textwrap


def test_custom_backend_runs_and_replays_with_evaluator_package_unavailable():
    package_root = str(REPO_ROOT)
    script = textwrap.dedent('''
        import importlib.abc
        import json
        import sys

        blocked = {'rag_stack_evaluator', 'torch', 'transformers', 'vllm', 'flashrag'}

        class BlockBackendImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                    raise AssertionError('Host extension API imported a backend/runtime: ' + fullname)

        sys.meta_path.insert(0, BlockBackendImports())
        from rag_stack.evaluation import (
            BackendSpec, EvaluationResult, Evaluator, EvaluatorContext, TraceSession,
            available_backends, create_quality_evaluator, get_backend_spec,
            normalize_evaluation_result, publish_trace, register_backend,
            supported_metrics, supported_pipeline_modes,
        )
        from rag_stack.evaluator_contract import bind_trace_workflow, normalize_backend_result
        from rag_stack.rag_ir import WorkflowSchema

        contexts = []
        workflow = WorkflowSchema(
            dataflow_type='sequential', stages=['generator_prefill', 'generator_decode'],
        )

        class IndependentEvaluator:
            def __init__(self, context):
                contexts.append(context)
                self.context = context

            def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
                assert metrics_override == ['quality']
                session = TraceSession()
                for question_id in self.context.dataset:
                    with session.query(question_id):
                        with session.stage('generator', model_id='independent-model') as call:
                            call.input(token_count=config['input_tokens'])
                            call.output(token_count=2)
                trace = session.finish(expected_question_ids=self.context.dataset)
                publish_trace(trace, workflow=workflow, on_trace_ready=on_trace_ready)
                return EvaluationResult(
                    metrics={'quality': self.context.options['score']},
                    trace=trace, workflow=workflow,
                )

        spec = BackendSpec(
            name='ownership_fixture', factory=IndependentEvaluator,
            metrics=('quality',), pipeline_modes=('sequential',),
        )
        register_backend(spec)
        assert get_backend_spec('ownership_fixture') is spec
        assert 'ownership_fixture' in available_backends()
        assert supported_metrics('ownership_fixture') == ('quality',)
        assert supported_pipeline_modes('ownership_fixture') == ('sequential',)
        evaluator = create_quality_evaluator(
            dataset=['task-b', 'task-a'], project_dir='fixture-project',
            backend='ownership_fixture', backend_options={'score': 0.75},
        )
        assert isinstance(evaluator, Evaluator)
        assert isinstance(contexts[0], EvaluatorContext)
        assert contexts[0].project_dir == 'fixture-project'

        published = []
        def on_trace_ready(trace):
            # The workflow must be available before the host begins overlapped CM.
            assert trace['provenance']['workflow_schema'] == workflow.to_dict()
            published.append(trace)

        result = evaluator.evaluate(
            {'input_tokens': 3}, metrics_override=['quality'], on_trace_ready=on_trace_ready,
        )
        assert isinstance(result, EvaluationResult)
        assert published[0] is result.trace
        assert [query['question_id'] for query in result.trace['queries']] == ['task-b', 'task-a']
        normalized = normalize_evaluation_result(result, require_trace=True)
        assert normalized['quality'] == 0.75
        assert normalized['__execution_dag__'] is result.trace

        config = {'_runtime': {'rag_ir_mode': 'dynamic'}}
        payload = normalize_backend_result(
            result, backend='ownership_fixture', pipeline_config=config,
            question_ids=['task-b', 'task-a'],
        )
        assert payload['__execution_dag__'] is result.trace
        assert '__workflow_schema__' not in payload
        assert config['_evaluator_workflow_schema'] == workflow.to_dict()
        assert config['_evaluator_trace_stages'] == ['generator']
        # Cache serialization and replay retain workflow/stage metadata without
        # reconstructing or importing the originating backend implementation.
        cached_trace = json.loads(json.dumps(payload['__execution_dag__']))
        replay_config = {}
        bind_trace_workflow(replay_config, cached_trace)
        assert replay_config['_evaluator_workflow_schema'] == config['_evaluator_workflow_schema']
        assert replay_config['_evaluator_trace_stages'] == ['generator']
        assert not any(
            name == backend or name.startswith(backend + '.')
            for name in sys.modules for backend in blocked
        )
    ''')
    script = f"import sys\nsys.path.insert(0, {package_root!r})\n" + script
    subprocess.run(
        [sys.executable, '-I', '-c', script], check=True, capture_output=True, text=True,
    )
