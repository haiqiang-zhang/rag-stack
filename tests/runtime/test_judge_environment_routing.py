"""Explicit judge routing survives dotenv, while dotenv still owns credentials."""

from tests import REPO_ROOT
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('explicit', [True, False])
def test_judge_model_routing_and_credential_precedence(tmp_path, explicit):
    env_file = tmp_path / 'test.env'
    env_file.write_text('JUDGE_MODEL=old-local-model\n'
                        'JUDGE_BASE_URL=http://old-local/v1\n'
                        'DEEPSEEK_API_KEY=synthetic-current-key\n')
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'env_file_path': str(env_file)}))
    environment = dict(os.environ, RAG_STACK_SETTINGS_PATH=str(settings),
                       DEEPSEEK_API_KEY='synthetic-stale-key')
    for key in ['JUDGE_MODEL', 'JUDGE_BASE_URL']:
        environment.pop(key, None)
    if explicit:
        environment.update(JUDGE_MODEL='deepseek-flash',
                           JUDGE_BASE_URL='https://api.deepseek.com')
    code = '''
import json, os, rag_stack
print(json.dumps({'model': os.environ['JUDGE_MODEL'],
                  'endpoint': os.environ['JUDGE_BASE_URL'],
                  'credential_updated': os.environ['DEEPSEEK_API_KEY'] == 'synthetic-current-key'}))
'''
    output = subprocess.check_output([sys.executable, '-c', code], env=environment,
                                      cwd=REPO_ROOT, text=True)
    result = json.loads(output.strip().splitlines()[-1])
    assert result == {'model': 'deepseek-flash' if explicit else 'old-local-model',
                      'endpoint': 'https://api.deepseek.com' if explicit else 'http://old-local/v1',
                      'credential_updated': True}
