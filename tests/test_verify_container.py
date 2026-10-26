"""Safety contracts for the disposable Docker validator; no Docker/provider I/O."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_container as verifier


PROJECT = 'tracemem-check-synthetic-owner'
TOKEN = 'synthetic-validator-token-not-a-provider-credential'


def healthy_inspection(project=PROJECT):
    return {
        'Config': {
            'Labels': {'com.docker.compose.project': project},
            'User': '10001:10001',
        },
        'HostConfig': {
            'ReadonlyRootfs': True,
            'CapDrop': ['ALL'],
            'SecurityOpt': ['no-new-privileges:true'],
            'Memory': 1024 ** 3,
            'NanoCpus': 2_000_000_000,
            'PidsLimit': 128,
        },
        'NetworkSettings': {
            'Ports': {'8000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '49152'}]},
        },
        'Mounts': [{
            'Destination': '/data', 'Type': 'volume', 'RW': True,
            'Name': project + '_memory-data',
        }],
        'State': {'Health': {'Status': 'healthy'}},
    }


class ContainerValidatorTests(unittest.TestCase):
    def test_environment_cannot_select_callers_project_token_or_port(self):
        inherited = {
            'PATH': 'synthetic-path',
            'COMPOSE_FILE': 'production.yaml',
            'COMPOSE_PROJECT_NAME': 'production',
            'COMPOSE_ENV_FILES': 'production.env',
            'COMPOSE_PROFILES': 'production',
            'TRACEMEM_SECRET_FILE': 'production-token',
            'TRACEMEM_IMAGE': 'production-image',
            'TRACEMEM_PORT': '8765',
            'TRACEMEM_DB': 'production.sqlite',
        }
        with patch.dict(os.environ, inherited, clear=True):
            isolated = verifier.isolated_environment(Path('synthetic-test-token'), 'test-image')
        self.assertEqual(isolated['PATH'], 'synthetic-path')
        self.assertFalse(any(name.startswith('COMPOSE_') for name in isolated))
        self.assertNotIn('TRACEMEM_DB', isolated)
        self.assertEqual(isolated['TRACEMEM_SECRET_FILE'], 'synthetic-test-token')
        self.assertEqual(isolated['TRACEMEM_IMAGE'], 'test-image')
        self.assertEqual(isolated['TRACEMEM_PORT'], '0')
        self.assertGreater(float(isolated['TRACEMEM_CPUS']), 0)
        self.assertNotIn('production', json.dumps(isolated))

    def test_inspection_accepts_owned_bounded_loopback_container(self):
        for option in ('no-new-privileges', 'no-new-privileges:true'):
            info = healthy_inspection()
            info['HostConfig']['SecurityOpt'] = [option]
            with self.subTest(option=option):
                checks = verifier.validate_inspection(info, PROJECT)
                self.assertTrue(checks['host_loopback_only'])
                self.assertTrue(checks['bounded_resources'])

    def test_inspection_rejects_unsafe_or_unowned_containers(self):
        cases = [
            ('wrong project', ('Config', 'Labels', 'com.docker.compose.project'), 'production'),
            ('root user', ('Config', 'User'), '0:0'),
            ('writable root', ('HostConfig', 'ReadonlyRootfs'), False),
            ('retained capabilities', ('HostConfig', 'CapDrop'), []),
            ('missing privilege restriction', ('HostConfig', 'SecurityOpt'), []),
            ('false privilege restriction', ('HostConfig', 'SecurityOpt'), ['no-new-privileges:false']),
            ('unbounded memory', ('HostConfig', 'Memory'), 0),
            ('unbounded processes', ('HostConfig', 'PidsLimit'), 0),
            ('unbounded CPU', ('HostConfig', 'NanoCpus'), 0),
            ('public bind', ('NetworkSettings', 'Ports', '8000/tcp'),
             [{'HostIp': '0.0.0.0', 'HostPort': '49152'}]),
            ('second public bind', ('NetworkSettings', 'Ports', '8000/tcp'),
             [{'HostIp': '127.0.0.1', 'HostPort': '49152'}, {'HostIp': '::', 'HostPort': '49152'}]),
            ('unpublished port', ('NetworkSettings', 'Ports', '8000/tcp'), None),
            ('missing data volume', ('Mounts',), []),
            ('unhealthy container', ('State', 'Health', 'Status'), 'unhealthy'),
        ]
        for label, path, value in cases:
            info = healthy_inspection()
            target = info
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(boundary=label), self.assertRaises(verifier.ValidationError):
                verifier.validate_inspection(info, PROJECT)
        for changed in ({'Type': 'bind'}, {'RW': False}, {'Name': 'production_memory-data'}):
            info = healthy_inspection()
            info['Mounts'][0].update(changed)
            with self.subTest(volume=changed), self.assertRaises(verifier.ValidationError):
                verifier.validate_inspection(info, PROJECT)

    def test_no_docker_records_failure_without_creating_runtime_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evidence' / 'validation.json'
            stdout = io.StringIO()
            with patch.object(verifier.shutil, 'which', return_value=None), \
                    patch.object(verifier.subprocess, 'run') as launch, \
                    patch.object(verifier.tempfile, 'TemporaryDirectory') as resources, \
                    patch.object(verifier.secrets, 'token_urlsafe') as token, \
                    redirect_stdout(stdout):
                result = verifier.main(['--output', str(output)])
            launch.assert_not_called()
            resources.assert_not_called()
            token.assert_not_called()
            evidence = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result, 1)
            self.assertEqual(evidence['state'], 'failed')
            self.assertEqual(evidence['failed_phase'], 'docker_preflight')
            self.assertTrue(evidence['owned_resources_cleaned'])
            self.assertEqual(evidence['checks'], {})
            self.assertEqual(evidence['model_calls'], 0)
            self.assertEqual(json.loads(stdout.getvalue()), evidence)

    def test_failed_cleanup_keeps_original_error_and_only_targets_owned_resources(self):
        commands = []
        owned_project = None

        def fake_run(command, **kwargs):
            nonlocal owned_project
            commands.append(copy.deepcopy(command))
            out = ''
            error = ''
            code = 0
            if command[:2] == ['docker', 'version']:
                out = json.dumps({'Os': 'linux', 'Version': 'synthetic', 'Arch': 'amd64'})
            elif command[:3] == ['docker', 'compose', 'version']:
                out = 'synthetic-compose'
            elif command[:2] == ['docker', 'compose']:
                project = command[command.index('-p') + 1]
                self.assertTrue(project.startswith('tracemem-check-'))
                owned_project = owned_project or project
                self.assertEqual(project, owned_project)
                self.assertNotIn('COMPOSE_PROJECT_NAME', kwargs['env'])
                self.assertNotIn('COMPOSE_FILE', kwargs['env'])
                args = command[command.index('-p') + 2:]
                if args[:1] == ['ps']:
                    out = 'synthetic-container-id'
                elif 'deploy/smoke_test.py' in args:
                    code, error = 17, 'seed rejected ' + TOKEN
                elif args[:1] == ['down']:
                    code, error = 18, 'synthetic cleanup failed'
            elif command[:3] == ['docker', 'image', 'inspect']:
                out = 'sha256:synthetic-image'
            elif command[:2] == ['docker', 'inspect']:
                out = json.dumps([healthy_inspection(owned_project)])
            elif command[:2] == ['git', 'rev-parse']:
                out = 'synthetic-commit'
            elif command[:3] == ['docker', 'image', 'rm']:
                self.assertTrue(command[3].startswith('trace-mem-mm:check-'))
            else:
                self.fail('unexpected external command: ' + repr(command))
            return subprocess.CompletedProcess(command, code, out, error)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'validation.json'
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(verifier.shutil, 'which', return_value='synthetic-docker'), \
                    patch.object(verifier.subprocess, 'run', side_effect=fake_run), \
                    patch.object(verifier.secrets, 'token_urlsafe', return_value=TOKEN), \
                    patch.dict(os.environ, {'COMPOSE_PROJECT_NAME': 'production', 'COMPOSE_FILE': 'production.yaml'}), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                result = verifier.main(['--output', str(output)])
            evidence = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result, 1)
            self.assertEqual(evidence['failed_phase'], 'synthetic_seed')
            self.assertIn('exit code 17', evidence['error'])
            self.assertFalse(evidence['owned_resources_cleaned'])
            self.assertEqual(evidence['cleanup_failed_resources'], [owned_project])
            cleanup = [cmd for cmd in commands if 'down' in cmd]
            self.assertEqual(len(cleanup), 1)
            self.assertIn('--volumes', cleanup[0])
            self.assertEqual(cleanup[0][cleanup[0].index('-p') + 1], owned_project)
            self.assertEqual(len([cmd for cmd in commands if cmd[:3] == ['docker', 'image', 'rm']]), 1)
            self.assertNotIn(TOKEN, stdout.getvalue() + stderr.getvalue() + json.dumps(evidence))
            self.assertIn('[redacted]', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
