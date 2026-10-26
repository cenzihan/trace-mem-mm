"""Build and validate the real deterministic container with disposable synthetic data.

Never points at an existing Compose project, token, or database. Only resources
created under fresh tracemem-check-* project names are removed on completion.
The output is aggregate deployment evidence, not a benchmark or provider receipt.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


class ValidationError(RuntimeError):
    pass


def isolated_environment(secret_file, image):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('COMPOSE_', 'TRACEMEM_'))}
    env.update(TRACEMEM_SECRET_FILE=str(secret_file), TRACEMEM_IMAGE=image,
               TRACEMEM_PORT='0', TRACEMEM_MEMORY_LIMIT='1g', TRACEMEM_CPUS='2.0')
    return env


def validate_inspection(info, project):
    config, host = info['Config'], info['HostConfig']
    if config['Labels'].get('com.docker.compose.project') != project:
        raise ValidationError('container project ownership mismatch')
    if config['User'] != '10001:10001' or not host['ReadonlyRootfs']:
        raise ValidationError('non-root or read-only container boundary missing')
    if 'ALL' not in host.get('CapDrop', []) or not any(
            value in ('no-new-privileges', 'no-new-privileges:true')
            for value in host.get('SecurityOpt', [])):
        raise ValidationError('container privilege restriction missing')
    if host.get('Memory', 0) <= 0 or host.get('PidsLimit', 0) <= 0 or host.get('NanoCpus', 0) <= 0:
        raise ValidationError('container resource bound missing')
    bindings = info['NetworkSettings']['Ports'].get('8000/tcp') or []
    if len(bindings) != 1 or bindings[0]['HostIp'] != '127.0.0.1' or int(bindings[0]['HostPort']) <= 0:
        raise ValidationError('expected exactly one host-loopback port binding')
    volumes = [item for item in info['Mounts'] if item['Destination'] == '/data']
    if len(volumes) != 1 or volumes[0]['Type'] != 'volume' or not volumes[0]['RW']:
        raise ValidationError('persistent writable data volume missing')
    if not volumes[0]['Name'].startswith(project + '_'):
        raise ValidationError('data volume is outside the disposable project')
    if info['State'].get('Health', {}).get('Status') != 'healthy':
        raise ValidationError('container is not healthy')
    return {'non_root': True, 'read_only_root': True, 'host_loopback_only': True,
            'persistent_named_volume': True, 'bounded_resources': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/container-validation.json')
    args = parser.parse_args(argv)
    report = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), state='running',
                  profile='deterministic_container', dense_online=False, rerank_online=False,
                  synthetic_data_only=True, model_calls=0, official_evaluation=False, checks={})
    phase = 'docker_preflight'
    token = ''
    projects = []
    env = None
    temp = None
    built = False
    image = 'trace-mem-mm:check-' + uuid.uuid4().hex

    def run(command, timeout=120):
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValidationError(type(exc).__name__ + ' during ' + phase) from None
        if result.returncode:
            detail = (result.stderr or result.stdout)[-4000:]
            if token:
                detail = detail.replace(token, '[redacted]')
            print(detail, file=sys.stderr)
            raise ValidationError(phase + ' failed with exit code ' + str(result.returncode))
        return result.stdout.strip()

    def compose(project, *arguments, timeout=120):
        if not project.startswith('tracemem-check-') or project not in projects:
            raise ValidationError('refusing to operate on an unowned Compose project')
        return run(['docker', 'compose', '--env-file', str(Path(temp.name) / 'empty.env'),
                    '-f', str(ROOT / 'compose.yaml'), '-p', project, *arguments], timeout)

    try:
        if not shutil.which('docker'):
            raise ValidationError('Docker CLI is unavailable; install Docker Engine with Compose v2')
        version = json.loads(run(['docker', 'version', '--format', '{{json .Server}}']))
        if version.get('Os') != 'linux':
            raise ValidationError('a Linux Docker Engine is required')
        report['docker_engine_version'] = version['Version']
        report['docker_engine_arch'] = version['Arch']
        run(['docker', 'compose', 'version', '--short'])
        temp = tempfile.TemporaryDirectory(prefix='tracemem-container-check-')
        root = Path(temp.name)
        (root / 'empty.env').write_text('', encoding='ascii')
        token = secrets.token_urlsafe(48)
        secret_file = root / 'service-token'
        secret_file.write_text(token + '\n', encoding='ascii')
        if os.name == 'posix':
            root.chmod(0o700)
            secret_file.chmod(0o644)
        env = isolated_environment(secret_file, image)
        # Explicit empty env file also applies to the backup helper's Compose calls.
        env['COMPOSE_ENV_FILES'] = str(root / 'empty.env')
        project = 'tracemem-check-' + uuid.uuid4().hex
        projects.append(project)
        phase = 'compose_config'
        compose(project, 'config', '--quiet')
        phase = 'image_build'
        print('Building the pinned image.', flush=True)
        compose(project, 'build', '--pull', timeout=600)
        built = True
        report['image_id'] = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])
        report['git_commit'] = run(['git', 'rev-parse', 'HEAD'])
        phase = 'startup'
        compose(project, 'up', '-d', '--no-build', '--wait', '--wait-timeout', '90', timeout=120)
        first_id = compose(project, 'ps', '-q', 'api')
        inspected = json.loads(run(['docker', 'inspect', first_id]))[0]
        report['checks'].update(validate_inspection(inspected, project))
        phase = 'filesystem_permissions'
        compose(project, 'exec', '-T', 'api', 'python', '-c',
                "import os; from pathlib import Path; assert os.getuid()==10001; "
                "p=Path('/data/.permission-check'); p.write_text('synthetic'); p.unlink(); "
                "p=Path('/tmp/permission-check'); p.write_text('synthetic'); p.unlink()")
        report['checks']['non_root_volume_write'] = True
        phase = 'synthetic_seed'
        print('Checking authenticated Add/Search and persistence.', flush=True)
        seed = json.loads(compose(project, 'exec', '-T', 'api', 'python', 'deploy/smoke_test.py', 'seed'))
        if seed.get('state') != 'passed':
            raise ValidationError('synthetic seed did not pass')
        report['checks']['authenticated_add_search'] = True
        phase = 'container_recreation'
        compose(project, 'up', '-d', '--no-build', '--force-recreate', '--wait', '--wait-timeout', '90', timeout=120)
        if compose(project, 'ps', '-q', 'api') == first_id:
            raise ValidationError('container was not replaced')
        verify = json.loads(compose(project, 'exec', '-T', 'api', 'python', 'deploy/smoke_test.py', 'verify'))
        if verify.get('state') != 'passed':
            raise ValidationError('persistent search verification did not pass')
        report['checks']['persistence_after_recreation'] = True
        phase = 'backup_export'
        backup = root / 'synthetic-backup'
        print('Checking backup export and restore to a fresh volume.', flush=True)
        run([sys.executable, '-B', str(ROOT / 'scripts/container_backup.py'), 'backup',
             '--destination', str(backup), '--project', project])
        restored_project = 'tracemem-check-' + uuid.uuid4().hex
        projects.append(restored_project)
        phase = 'restore_new_volume'
        run([sys.executable, '-B', str(ROOT / 'scripts/container_backup.py'), 'restore',
             '--backup', str(backup), '--project', restored_project])
        compose(restored_project, 'up', '-d', '--no-build', '--wait', '--wait-timeout', '90', timeout=120)
        restored = json.loads(compose(restored_project, 'exec', '-T', 'api', 'python', 'deploy/smoke_test.py', 'verify'))
        if restored.get('state') != 'passed':
            raise ValidationError('restored evidence verification did not pass')
        report['checks']['backup_export_and_restore'] = True
        report['state'] = 'passed'
    except (ValidationError, ValueError, KeyError, OSError) as exc:
        report['state'] = 'failed'
        report['failed_phase'] = phase
        report['error'] = str(exc).replace(token, '[redacted]') if token else str(exc)
    finally:
        phase = 'cleanup'
        cleanup_errors = []
        for project in reversed(projects):
            try:
                compose(project, 'down', '--volumes', '--remove-orphans', '--timeout', '10', timeout=60)
            except ValidationError:
                cleanup_errors.append(project)
        if built:
            try:
                run(['docker', 'image', 'rm', image], timeout=60)
            except ValidationError:
                cleanup_errors.append('temporary_image')
        report['owned_resources_cleaned'] = not cleanup_errors
        if cleanup_errors:
            report.update(state='failed', cleanup_failed_resources=cleanup_errors)
        if temp is not None:
            temp.cleanup()
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['state'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
