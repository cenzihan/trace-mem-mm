"""Synthetic SQLite recovery contracts with mocked Docker process boundaries."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_memory.maintenance import backup_database
from agent_memory.system import AgentMemorySystem
from scripts import container_backup as backup


class DockerFixture:
    def __init__(self, root, source):
        self.root, self.source = root, source
        self.commands = []
        self.remote = {}
        self.restored = {}
        self.target_containers = ""
        self.target_volumes = ""
        self.unlabelled_volume = ""
        self.fail_cleanup = False
        self.fail_with_secret = False
        self.before_restore = None

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if kwargs.get("shell"):
            raise AssertionError("No shell is allowed")
        if kwargs.get("stdin") != subprocess.DEVNULL:
            raise AssertionError("Noninteractive commands must not consume operator stdin")
        self.assert_capture = kwargs["capture_output"] and kwargs["text"]
        out, code = "", 0
        if self.fail_with_secret:
            return subprocess.CompletedProcess(command, 1, "sensitive-token-output", "sensitive-token-error")
        if command[0] == "git":
            out = "a" * 40 if "rev-parse" in command else ""
        elif command[:2] == ["docker", "inspect"]:
            out = "sha256:" + "b" * 64
        elif command[:2] == ["docker", "ps"]:
            out = self.target_containers
        elif command[:3] == ["docker", "volume", "ls"]:
            out = self.target_volumes if command[-1].startswith("label=") else self.unlabelled_volume
        elif command[:2] == ["docker", "compose"]:
            position = command.index("--project-name")
            project, args = command[position + 1], command[position + 2:]
            if args[0] == "ps":
                out = "c" * 64
            elif args[0] == "exec" and "backup" in args:
                remote = args[-1]
                local = self.root / (Path(remote).name)
                backup_database(self.source, local)
                self.remote[remote] = local
            elif args[0] == "cp":
                shutil.copyfile(self.remote[args[1][4:]], args[2])
            elif args[0] == "exec" and "-c" in args:
                self.assert_cleanup_path = args[-1]
                if self.fail_cleanup:
                    code = 1
                else:
                    self.remote.pop(args[-1]).unlink()
            elif args[0] == "run":
                if self.before_restore:
                    self.before_restore()
                mounted = args[args.index("--volume") + 1]
                source_path = Path(mounted.removesuffix(":/restore/database.sqlite:ro"))
                if not source_path.is_file():
                    raise AssertionError("A single readable file must be mounted")
                if os.name == "posix":
                    self.assert_stage_mode = stat.S_IMODE(source_path.stat().st_mode)
                    self.assert_stage_parent_mode = stat.S_IMODE(source_path.parent.stat().st_mode)
                destination = self.root / (project + "-restored.sqlite")
                backup_database(source_path, destination)
                self.restored[project] = destination
            else:
                raise AssertionError("Unexpected compose action")
        else:
            raise AssertionError("Unexpected external command")
        return subprocess.CompletedProcess(command, code, out, "")


class ContainerBackupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tracemem-backup-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.sqlite"
        self.memory = AgentMemorySystem(self.source)
        self.memory.add({"user_id": "synthetic-user", "session_id": "synthetic-session",
                         "request_id": "synthetic-request", "messages": [{"role": "user",
                         "content": "Private synthetic cobalt source; never print in manifests."}]})
        self.fake = DockerFixture(self.root, self.source)
        self.process = self.enterContext(patch.object(backup.subprocess, "run", side_effect=self.fake))
        self.env = self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.destination = self.root / "backup-one"

    def create_backup(self):
        result = backup.backup(self.destination)
        self.assertEqual(result["status"], "backed_up")
        return json.loads((self.destination / backup.MANIFEST).read_text())

    def test_backup_creates_verified_private_manifest_without_content(self):
        manifest = self.create_backup()
        self.assertEqual(manifest["image_id"], "sha256:" + "b" * 64)
        self.assertEqual(manifest["git_revision"], "a" * 40)
        self.assertEqual(manifest["git_revision_scope"], "operator_checkout")
        self.assertEqual(manifest["source_project"], "tracemem")
        text = (self.destination / backup.MANIFEST).read_text()
        self.assertNotIn("cobalt", text)
        self.assertNotIn("synthetic-user", text)
        backup.read_backup(self.destination)
        with contextlib.closing(sqlite3.connect(self.destination / backup.DATABASE)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM memories").fetchone()[0], 1)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o700)
            for name in (backup.DATABASE, backup.MANIFEST):
                self.assertEqual(stat.S_IMODE((self.destination / name).stat().st_mode), 0o600)
        self.assertTrue(self.fake.assert_capture)
        self.assertNotIn("/data/system.sqlite", self.fake.assert_cleanup_path)
        self.assertEqual(self.fake.remote, {})

    def test_existing_destination_is_never_overwritten_or_contacted(self):
        self.destination.mkdir()
        sentinel = self.destination / "keep"
        sentinel.write_text("untouched")
        with self.assertRaises(backup.BackupError):
            backup.backup(self.destination)
        self.process.assert_not_called()
        self.assertEqual(sentinel.read_text(), "untouched")

    def test_backup_cleanup_failure_preserves_valid_local_backup(self):
        self.fake.fail_cleanup = True
        result = backup.backup(self.destination)
        self.assertFalse(result["container_temporary_backup_removed"])
        backup.read_backup(self.destination)
        self.assertEqual(len(self.fake.remote), 1)

    def test_restore_recovers_real_sqlite_to_unused_project_without_startup(self):
        self.create_backup()
        source_hash = backup.sha256_file(self.source)
        result = backup.restore(self.destination, "fresh-restore")
        self.assertFalse(result["service_started"])
        self.assertFalse(result["ports_published"])
        recovered = self.fake.restored["fresh-restore"]
        with contextlib.closing(sqlite3.connect(recovered)) as connection:
            value = connection.execute("SELECT content FROM memories").fetchone()[0]
            self.assertIn("cobalt", value)
        self.assertEqual(backup.sha256_file(self.source), source_hash)
        restore_commands = [cmd for cmd in self.fake.commands if "--volume" in cmd]
        self.assertEqual(len(restore_commands), 1)
        self.assertIn("--no-deps", restore_commands[0])
        self.assertNotIn("--service-ports", restore_commands[0])
        self.assertNotIn("up", restore_commands[0])
        if os.name == "posix":
            self.assertEqual(self.fake.assert_stage_mode, 0o644)
            self.assertEqual(self.fake.assert_stage_parent_mode, 0o700)

    def test_bad_hash_refuses_restore_before_docker(self):
        manifest = self.create_backup()
        manifest["sha256"] = "0" * 64
        (self.destination / backup.MANIFEST).write_text(json.dumps(manifest))
        self.process.reset_mock()
        with self.assertRaisesRegex(backup.BackupError, "does not match"):
            backup.restore(self.destination, "unused")
        self.process.assert_not_called()

    def test_corrupt_sqlite_refuses_restore_even_with_matching_hash(self):
        manifest = self.create_backup()
        path = self.destination / backup.DATABASE
        path.write_bytes(b"not a database" * 100)
        manifest.update(bytes=path.stat().st_size, sha256=backup.sha256_file(path))
        (self.destination / backup.MANIFEST).write_text(json.dumps(manifest))
        self.process.reset_mock()
        with self.assertRaises(backup.BackupError):
            backup.restore(self.destination, "unused")
        self.process.assert_not_called()

    def test_provider_budget_database_refused(self):
        self.create_backup()
        path = self.destination / backup.DATABASE
        with contextlib.closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE attempts(id INTEGER)")
            connection.commit()
        with self.assertRaisesRegex(backup.BackupError, "not service memory"):
            backup.validate_database(path)

    def test_cross_user_card_corruption_refused(self):
        self.create_backup()
        path = self.destination / backup.DATABASE
        with contextlib.closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE cards SET user_id='different-owner'")
            connection.commit()
        with self.assertRaisesRegex(backup.BackupError, "user isolation"):
            backup.validate_database(path)

    def test_sidecars_refused_before_restore(self):
        self.create_backup()
        Path(str(self.destination / backup.DATABASE) + "-wal").write_bytes(b"unverified")
        with self.assertRaisesRegex(backup.BackupError, "sidecar"):
            backup.restore(self.destination, "unused")

    def test_existing_container_or_volume_refuses_restore(self):
        self.create_backup()
        for attribute, value in (("target_containers", "container-id"),
                                 ("target_volumes", "named-volume"),
                                 ("unlabelled_volume", "unused_memory-data")):
            with self.subTest(attribute=attribute):
                setattr(self.fake, attribute, value)
                with self.assertRaises(backup.BackupError):
                    backup.restore(self.destination, "unused")
                setattr(self.fake, attribute, "")
        self.assertEqual(self.fake.restored, {})

    def test_original_project_refused_even_if_not_running(self):
        self.create_backup()
        self.process.reset_mock()
        with self.assertRaisesRegex(backup.BackupError, "different"):
            backup.restore(self.destination, "tracemem")
        self.process.assert_not_called()

    def test_restore_detects_backup_changed_during_staging(self):
        self.create_backup()
        real_copy = shutil.copyfile
        def altered_copy(source, destination):
            real_copy(source, destination)
            with contextlib.closing(sqlite3.connect(destination)) as connection:
                connection.execute("UPDATE memories SET content='changed synthetic content'")
                connection.commit()
        with patch.object(backup.shutil, "copyfile", side_effect=altered_copy):
            with self.assertRaisesRegex(backup.BackupError, "changed during"):
                backup.restore(self.destination, "unused")
        self.assertEqual(self.fake.restored, {})

    def test_manifest_cannot_redirect_database_outside_backup(self):
        manifest = self.create_backup()
        manifest["database"] = "../source.sqlite"
        (self.destination / backup.MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(backup.BackupError, "fields are invalid"):
            backup.read_backup(self.destination)

    def test_explicit_environment_file_preserved_without_repository_env(self):
        explicit = str(self.root / "empty.env")
        os.environ["COMPOSE_ENV_FILES"] = explicit
        command = backup.compose("fresh", "ps")
        self.assertEqual(command[command.index("--env-file") + 1], explicit)
        self.assertEqual(command[command.index("--file") + 1], str(backup.ROOT / "compose.yaml"))
        os.environ["COMPOSE_ENV_FILES"] = ""
        with self.assertRaises(backup.BackupError):
            backup.compose("fresh", "ps")

    def test_cli_hides_subprocess_output_and_leaves_source_untouched(self):
        self.fake.fail_with_secret = True
        stdout, stderr = io.StringIO(), io.StringIO()
        source_hash = backup.sha256_file(self.source)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = backup.main(["backup", "--destination", str(self.destination)])
        self.assertEqual(code, 1)
        self.assertNotIn("sensitive-token", stdout.getvalue() + stderr.getvalue())
        self.assertFalse(self.destination.exists())
        self.assertEqual(source_hash, backup.sha256_file(self.source))

    def test_invalid_project_rejected_without_external_commands(self):
        for name in ("", "UPPER", "-option", "path/name", "a;rm", "x" * 64):
            with self.subTest(name=name), self.assertRaises(backup.BackupError):
                backup.backup(self.destination, name)
        self.process.assert_not_called()


class BackupStdinRegressionTests(unittest.TestCase):
    def test_child_cannot_swallow_following_operator_commands(self):
        # Use real subprocess pipes rather than Docker: the outer process models
        # a shell reading a heredoc, and its child tries to drain inherited stdin.
        script = (
            "import json, sys\n"
            "from scripts.container_backup import run\n"
            "child = run([sys.executable, '-c', "
            "'import sys; print(len(sys.stdin.buffer.read()))'])\n"
            "remaining = sys.stdin.read()\n"
            "print(json.dumps({'child_bytes': int(child), 'remaining': remaining}))\n"
        )
        following_commands = 'echo backup-finished\necho next-step\n'
        result = subprocess.run(
            [sys.executable, '-B', '-c', script], cwd=backup.ROOT,
            input=following_commands, capture_output=True, text=True, timeout=15,
            check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {'child_bytes': 0, 'remaining': following_commands})


if __name__ == "__main__":
    unittest.main()
