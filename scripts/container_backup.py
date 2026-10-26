"""Back up a Compose service, or restore into an unused Compose project.

Only SQLite service data is handled. No model calls, service startup, or published
ports are involved in restore. Process output is captured and never displayed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import uuid


ROOT = Path(__file__).resolve().parents[1]
DATABASE = "database.sqlite"
MANIFEST = "manifest.json"
APPLICATION_ID = 0x414D4D32
PROJECT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")


class BackupError(Exception):
    """An operator-safe error; never include raw subprocess output."""


def run(command):
    try:
        # Docker exec must not consume a caller's piped/heredoc shell program.
        result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise BackupError("Required local command could not complete.") from None
    if result.returncode != 0:
        raise BackupError("A Docker or Git operation failed; process output was withheld.")
    if len(result.stdout) > 131072:
        raise BackupError("Unexpectedly large command response.")
    return result.stdout.strip()


def project_name(value):
    if not isinstance(value, str) or not PROJECT_PATTERN.fullmatch(value):
        raise BackupError("Use a project name of 1..63 lowercase letters, digits, hyphens or underscores, starting with a letter or digit.")
    return value


def compose(project, *arguments):
    command = ["docker", "compose", "--project-directory", str(ROOT),
               "--file", str(ROOT / "compose.yaml")]
    # The container verifier supplies an explicit empty file to isolate tests
    # from a checkout's .env. Translate this setting to CLI flags as well.
    env_files = os.environ.get("COMPOSE_ENV_FILES")
    if env_files is not None:
        files = [value.strip() for value in env_files.split(",") if value.strip()]
        if not files:
            raise BackupError("COMPOSE_ENV_FILES must name at least one environment file.")
        for filename in files:
            command.extend(["--env-file", filename])
    return [*command, "--project-name", project_name(project), *arguments]


def private_directory(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise BackupError("Backup destination must be a new directory; nothing is overwritten.")
    if not path.parent.is_dir():
        raise BackupError("Backup destination parent directory must already exist.")
    path.mkdir(mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_database(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise BackupError("Backup database must be a regular file, not a symbolic link.")
    # A standalone backup must never depend on an unverified SQLite sidecar.
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise BackupError("Backup database must not have SQLite sidecar files.")
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        kind = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if kind != APPLICATION_ID or version not in (1, 2, 3):
            raise BackupError("Database is not a supported TraceMem service database.")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"requests", "memories", "evidence"} <= tables or tables & {"budget", "attempts", "smoke_attempts"}:
            raise BackupError("Database schema is not service memory storage.")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise BackupError("SQLite integrity verification failed.")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise BackupError("SQLite foreign-key verification failed.")
        if "cards" in tables:
            invalid = connection.execute("SELECT count(*) FROM cards c JOIN memories m ON c.memory_id=m.id WHERE c.user_id!=m.user_id OR c.session_id!=m.session_id").fetchone()[0]
            if invalid:
                raise BackupError("Database user isolation verification failed.")
    except sqlite3.Error:
        raise BackupError("SQLite backup could not be verified.") from None
    finally:
        if connection is not None:
            connection.close()
    return {"application_id": kind, "schema_version": version,
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def source_identity(project):
    containers = run(compose(project, "ps", "--quiet", "api")).splitlines()
    if len(containers) != 1 or not re.fullmatch(r"[a-f0-9]{12,64}", containers[0]):
        raise BackupError("Backup requires exactly one running API container.")
    image = run(["docker", "inspect", "--format", "{{.Image}}", containers[0]])
    revision = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = bool(run(["git", "-C", str(ROOT), "status", "--porcelain"]))
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", revision):
        raise BackupError("Image or checkout identity could not be verified.")
    return {"image_id": image, "git_revision": revision,
            "git_revision_scope": "operator_checkout", "git_dirty": dirty}


def backup(destination, project="tracemem"):
    project_name(project)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise BackupError("Backup destination must be a new directory; nothing is overwritten.")
    identity = source_identity(project)
    destination = private_directory(destination)
    remote = "/data/backup-" + uuid.uuid4().hex + ".sqlite"
    run(compose(project, "exec", "-T", "api", "python", "-m", "agent_memory",
                "--db", "/data/system.sqlite", "backup", remote))
    database = destination / DATABASE
    # The destination directory was exclusively created above and is private.
    run(compose(project, "cp", "api:" + remote, str(database)))
    details = validate_database(database)
    if os.name == "posix":
        database.chmod(0o600)
    manifest = {"format_version": 1, "database": DATABASE,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_project": project, **details, **identity}
    with (destination / MANIFEST).open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    if os.name == "posix":
        (destination / MANIFEST).chmod(0o600)
    cleanup = True
    try:
        # Only this operation's UUID temporary backup can be removed. No glob,
        # shell, recursive removal, or service database path is accepted.
        code = ("import pathlib,re,sys; p=sys.argv[1]; "
                "assert re.fullmatch(r'/data/backup-[a-f0-9]{32}\\.sqlite',p); "
                "pathlib.Path(p).unlink()")
        run(compose(project, "exec", "-T", "api", "python", "-c", code, remote))
    except BackupError:
        cleanup = False
    return {"status": "backed_up", "destination": str(destination),
            "sha256": details["sha256"], "bytes": details["bytes"],
            "container_temporary_backup_removed": cleanup}


def read_backup(directory):
    directory = Path(directory).absolute()
    manifest_path = directory / MANIFEST
    if directory.is_symlink() or not directory.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise BackupError("A regular backup directory and manifest are required.")
    if manifest_path.stat().st_size > 16384:
        raise BackupError("Backup manifest is too large.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError):
        raise BackupError("Backup manifest is invalid.") from None
    if (not isinstance(manifest, dict) or manifest.get("format_version") != 1
            or manifest.get("database") != DATABASE
            or type(manifest.get("bytes")) is not int or manifest["bytes"] < 1
            or not isinstance(manifest.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", manifest["sha256"])
            or not isinstance(manifest.get("image_id"), str)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest["image_id"])
            or not isinstance(manifest.get("git_revision"), str)
            or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", manifest["git_revision"])):
        raise BackupError("Backup manifest fields are invalid.")
    project_name(manifest.get("source_project"))
    database = directory / DATABASE
    details = validate_database(database)
    if any(manifest.get(key) != details[key] for key in details):
        raise BackupError("Backup hash, size, or schema does not match its manifest.")
    return database, manifest


def ensure_unused_project(project):
    label = "label=com.docker.compose.project=" + project
    if run(["docker", "ps", "--all", "--quiet", "--filter", label]):
        raise BackupError("Restore target already has containers; use a new project name.")
    if run(["docker", "volume", "ls", "--quiet", "--filter", label]):
        raise BackupError("Restore target already has volumes; use a new project name.")
    expected = project + "_memory-data"
    named = run(["docker", "volume", "ls", "--quiet", "--filter", "name=" + expected]).splitlines()
    if expected in named:
        raise BackupError("Restore target data volume already exists, even without Compose labels.")


def restore(directory, project):
    project_name(project)
    database, manifest = read_backup(directory)
    if project == manifest["source_project"]:
        raise BackupError("Restore requires a different, unused Compose project name.")
    ensure_unused_project(project)
    with tempfile.TemporaryDirectory(prefix="tracemem-restore-") as temporary:
        staged = Path(temporary) / DATABASE
        shutil.copyfile(database, staged)
        if os.name == "posix":
            Path(temporary).chmod(0o700)
            staged.chmod(0o644)
        # Recheck the isolated copy, including its hash, before any Docker write.
        details = validate_database(staged)
        if any(manifest.get(key) != details[key] for key in details):
            raise BackupError("Backup changed during restore preparation.")
        ensure_unused_project(project)
        # Bind only the file: the private host parent remains 0700 and is never
        # traversed by the container's UID. No service ports are published.
        mount = str(staged.resolve()) + ":/restore/database.sqlite:ro"
        code = ("from agent_memory.maintenance import backup_database; "
                "backup_database('/restore/database.sqlite','/data/system.sqlite')")
        run(compose(project, "run", "--rm", "--no-deps", "--pull", "never",
                    "--volume", mount, "api", "python", "-c", code))
    return {"status": "restored", "project": project,
            "sha256": manifest["sha256"], "service_started": False,
            "ports_published": False, "original_project_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    save = subcommands.add_parser("backup", help="Create a new private backup directory")
    save.add_argument("--destination", required=True, type=Path)
    save.add_argument("--project", default="tracemem")
    load = subcommands.add_parser("restore", help="Restore into a new project without starting its API")
    load.add_argument("--backup", required=True, type=Path)
    load.add_argument("--project", required=True)
    args = parser.parse_args(argv)
    try:
        result = backup(args.destination, args.project) if args.action == "backup" else restore(args.backup, args.project)
    except BackupError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error):
        print("Local backup file operation failed; details were withheld.", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
