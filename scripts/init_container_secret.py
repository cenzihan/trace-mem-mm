"""Create a new local service authentication token without displaying it.

This is not an OpenAI/Bailian key. Existing files are never replaced. On POSIX,
the parent is private (0700); its file is readable by the container's UID after
the explicit Compose secret bind mount (0644). Windows inherits directory ACLs.
"""
import os
from pathlib import Path
import secrets


def main():
    directory = Path(__file__).resolve().parents[1] / '.secrets'
    if directory.is_symlink():
        raise ValueError('secret directory must not be a symlink')
    directory.mkdir(mode=0o700, exist_ok=True)
    if os.name == 'posix':
        directory.chmod(0o700)
    path = directory / 'service_token'
    with path.open('x', encoding='ascii') as target:
        target.write(secrets.token_urlsafe(48) + '\n')
    if os.name == 'posix':
        path.chmod(0o644)
    print('Created local Compose service token; its value was not printed.')


if __name__ == '__main__':
    main()
