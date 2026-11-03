# Repository guidance

## Current implementation

- The final published service is `agent_memory.gpu6_app:create_gpu6_app`, version
  `tracemem-mm-gpu6-v4.3.15-rc7`. Other factories and research configurations
  remain historical or alternative profiles, not the final deployment default.
- Preserve the original sources, user isolation, model identities, receipt
  ownership and conservative accounting of unknown outcomes.
- Do not reintroduce the withdrawn rc8 readiness changes without a separate,
  reviewed fix and isolated concurrency validation.

## Development and validation

- Run `python scripts/run_checks.py` for the offline suite. Report dependency
  skips explicitly; mocked tests do not establish real provider throughput.
- Native dependency checks run in `Dockerfile.mem0` with synthetic inputs and
  disabled networking. Never use protected questions or gold answers as tests.
- Keep deployment changes separate from source publication. Repository updates
  do not authorize touching a running service, state, credentials or lease.
- Keep README concise and focused on the final architecture and local usage.
  Do not add evaluation scores or operational incident timelines to it.

## Publication

- Publish only audited source, synthetic fixtures and aggregate evidence.
  Never commit credentials, provider logs, private error inputs, runtime
  databases, model weights or evaluation content.
- Preserve attribution and third-party licenses.
- Publish a clean snapshot with the prior remote main as parent; retain local
  experimental history locally rather than uploading it with the snapshot.
