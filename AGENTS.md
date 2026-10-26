# Development boundaries

Current user authorization (2026-10-26): upload the reviewed code/results to the
existing private cenzihan/trace-mem-mm repository and prepare Docker deployment.
Publish a clean snapshot with the prior remote main as parent; preserve local
experimental history without uploading its benchmark documentation examples.
Docker files are a deterministic-service preview, not V6 online integration or
a production release. No public endpoint or official evaluation was requested.
The older checkout publication restriction below is superseded for this upload.

Academic multimodal only: text-embedding-v4 and gpt-4o-mini. Offline by default.
Preserve original evidence and user isolation. Never store evaluation questions,
gold, secrets, provider logs, runtime databases or model weights in Git.
Native dataset provenance/license/split/overlap review precedes access. No training
or prompt tuning on evaluation answers. Default-service Qwen reranking is disabled;
the frozen V6 research scripts separately support explicit candidate-only scoring.
No public deployment, enrollment, repository publication or new credentials are
authorized by this checkout. Run python scripts/run_checks.py for offline tests.
