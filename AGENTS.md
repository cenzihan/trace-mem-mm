# Development boundaries

Academic multimodal only: text-embedding-v4 and gpt-4o-mini. Offline by default.
Preserve original evidence and user isolation. Never store evaluation questions,
gold, secrets, provider logs, runtime databases or model weights in Git.
Native dataset provenance/license/split/overlap review precedes access. No training
or prompt tuning on evaluation answers. Qwen candidate reranking is disabled.
No public deployment, enrollment, repository publication or new credentials are
authorized by this checkout. Run python scripts/run_checks.py for offline tests.
