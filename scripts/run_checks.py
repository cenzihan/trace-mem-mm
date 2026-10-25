"""Run offline tests, including bounded ephemeral loopback HTTP integration."""
import json
import platform
import sys
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    def ids(node):
        if isinstance(node,unittest.TestSuite):
            for child in node:yield from ids(child)
        else:yield node.id()
    modules=dict(Counter(name.split('.')[0] for name in ids(suite)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(read_at_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                  skipped=len(result.skipped), successful=result.wasSuccessful(), mode="offline deterministic retrieval + isolated mock provider tests",
                  official_smoke=False, official_full=False, external_model_calls=0,
                  listener_started=True, listener_scope="127.0.0.1 ephemeral test only; closed on exit", public_binding=False,
                  fixtures="original synthetic PNG/text only",modules=modules)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "offline-tests.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
