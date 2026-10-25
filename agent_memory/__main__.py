"""Offline demonstration only. No listeners, network calls, or official evaluation."""
import json
import tempfile
import sys
from pathlib import Path

from .store import MemoryStore


def main():
    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    with tempfile.TemporaryDirectory(prefix="aml-synthetic-") as tmp:
        store = MemoryStore(Path(tmp) / "memory.sqlite")
        add = json.loads((fixtures / "add.json").read_text(encoding="utf-8"))
        query = json.loads((fixtures / "search.json").read_text(encoding="utf-8"))
        result = store.add(add)
        found = store.search(query)
        print(json.dumps({"mode": "mock", "official_smoke": False, "add": result,
                          "result_count": len(found["data"]), "top_id": found["data"][0]["id"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv)>1:
        from .cli import main as cli_main
        cli_main()
    else:
        main()
