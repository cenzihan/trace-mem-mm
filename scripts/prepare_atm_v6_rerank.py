"""Prepare private V6 union pairs without models, gold, or reader requests."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.v6_rerank import prepare_project
from scripts.prepare_atm_v5_queries import load_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reranker-spec', '--spec-file', dest='spec_file', required=True, type=Path,
        help='Explicit reranker JSON specification outside the new V6 runtime directory')
    args = parser.parse_args()
    result = prepare_project(reranker_spec=load_json(args.spec_file.read_bytes()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
