"""First freeze V6 rankings with `rank`; score gold in a separate `score` process."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.v6_rerank import rank_project, score_project


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('rank', 'score'))
    args = parser.parse_args()
    result = rank_project() if args.phase == 'rank' else score_project()
    print(json.dumps(result, ensure_ascii=False, indent=2))
