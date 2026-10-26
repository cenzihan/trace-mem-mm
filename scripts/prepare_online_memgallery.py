"""Prepare complete source-only native Add plans and original public Search plans.

No gold file, network, provider, model or live budget is opened here. Images
remain explicit native references until a separately verified local media reader
materializes them. Native payloads must stay outside version control.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.native_memgallery import media_reference, plan_rounds


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, questions, destination, *, source_sha256, questions_sha256):
    source, questions, destination = map(Path, (source, questions, destination))
    if destination.exists() or destination.is_symlink():
        raise ValueError('new private output directory required')
    if source.is_symlink() or questions.is_symlink():
        raise ValueError('ordinary verified source and public-question files required')
    if digest(source) != source_sha256 or digest(questions) != questions_sha256:
        raise ValueError('native source/public question hash changed')
    memory = json.loads(source.read_text(encoding='utf-8'))
    public = json.loads(questions.read_text(encoding='utf-8'))
    if not isinstance(memory, dict) or set(memory) != {'multi_session_dialogues'}:
        raise ValueError('source-only projection required; no profile or QA root fields')
    if not isinstance(public, list) or not public:
        raise ValueError('nonempty public-question-only projection required')
    allowed = {'index', 'point', 'question', 'question_image', 'image_caption'}
    if any(not isinstance(row, dict) or set(row)-allowed or 'question' not in row for row in public):
        raise ValueError('public question projection contains undeclared or answer fields')
    namespace = 'native-online-' + source_sha256[:20]
    adds = plan_rounds(memory, namespace)
    if not adds:
        raise ValueError('complete nonempty native source scene required')
    searches, seen = [], set()
    refs = {image['relative_path'] for plan in adds for image in plan['provenance']['images']}
    for row in public:
        if type(row.get('index')) is not int or row['index'] in seen:
            raise ValueError('unique original public question indices required')
        seen.add(row['index'])
        text = row['question']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('original nonempty question text required')
        parts = [dict(type='text', text=text)]
        image = row.get('question_image')
        if image:
            relative = media_reference(image)
            refs.add(relative)
            parts.append(dict(type='native_image_ref', relative_path=relative))
        caption = row.get('image_caption')
        if caption:
            if not isinstance(caption, str):
                raise ValueError('original author caption must be text')
            parts.append(dict(type='text', text=caption))
        searches.append(dict(question_index=row['index'], point=row.get('point'),
                             payload=dict(user_id=namespace, query=parts, top_k=100)))
    destination.mkdir(parents=True, mode=0o700)
    for name, values in (('add-plans.jsonl', adds), ('search-plans.jsonl', searches)):
        with (destination/name).open('x', encoding='utf-8') as target:
            for value in values:
                target.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
    with (destination/'media-references.json').open('x', encoding='utf-8') as target:
        json.dump(sorted(refs), target, ensure_ascii=False, indent=2)
    report = dict(state='native_plans_frozen_not_executed', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=source_sha256, questions_sha256=questions_sha256,
                  source_sessions=len(memory['multi_session_dialogues']), source_rounds=len(adds),
                  source_messages=sum(len(plan['messages']) for plan in adds),
                  public_questions=len(searches), source_image_occurrences=sum(len(plan['provenance']['images']) for plan in adds),
                  query_image_occurrences=sum(bool(row.get('question_image')) for row in public),
                  distinct_media_references=len(refs), all_distractors_retained=True,
                  gold_opened=False, model_requests=0, raw_images_materialized=False,
                  scoring='engineering regression on an earlier real scene, not unseen benchmark evidence',
                  artifacts={p.name:digest(p) for p in destination.iterdir() if p.is_file()})
    with (destination/'PREPARATION.json').open('x', encoding='utf-8') as target:
        json.dump(report, target, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--questions-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.questions, args.destination,
                     source_sha256=args.source_sha256, questions_sha256=args.questions_sha256), indent=2))
