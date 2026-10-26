"""One preselected complete scene; quarantine and split before any model access.

The only network request obtains the pinned native container. No provider, image,
credential or scorer is invoked. Existing preparation is never overwritten.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_memory.native_memgallery import plan_rounds

REVISION = 'af912daba984e896e253016b7c7e334ef92c2a6f'
SCENE = 'data/dialog/Architecture_Art_Culture_Exhibition_Technology.json'
SCENE_BYTES = 126837
SCENE_OID = '11451cd23f4a3207cf8120128b30ae331a9b3e1f'
EXCLUDED = ('data/dialog/AI_Robotics_Automation_Future_Tech.json',
            'data/dialog/Academic_Animal_Pet_Research_Life.json')
WORK = 'runtime/memgallery-v7'
REPORT = 'reports/memgallery-v7-preparation.json'
METADATA = 'data/metadata-20261025/memgallery-tree.json'
SEED = 20261026
VARIANTS = ('dense', 'dense_reranked', 'hybrid', 'hybrid_reranked')
QUESTION_KEYS = frozenset(('point', 'question', 'question_image', 'image_caption'))
GOLD_KEYS = frozenset(('answer', 'clue', 'session_id'))


class PreparationError(ValueError):
    def __init__(self, reason, schema=None):
        super().__init__(reason)
        self.reason = reason
        self.schema = schema or {}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_load(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise PreparationError('duplicate_json_key')
            out[key] = value
        return out
    def constant(_):
        raise PreparationError('nonfinite_json_value')
    def finite(number):
        value = float(number)
        if not math.isfinite(value):
            raise PreparationError('nonfinite_json_value')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)


def save_new(path, value, *, raw=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value if raw else canonical(value) + b'\n'
    with path.open('xb') as file:
        file.write(body)
        file.flush()
        os.fsync(file.fileno())
    return digest(body)


def field_shapes(rows):
    counts = collections.defaultdict(collections.Counter)
    for row in rows:
        if isinstance(row, dict):
            for key, value in row.items():
                counts[key][type(value).__name__] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(counts.items())}


def image_field(value):
    return value is None or isinstance(value, str) or (
        isinstance(value, list) and all(isinstance(x, str) for x in value))


def nonempty_image(value):
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, list) and any(x.strip() for x in value)


def split_container(container):
    if not isinstance(container, dict):
        raise PreparationError('container_object_required')
    allowed = {'character_profile', 'multi_session_dialogues', 'human-annotated QAs'}
    unknown = set(container) - allowed
    if unknown:
        raise PreparationError('unknown_container_fields',
                               {k: type(container[k]).__name__ for k in sorted(unknown)})
    sessions = container.get('multi_session_dialogues')
    rows = container.get('human-annotated QAs')
    if not isinstance(sessions, list) or not sessions or not isinstance(rows, list) or not rows:
        raise PreparationError('nonempty_source_and_question_arrays_required')
    shape = field_shapes(rows)
    unknown = set(shape) - QUESTION_KEYS - GOLD_KEYS
    if unknown:
        raise PreparationError('unclassified_question_fields', {k: shape[k] for k in sorted(unknown)})
    questions, gold = [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not {'point', 'question', *GOLD_KEYS} <= set(row):
            raise PreparationError('required_question_or_reference_fields_missing', shape)
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ('point', 'question')):
            raise PreparationError('question_or_point_type_invalid', shape)
        if any(k in row and not image_field(row[k]) for k in ('question_image', 'image_caption')):
            raise PreparationError('question_image_or_caption_type_invalid', shape)
        if not isinstance(row['answer'], str) or any(
                not isinstance(row[k], list) or any(not isinstance(x, str) for x in row[k])
                for k in ('clue', 'session_id')):
            raise PreparationError('reference_field_type_invalid', shape)
        has_image = nonempty_image(row.get('question_image'))
        has_caption = nonempty_image(row.get('image_caption'))
        if has_image and not has_caption:
            raise PreparationError('query_image_requires_nonempty_caption', shape)
        if has_caption and not has_image:
            raise PreparationError('query_caption_without_image_reference', shape)
        # Explicit, pre-frozen normalization of absent/empty optional fields only.
        # Nonempty values are preserved byte-for-byte as their original JSON types.
        questions.append(dict(index=index, point=row['point'], question=row['question'],
                              question_image=row['question_image'] if has_image else None,
                              image_caption=row['image_caption'] if has_caption else None))
        gold.append(dict(index=index, **{k: row[k] for k in row if k in GOLD_KEYS}))
    # A source projection contains the complete original dialogue objects only.
    # Reject embedded annotation keys rather than silently deleting source data.
    def source_check(value):
        if isinstance(value, dict):
            blocked = set(value) & {'human-annotated QAs', 'question', 'answer', 'clue', 'evidence_ids'}
            if blocked:
                raise PreparationError('annotation_field_inside_source',
                                       {k: type(value[k]).__name__ for k in sorted(blocked)})
            for item in value.values():
                source_check(item)
        elif isinstance(value, list):
            for item in value:
                source_check(item)
    source_check(sessions)
    memory = {'multi_session_dialogues': sessions}
    plans = plan_rounds(memory, 'memgallery:v7:' + REVISION + ':' + Path(SCENE).stem)
    if not plans:
        raise PreparationError('nonempty_complete_source_required')
    source_images = {image['relative_path'] for plan in plans for image in plan['provenance']['images']}
    counts = dict(source_sessions=len(sessions), source_rounds=len(plans), source_messages=2 * len(plans),
                  source_unique_image_references=len(source_images),
                  source_image_occurrences=sum(len(plan['provenance']['images']) for plan in plans),
                  questions=len(questions), point_counts=dict(sorted(collections.Counter(q['point'] for q in questions).items())),
                  vs_questions=sum(q['point'] == 'VS' for q in questions),
                  questions_with_image=sum(nonempty_image(q.get('question_image')) for q in questions),
                  questions_with_image_caption=sum(nonempty_image(q.get('image_caption')) for q in questions))
    return memory, questions, gold, dict(question_field_types=shape, **counts)


def select_reader(metadata, count=16):
    if type(count) is not int or count < 1 or len(metadata) < count:
        raise PreparationError('insufficient_questions_for_fixed_reader_count')
    if any(set(row) != {'index', 'point'} or type(row['index']) is not int
           or row['index'] < 0 or not isinstance(row['point'], str) or not row['point']
           for row in metadata):
        raise PreparationError('selection_accepts_only_index_and_point')
    if len({row['index'] for row in metadata}) != len(metadata):
        raise PreparationError('duplicate_question_index')
    rank = lambda row: (digest(canonical([SEED, SCENE, row['point'], row['index']])), row['index'])
    groups = collections.defaultdict(list)
    for row in metadata:
        groups[row['point']].append(row)
    points = sorted(groups, key=lambda point: (digest(canonical([SEED, SCENE, point])), point))
    selected = [min(groups[point], key=rank) for point in points[:count]]
    indices = {row['index'] for row in selected}
    selected.extend(sorted((row for row in metadata if row['index'] not in indices), key=rank)[:count-len(selected)])
    return dict(selected_indices=[row['index'] for row in selected],
                selected_point_counts=dict(sorted(collections.Counter(row['point'] for row in selected).items())),
                reader_questions=count, planned_reader_answers=count * len(VARIANTS))


def selection_rule(root):
    metadata_path = root / METADATA
    raw = metadata_path.read_bytes()
    tree = strict_load(raw)
    rows = sorted((x for x in tree if x.get('type') == 'file' and x.get('path', '').startswith('data/dialog/')
                   and x['path'].endswith('.json')), key=lambda x: x['path'])
    if len(rows) != 20 or len({x['path'] for x in rows}) != len(rows):
        raise PreparationError('complete_unique_20_scene_metadata_required')
    chosen = next((x for x in rows if x['path'] not in EXCLUDED), None)
    if chosen != {'type': 'file', 'oid': SCENE_OID, 'size': SCENE_BYTES, 'path': SCENE}:
        raise PreparationError('fixed_scene_metadata_mismatch')
    codes = ('scripts/prepare_memgallery_v7.py', 'agent_memory/native_memgallery.py')
    return dict(state='selection_frozen_before_native_download', revision=REVISION, chosen=SCENE,
                expected_bytes=SCENE_BYTES, expected_git_blob_oid=SCENE_OID, excluded=list(EXCLUDED),
                scene_rule='lexicographic native path; exclude both previously read/evaluated scenes',
                metadata_file=METADATA, metadata_sha256=digest(raw), seed=SEED,
                reader_count=16, reader_selection='hash-stratified by point: point order SHA256(JSON[seed,scene,point]); '
                'one lowest SHA256(JSON[seed,scene,point,index]) in each point, then lowest remaining item hashes to 16; '
                'if more than 16 points use first 16 point hashes; JSON canonical UTF8 compact sort_keys',
                representation='complete original user and assistant text plus dataset-provided captions only',
                representation_scope='text condition; provided captions retained with upstream provenance; no fresh visual model',
                raw_media=0, variants=list(VARIANTS), retrieval_candidate_k=100, reader_top_k=20,
                reader_budget=dict(text_tokens=8192, utf8_bytes=24000, images=0, output_tokens=768),
                source_question_gold_separation=True, question_text_used_for_selection=False,
                question_schema=dict(fields=['index', 'question', 'point', 'question_image', 'image_caption'],
                    absent_or_empty_image_fields='explicit null/null; original quarantined bytes retained',
                    nonempty_image_or_caption_values='preserve original JSON type and value',
                    image_without_caption='block', caption_without_image='block', unknown_fields='block and report field types'),
                preprocessing_models_called=[], provider_requests=0, gpu_calls=0,
                code_hashes={name: digest((root / name).read_bytes()) for name in codes},
                license='MIT declared in the pinned HF dataset card; linked HF LICENSE is absent (404)',
                public_redistribution=False, frozen_old_experiments_modified=False)


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co' or parsed.username or parsed.password:
            raise PreparationError('native_container_redirect_not_allowed')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_container(url):
    opener = urllib.request.build_opener(OfficialRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    with opener.open(urllib.request.Request(url, headers={'User-Agent': 'TraceMem-MM-pinned-V7-preparation'}), timeout=30) as response:
        return response.read(SCENE_BYTES + 1)


def prepare(root=ROOT, *, fetcher=None):
    root = Path(root).resolve()
    work = root / WORK
    if work.exists() or (root / REPORT).exists():
        raise PreparationError('preparation_already_claimed_do_not_reselect_or_repeat')
    rule = selection_rule(root)
    work.mkdir(parents=True, exist_ok=False)
    rule_hash = save_new(work / 'FROZEN_SELECTION_RULE.json', rule)
    try:
        url = f'https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery/raw/{REVISION}/{SCENE}'
        raw = (fetcher or fetch_container)(url)
        if not isinstance(raw, bytes) or len(raw) != SCENE_BYTES:
            raise PreparationError('native_container_size_mismatch')
        oid = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
        if oid != SCENE_OID:
            raise PreparationError('native_container_git_blob_mismatch')
        raw_hash = save_new(work / 'quarantine/scene.raw.json', raw, raw=True)
        memory, questions, gold, schema = split_container(strict_load(raw))
        selection = select_reader([{'index': q['index'], 'point': q['point']} for q in questions])
        selection.update(rule_file=WORK + '/FROZEN_SELECTION_RULE.json', rule_sha256=rule_hash,
                         seed=SEED, provider_requests=0)
        products = {'memory-source.json': memory, 'questions-public.json': questions,
                    'scorer-only/gold.json': gold, 'READER_SELECTION.json': selection}
        hashes = {WORK + '/' + name: save_new(work / name, value) for name, value in products.items()}
        public_hashes = {name: value for name, value in hashes.items() if '/scorer-only/' not in name}
        report = dict(state='prepared_complete_scene_no_model_calls', work=WORK, revision=REVISION,
            selection_rule_sha256=rule_hash, quarantine_sha256=raw_hash,
            scene_bytes=len(raw), scene_git_blob_oid=oid, schema=schema,
            reader_questions=selection['reader_questions'], planned_reader_answers=selection['planned_reader_answers'],
            selected_point_counts=selection['selected_point_counts'],
            selected_with_image=sum(nonempty_image(questions[i].get('question_image')) for i in selection['selected_indices']),
            selected_with_image_caption=sum(nonempty_image(questions[i].get('image_caption')) for i in selection['selected_indices']),
            selected_vs_questions=sum(questions[i]['point'] == 'VS' for i in selection['selected_indices']),
            public_artifact_hashes=public_hashes, scorer_only_gold_sha256=hashes[WORK + '/scorer-only/gold.json'],
            schema_anomalies=[], raw_media_downloads=0, provider_requests=0, gpu_calls=0,
            gold_values_exposed=False, original_datasets_modified=False,
            scope='new project-scene validation on public benchmark; not an official or globally unseen test')
        save_new(work / 'PREPARATION.json', report)
        save_new(root / REPORT, report)
        return report
    except Exception as exc:
        failure = dict(state='stopped_preserve_frozen_selection_no_retry', exception_type=type(exc).__name__,
                       selection_rule_sha256=rule_hash, provider_requests=0, raw_media_downloads=0)
        if isinstance(exc, PreparationError):
            failure.update(reason=exc.reason, schema=exc.schema)
        save_new(work / 'PREPARATION_FAILED.json', failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='freeze the selection and acquire this one pinned container')
    args = parser.parse_args()
    if not args.prepare:
        parser.error('--prepare is required')
    try:
        print(json.dumps(prepare(), ensure_ascii=True, sort_keys=True))
    except Exception as exc:
        result = dict(state='stopped', exception_type=type(exc).__name__)
        if isinstance(exc, PreparationError):
            result.update(reason=exc.reason, schema=exc.schema)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
