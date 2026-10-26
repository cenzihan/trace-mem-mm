"""Pure V7 source, identity, metadata selection and retrieval helpers.

No file/network/model/gold access. This condition uses original dialogue text
and author-provided image captions, not raw image understanding. Reader and
typed scoring are deliberately separate from these helpers.
"""
import hashlib
import json

from .native_memgallery import media_reference
from .retrieval import ranked, rrf
from .v6_rerank import pair_id, stable_rerank, validate_ranking


SEED = 20261026
VARIANTS = ('dense', 'dense_reranked', 'hybrid', 'hybrid_reranked')
REPRESENTATION = 'memgallery-original-dialogue-author-image-caption-v1'
QUERY_FIELDS = {'index', 'question', 'point', 'question_image', 'query_image', 'image_caption'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _identity(value):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError('original nonempty single-line source identity required')
    return value


def native_id(session, round_id):
    """Lossless composite identity; a round label alone is never sufficient."""
    return canonical([_identity(session), _identity(round_id)])


def source_documents(source_only):
    """Return every source round in original order, retaining every image mapping.

    Accept an already separated source-only object; never accept a QA/profile
    container. Image references are validated strings only, never opened.
    """
    if not isinstance(source_only, dict) or set(source_only) != {'multi_session_dialogues'}:
        raise ValueError('exact source-only dialogue projection required; QA/profile forbidden')
    sessions = source_only['multi_session_dialogues']
    if not isinstance(sessions, list) or not sessions:
        raise ValueError('complete nonempty native session list required')
    result, seen, session_ids = [], set(), set()
    for session in sessions:
        if not isinstance(session, dict) or set(session) - {'session_id', 'date', 'dialogues'}:
            raise ValueError('source session fields only')
        sid = _identity(session['session_id'])
        if sid in session_ids:
            raise ValueError('duplicate native session')
        session_ids.add(sid)
        date, dialogues = session['date'], session['dialogues']
        if not isinstance(date, str) or not isinstance(dialogues, list) or not dialogues:
            raise ValueError('original date string and complete dialogue list required')
        for item in dialogues:
            if (not isinstance(item, dict) or set(item) -
                    {'round', 'user', 'assistant', 'image_id', 'input_image', 'image_caption'}):
                raise ValueError('source round fields only')
            rid = _identity(item['round'])
            identity = native_id(sid, rid)
            if identity in seen:
                raise ValueError('duplicate native session/round identity')
            seen.add(identity)
            user, assistant = item['user'], item['assistant']
            if any(not isinstance(v, str) or not v.strip() for v in (user, assistant)):
                raise ValueError('complete original user and assistant text required')
            refs, captions, ids = (item.get(key, []) for key in ('input_image', 'image_caption', 'image_id'))
            if (any(not isinstance(v, list) for v in (refs, captions, ids))
                    or not len(refs) == len(captions) == len(ids)):
                raise ValueError('all image references, captions and IDs must align')
            images, image_text = [], []
            for position, (ref, caption, image_id) in enumerate(zip(refs, captions, ids)):
                if (not isinstance(caption, str) or type(image_id) not in (str, int)
                        or (isinstance(image_id, str) and not image_id.strip())):
                    raise ValueError('original image ID and caption required')
                relative = media_reference(ref)
                images.append(dict(image_id=image_id, image_ref=ref, relative_path=relative,
                                   caption=caption, ordinal=position))
                image_text.append('[image_id=' + canonical(image_id) + '; image_ref=' + canonical(ref)
                                  + ']\n' + caption)
            text = '\n'.join(('SESSION: ' + sid, 'DATE: ' + date, 'ROUND: ' + rid,
                               'USER:\n' + user, 'AUTHOR_IMAGE_CAPTIONS:\n' + '\n'.join(image_text),
                               'ASSISTANT:\n' + assistant))
            result.append(dict(id=identity, native_id=identity, text=text, ordinal=len(result),
                               source_session=sid, source_round=rid, source_date=date,
                               user=user, assistant=assistant, image_ids=list(ids), image_refs=list(refs),
                               image_captions=list(captions), images=images,
                               representation=REPRESENTATION, raw_media_loaded=False))
    return result


def source_mapping(documents):
    """Freeze round and image occurrence maps without collapsing repeated labels."""
    mapping, by_round, by_image = {}, {}, {}
    for row in documents:
        identity = native_id(row['source_session'], row['source_round'])
        if row['id'] != identity or row['native_id'] != identity or identity in mapping:
            raise ValueError('unique exact native source mapping required')
        mapping[identity] = dict(source_session=row['source_session'], source_round=row['source_round'],
                                 source_date=row['source_date'], image_ids=list(row['image_ids']))
        by_round.setdefault(row['source_round'], []).append(identity)
        for image_id in row['image_ids']:
            by_image.setdefault(canonical(image_id), []).append(identity)
    return dict(by_native_id=mapping, by_round_label=by_round, by_image_id=by_image,
                ambiguous_round_labels=sorted(key for key, values in by_round.items() if len(values) > 1))


def project_queries(rows):
    """Validate an already separated question projection; preserve image fields.

    Native point is a task category, never silently converted to an ATM qtype.
    Optional image/caption values are retained without opening or rewriting them.
    """
    if not isinstance(rows, list):
        raise ValueError('question-only list required')
    result, seen = [], set()
    for row in rows:
        if (not isinstance(row, dict) or not {'index', 'question', 'point'} <= set(row)
                or set(row) - QUERY_FIELDS):
            raise ValueError('question-only fields required; answers/clues/profile forbidden')
        index = row['index']
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('unique original nonnegative question index required')
        seen.add(index)
        _identity(row['point'])
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('unchanged original question required')
        for field in set(row) - {'index', 'question', 'point'}:
            value = row[field]
            if not (value is None or isinstance(value, str)
                    or isinstance(value, list) and all(isinstance(v, str) for v in value)):
                raise ValueError('query image fields must retain native strings or string arrays')
        result.append(json.loads(canonical(row)))
    return result


def select_pilot(metadata, *, count=16, seed=SEED):
    """One hash-selected question per point, then global hash fill; metadata only."""
    if seed != SEED or type(seed) is not int or type(count) is not int or not 1 <= count <= 16:
        raise ValueError('fixed V7 seed and pilot size within 1..16 required')
    if not isinstance(metadata, list):
        raise ValueError('metadata-only list required')
    groups, all_rows, seen = {}, [], set()
    for row in metadata:
        if not isinstance(row, dict) or set(row) != {'index', 'point'}:
            raise ValueError('selection accepts only index/point, never question or gold')
        index = row['index']
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('unique original question index required')
        seen.add(index)
        point = _identity(row['point'])
        value = dict(row, selection_hash=digest(['memgallery-v7-point-first-v1', seed, point, index]))
        groups.setdefault(point, []).append(value)
        all_rows.append(value)
    if len(all_rows) < count or len(groups) > count:
        raise ValueError('pilot must fit one item per point without inventing questions')
    key = lambda row: (row['selection_hash'], row['index'])
    representatives = sorted((min(values, key=key) for values in groups.values()), key=key)
    selected_indices = {row['index'] for row in representatives}
    fill = sorted((row for row in all_rows if row['index'] not in selected_indices), key=key)
    selected = representatives + fill[:count - len(representatives)]
    result = dict(schema='memgallery-v7-selection-v1', seed=seed, count=count,
                  selection_rule='SHA256(canonical [memgallery-v7-point-first-v1,seed,point,index]); one minimum per point, then global hash fill',
                  metadata_sha256=digest(sorted(metadata, key=lambda row: row['index'])),
                  selected=selected, variants=list(VARIANTS), planned_reader_answers=count * 4,
                  question_or_gold_used=False, point_used_for_selection=True)
    result['selection_sha256'] = digest(result)
    return result


def base_rankings(index, question, query_vector, *, namespace):
    """SharedDenseIndex math: one full dense pass plus equal native Top100 RRF60."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('explicit original query representation required')
    dense = index.dense(query_vector, namespace=namespace, top_k=100)
    flat = [identity for identity, _ in index.sparse.search(question, 100)]
    fused = rrf([flat, dense['ranked_ids']], [1.0, 1.0], 60)
    order = ranked(fused)[:100]
    hybrid = dict(ranked_ids=order, scores=[fused[identity] for identity in order],
                  source_count=dense['source_count'], method='shared_source_flat_dense_rrf60',
                  lanes=dict(flat=flat, dense=list(dense['ranked_ids'])), query_persisted=False)
    known = set(index.source_chunks)
    for row in (dense, hybrid):
        validate_ranking(row, known)
        if len(row['ranked_ids']) != min(100, len(known)):
            raise ValueError('complete native Top100 candidate set required')
    return dict(dense=dense, hybrid=hybrid)


def union_pairs(query_identity, bases):
    _identity(query_identity)
    if set(bases) != {'dense', 'hybrid'}:
        raise ValueError('both fixed base retrievers required')
    known = set(bases['dense']['ranked_ids']) | set(bases['hybrid']['ranked_ids'])
    for row in bases.values():
        validate_ranking(row, known)
    return [dict(pair_id=pair_id(query_identity, identity), query_id=query_identity, source_id=identity)
            for identity in sorted(known)]


def rerank_variants(bases, scores_by_source):
    """Use one score per union pair; each reranker retains its own original pool."""
    expected = {row['source_id'] for row in union_pairs('validation-only', bases)}
    if not isinstance(scores_by_source, dict) or set(scores_by_source) != expected:
        raise ValueError('one finite score for every union source, without extras, required')
    result = {}
    for base in ('dense', 'hybrid'):
        result[base] = json.loads(canonical(bases[base]))
        order = stable_rerank(bases[base]['ranked_ids'], scores_by_source)
        result[base + '_reranked'] = dict(ranked_ids=order,
            scores=[scores_by_source[identity] for identity in order],
            method=base + '_fixed_candidate_rerank', top100_pool_preserved=True)
    return result
