"""Source-only three-lane sparse retrieval with native-parent rank fusion.

This research candidate does not change the frozen N0 formatter or production
API. It stores complete source text, never query/gold data, and performs no I/O,
model calls, or embedding. Detail token windows are internal indexing units;
returned evidence IDs always identify original sources.
"""
import heapq
import math
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from .atm_n0 import native_sgm_corpus
from .retrieval import tokenize


LANES = ('flat', 'summary', 'detail')
DETAIL_WINDOW = 512
DETAIL_STRIDE = 384
LANE_LIMIT = 100
RRF_CONSTANT = 60
FORBIDDEN = {
    'question', 'questions', 'query', 'queries', 'answer', 'answers', 'gold',
    'qas', 'evidence_ids', 'niah_evidence_ids', 'clue',
}


def _namespace(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('explicit nonempty source namespace required')
    return value


def _source_only(value):
    """Reject evaluation-field containers, including ignored nested metadata."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key.casefold() in FORBIDDEN for key in value):
            raise ValueError('source-only fields required; query/QA/gold forbidden')
        for child in value.values():
            _source_only(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _source_only(child)


def _text(row, field):
    value = row.get(field, '')
    if value is None:
        return ''
    if not isinstance(value, str):
        raise ValueError('source text field must be a string')
    return value


def build_sgm_views(images, videos, emails, *, expected_images,
                    expected_videos, expected_emails, namespace):
    """Validate the complete native corpus and derive three source-only views.

    N0 flat text is unchanged byte for byte. Summary/detail fields contain only
    selected source strings, without injected IDs or field labels. The detail
    text is kept whole here; indexing later applies 512-token windows with a
    384-token stride to the exact output of the frozen tokenizer.
    """
    namespace = _namespace(namespace)
    _source_only((images, videos, emails))
    for rows in (images, videos, emails):
        if not isinstance(rows, list):
            raise ValueError('native source lists required')
        if any(isinstance(row, dict) and 'namespace' in row and row['namespace'] != namespace
               for row in rows):
            raise ValueError('native input namespace mismatch')
    if type(expected_emails) is not int or expected_emails < 0:
        raise ValueError('nonnegative full email inventory count required')
    flat_records, metadata = native_sgm_corpus(
        images, videos, emails, expected_images=expected_images,
        expected_videos=expected_videos, expected_emails=expected_emails)
    for record in flat_records:
        record.update(namespace=namespace, native_id=record['id'])

    views = [dict(namespace=namespace, native_id=row['id'], lane='flat', text=row['text'])
             for row in flat_records]
    for modality, rows, path_field in (
            ('image', images, 'image_path'), ('video', videos, 'video_path'),
            ('email', emails, 'id')):
        for row in rows:
            identity = _text(row, path_field)
            if modality != 'email':
                identity = PurePosixPath(identity.replace('\\', '/')).stem
                short = _text(row, 'short_caption')
                summary = short if short.strip() else _text(row, 'caption')
                tags = row.get('tags', []) or []  # Already validated by native_sgm_corpus.
                fields = (_text(row, 'caption'), _text(row, 'ocr_text'),
                          ', '.join(tags), _text(row, 'timestamp'),
                          _text(row, 'location_name'))
            else:
                summary = _text(row, 'short_summary')
                fields = (_text(row, 'detail'), _text(row, 'timestamp'))
            detail = '\n'.join(field for field in fields if field.strip())
            for lane, text in (('summary', summary), ('detail', detail)):
                if text.strip():
                    views.append(dict(namespace=namespace, native_id=identity,
                                      lane=lane, text=text))
    views.sort(key=lambda row: (row['native_id'], LANES.index(row['lane']), row['text']))
    view_counts = Counter(v['lane'] for v in views)
    metadata = dict(metadata, namespace=namespace, condition='external_sgm_research',
                    source_count=len(flat_records), view_counts={lane: view_counts[lane] for lane in LANES},
                    detail_window_tokens=DETAIL_WINDOW, detail_stride_tokens=DETAIL_STRIDE,
                    token_windowing='exact tokenize output; no re-tokenization',
                    native_parent_scoring='max unit score per lane, then equal RRF60',
                    lane_parent_limit=LANE_LIMIT, source_text_preserved=True,
                    academic_strict_end_to_end=False, query_or_gold_indexed=False)
    return flat_records, views, metadata


def _detail_windows(terms):
    if not terms:
        yield []
        return
    for start in range(0, len(terms), DETAIL_STRIDE):
        yield terms[start:start + DETAIL_WINDOW]
        if start + DETAIL_WINDOW >= len(terms):
            break


class _LaneBM25:
    """BM25 over all indexing units, with max parent collapse before top-K."""
    def __init__(self, units):
        self.parents = []
        self.lengths = []
        self.postings = defaultdict(list)
        for parent, terms in units:
            position = len(self.parents)
            self.parents.append(parent)
            counts = Counter(terms)
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((position, frequency))
        self.count = len(self.parents)
        self.average = sum(self.lengths) / max(1, self.count) or 1

    def search(self, query_terms):
        unit_scores = defaultdict(float)
        for term in sorted(set(query_terms)):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.count - df + .5) / (df + .5))
            for position, frequency in postings:
                length = self.lengths[position]
                unit_scores[position] += idf * frequency * 2.2 / (
                    frequency + 1.2 * (.25 + .75 * length / self.average))
        parent_scores = {}
        for position, score in unit_scores.items():
            parent = self.parents[position]
            parent_scores[parent] = max(parent_scores.get(parent, 0.0), score)
        order = heapq.nsmallest(LANE_LIMIT, parent_scores,
                               key=lambda parent: (-parent_scores[parent], parent))
        return dict(ranked_ids=order, scores={parent: parent_scores[parent] for parent in order})


class ParentSparseIndex:
    """Namespace-bound three-lane BM25/RRF with no query persistence.

    Every record must have exactly one flat view matching its full N0 text.
    Repeated (native parent, lane, text) views and repeated token windows within
    a native parent/lane are deduplicated before BM25 statistics. All matching
    chunks are scored before collapsing to native parents; chunk-heavy sources
    cannot occupy many slots.
    """
    def __init__(self, records, views, namespace):
        self.namespace = _namespace(namespace)
        if not isinstance(records, (list, tuple)) or not isinstance(views, (list, tuple)):
            raise ValueError('explicit source records and views required')
        _source_only((records, views))
        self.records = {}
        for row in records:
            if not isinstance(row, dict) or row.get('namespace') != self.namespace:
                raise ValueError('source record namespace mismatch')
            if set(row) - {'id', 'native_id', 'namespace', 'modality', 'text',
                           'timestamp', 'description_present'}:
                raise ValueError('undeclared source record field')
            parent = row.get('native_id')
            if not isinstance(parent, str) or not parent or not isinstance(row.get('text'), str):
                raise ValueError('native source identity and text required')
            if row.get('id', parent) != parent:
                raise ValueError('native identity aliases disagree')
            if parent in self.records:
                raise ValueError('duplicate native source; no overwrite')
            self.records[parent] = dict(row)

        unique = set()
        flat_parents = set()
        for view in views:
            if not isinstance(view, dict) or set(view) != {'namespace', 'native_id', 'lane', 'text'}:
                raise ValueError('only source namespace/native_id/lane/text view fields allowed')
            if view['namespace'] != self.namespace:
                raise ValueError('source view namespace mismatch')
            parent, lane, text = view['native_id'], view['lane'], view['text']
            if not isinstance(parent, str) or parent not in self.records:
                raise ValueError('view has no native source parent')
            if lane not in LANES or not isinstance(text, str):
                raise ValueError('known lane and source text required')
            if lane == 'flat':
                if text != self.records[parent]['text']:
                    raise ValueError('flat text must match the full native source')
                flat_parents.add(parent)
            unique.add((parent, lane, text))
        if flat_parents != set(self.records):
            raise ValueError('every native source must retain its flat view')

        lane_units = {lane: [] for lane in LANES}
        seen_units = set()
        self.view_counts = {lane: 0 for lane in LANES}
        for parent, lane, text in sorted(unique):
            if lane != 'flat' and not text.strip():
                continue
            self.view_counts[lane] += 1
            terms = tokenize(text)
            chunks = _detail_windows(terms) if lane == 'detail' else (terms,)
            for chunk in chunks:
                tokens = tuple(chunk)
                unit_key = (parent, lane, tokens)
                if unit_key not in seen_units:
                    seen_units.add(unit_key)
                    lane_units[lane].append((parent, tokens))
        self._lanes = {lane: _LaneBM25(lane_units[lane]) for lane in LANES}
        self.source_count = len(self.records)
        self.unit_counts = {lane: self._lanes[lane].count for lane in LANES}

    def search(self, query, top_k=100):
        if not isinstance(query, str) or type(top_k) is not int or not 1 <= top_k <= LANE_LIMIT:
            raise ValueError('string query and top_k within 1..100 required')
        terms = tokenize(query)
        lanes = {lane: self._lanes[lane].search(terms) for lane in LANES}
        fused = defaultdict(float)
        for lane in LANES:
            for rank, parent in enumerate(lanes[lane]['ranked_ids'], 1):
                fused[parent] += 1.0 / (RRF_CONSTANT + rank)
        order = heapq.nsmallest(top_k, fused, key=lambda parent: (-fused[parent], parent))
        return dict(ranked_ids=order, scores={parent: fused[parent] for parent in order}, lanes=lanes)
