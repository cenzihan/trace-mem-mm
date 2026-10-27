"""Mem0 raw Add and one joint multimodal evidence-selection call per Search."""
import json

from .caption_image import caption_part
from .joint_selection import JointSelector, query_admission, MAX_SOURCES
from .mem0_raw import Mem0RawSystem
from .protocol import content_parts, validate_add, validate_search

VERSION = 'tracemem-mm-mem0-joint-v4.2.0'


class Mem0JointSystem(Mem0RawSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = JointSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
            models=['text-embedding-v4', 'gpt-4o-mini'], llm_calls_enabled=True,
            vision='joint-selection-of-bounded-JPEG-views',
            retrieval='native-raw+BM25+exact-image+joint-selection+same-add-context',
            selection_max_calls_per_search=1, selection_max_sources=32,
            selection_max_selected=8, selection_max_images=8,
            selection_text_prefix_bytes=1500, selection_query_text_bytes=4000,
            selection_preview='JPEG512-q85-max256KiB', hosted_rerank_calls=0,
            generated_answers=False, calibrated_abstention=False,
            observed_llm_model='gpt-4o-mini-2024-07-18')

    def add(self, payload):
        validate_add(payload)
        for message in payload['messages']:
            for part in content_parts(message['content']):
                if part['type'] == 'image_url':caption_part(part)
        return super().add(payload)

    def search(self, payload):
        query_admission(validate_search(payload))
        return super().search(payload)

    def _select_sources(self, payload, parts, sources):
        return self.selector.select(parts, sources)

    def _order_sources(self, payload, parts, rows, order, identities):
        uid = payload['user_id']
        with self.connect() as db:
            image_ids = {r[0] for r in db.execute('''SELECT DISTINCT e.memory_id
                FROM evidence e JOIN memories m ON m.id=e.memory_id WHERE m.user_id=?''', (uid,))}
            image_order = [mid for mid in order if mid in image_ids]
            image_order.extend(r[0] for r in rows if r[0] in image_ids and r[0] not in image_order)
            pool = list(dict.fromkeys([*order[:16], *image_order[:16]]))
            pool.extend(mid for mid in order if mid not in pool)
            pool = pool[:MAX_SOURCES]
            if not pool:return sorted(identities)
            groups = {session: i for i, session in enumerate(sorted({r[2] for r in rows}))}
            sources = []
            for mid in pool:
                value = db.execute('''SELECT content,created_at,session_id FROM memories
                    WHERE user_id=? AND id=?''', (uid, mid)).fetchone()
                if value is None:raise ValueError('candidate original disappeared')
                sources.append(dict(content=json.loads(value[0]), created_at=value[1],
                                    session_group=groups[value[2]]))
        selected = self._select_sources(payload, parts, sources)
        # No unselected tail: [] is a valid evidence selection, never a failure fallback.
        # Exact image identities remain direct evidence independent of model selection.
        return list(dict.fromkeys([*sorted(identities), *(pool[i] for i in selected)]))
