"""Raw Mem0 plus explicitly bounded hosted visual reranking; no answer model."""
import json

from .mem0_raw import Mem0RawSystem
from .protocol import content_parts,validate_add,validate_search
from .caption_image import caption_part
from .visual_rerank import query_views,MAX_SOURCES

VERSION='tracemem-mm-mem0-visual-v4.1.0'


class Mem0VisualSystem(Mem0RawSystem):
    def __init__(self,*args,scoring_adapter,**kwargs):
        self.visual_scorer=scoring_adapter
        super().__init__(*args,**kwargs)

    def ready(self):
        super().ready()
        self.visual_scorer.ready()

    def capabilities(self):
        return dict(super().capabilities(),version=VERSION,
            models=['text-embedding-v4','qwen3-vl-rerank'],
            vision='hosted-visual-reranking-of-bounded-JPEG-views',
            retrieval='native-raw+BM25+exact-image+visual-rerank+same-add-context',
            visual_source_pool=40,visual_query_images=1,visual_images_per_source=1,
            visual_text_prefix_bytes=6000,visual_max_calls_per_search=24,
            visual_preview='JPEG512-q85-max256KiB',original_image_mutated=False,
            visual_scoring_is_answerability=False,calibrated_abstention=False,
            reranker_weight_identity_verified=False)

    def add(self,payload):
        validate_add(payload)
        # Reject undecodable image inputs before native Add or any provider call.
        for message in payload['messages']:
            for part in content_parts(message['content']):
                if part['type']=='image_url':caption_part(part)
        return super().add(payload)

    def search(self,payload):
        query_views(validate_search(payload))
        return super().search(payload)

    def _order_sources(self,payload,parts,rows,order,identities):
        uid=payload['user_id']
        with self.connect() as db:
            image_ids={r[0] for r in db.execute("""SELECT DISTINCT e.memory_id
                FROM evidence e JOIN memories m ON m.id=e.memory_id WHERE m.user_id=?""",(uid,))}
            # Reserve half the bounded pool for image sources. Existing rank wins;
            # otherwise stable source order is used, never benchmark/question IDs.
            image_order=[mid for mid in order if mid in image_ids]
            image_order.extend(row[0] for row in rows if row[0] in image_ids and row[0] not in image_order)
            pool=list(dict.fromkeys([*order[:20],*image_order[:20]]))
            pool.extend(mid for mid in order if mid not in pool)
            pool=pool[:MAX_SOURCES]
            if not pool:return order
            sources=[]
            for mid in pool:
                value=db.execute('SELECT content FROM memories WHERE user_id=? AND id=?',(uid,mid)).fetchone()
                if value is None:raise ValueError('candidate original disappeared')
                sources.append({'id':mid,'content':json.loads(value[0])})
        scores=self.visual_scorer.score(parts,sources)
        ranked_sources=[pool[i] for i in sorted(range(len(pool)),key=lambda i:(-scores[i],i))]
        return list(dict.fromkeys([*sorted(identities),*ranked_sources,*order]))

