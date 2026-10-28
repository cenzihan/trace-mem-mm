"""Original Search query plus all supplied choices as untrusted retrieval context."""
from .joint_selection import retrieval_text
from .mem0_joint import Mem0JointSystem
from .protocol import validate_search

VERSION = 'tracemem-mm-mem0-options-v4.3.3'


class Mem0OptionsSystem(Mem0JointSystem):
    def capabilities(self):
        return dict(super().capabilities(), version=VERSION,
                    choice_context='all-options-untrusted-no-answer-key',
                    choice_max_count=32, choice_total_utf8_bytes=3500,
                    retrieval_context_utf8_bytes=8192,
                    retrieval='native-raw+BM25+exact-image+options-joint-selection+same-add-context')

    def search(self, payload):
        parts = validate_search(payload)
        self._retrieval_text(payload, parts)  # All capacity checks precede any model dispatch.
        return super().search(payload)

    def _retrieval_text(self, payload, parts):
        return retrieval_text(parts, payload.get('options', []))

    def _select_sources(self, payload, parts, sources):
        return self.selector.select(parts, sources, payload.get('options', []))
