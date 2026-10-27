"""Raw Mem0 plus bounded complete candidate-first-image coverage."""
from .coverage_selection import CoverageSelector, MAX_CALLS
from .mem0_options import Mem0OptionsSystem

VERSION = 'tracemem-mm-mem0-coverage-v4.4.0'


class Mem0CoverageSystem(Mem0OptionsSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.selector = CoverageSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(),version=VERSION,
            retrieval='native-raw+BM25+exact-image+options-coverage-selection+same-add-context',
            selection_max_calls_per_search=MAX_CALLS,
            selection_max_selected=32,selection_max_selected_per_batch=8,
            selection_first_image_coverage='all-candidate-first-images',
            selection_batch_merge='best-local-rank-original-order-ties',
            selection_text_candidates_repeated=True,
            selection_max_images=8,selection_max_request_bytes=3*1024*1024)
