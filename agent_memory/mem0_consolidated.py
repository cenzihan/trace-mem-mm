"""Global survivor comparison after bounded image-coverage selection."""
from .consolidated_selection import ConsolidatingSelector, MAX_TOTAL_CALLS
from .mem0_coverage import Mem0CoverageSystem

VERSION = 'tracemem-mm-mem0-coverage-v4.4.1'


class Mem0ConsolidatedSystem(Mem0CoverageSystem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.selector = ConsolidatingSelector(self.transport)

    def capabilities(self):
        return dict(super().capabilities(),version=VERSION,
            selection_max_calls_per_search=MAX_TOTAL_CALLS,
            selection_max_selected=8,
            selection_batch_merge='bounded-global-survivor-comparison',
            selection_final_image_policy='round-robin-local-rank-max8-including-query')

