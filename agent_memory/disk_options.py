"""Disk-backed storage profile; model selection and request limits unchanged."""
from .two_lane_options import TwoLaneOptionsSystem


class DiskOptionsSystem(TwoLaneOptionsSystem):
    disk_vectors=True

    def capabilities(self):
        return dict(super().capabilities(),version='tracemem-mm-disk-v4.3.14-rc1',
                    vector_storage='sqlite-bounded-exact-cosine',vector_scan_batch=128,
                    vector_cache_kib=2048,vector_ties='score-then-id-descending')
