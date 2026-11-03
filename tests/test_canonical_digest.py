import hashlib
import tracemalloc
import unittest
from agent_memory.protocol import canonical, canonical_digest

class CanonicalDigestTests(unittest.TestCase):
    def test_existing_request_identity_is_unchanged(self):
        fixtures = [None, [], {}, {'z':'汉字😀\\"\n','a':[0,-0.0,1e-12,True,None]},
                    {'messages':[{'content':[{'text':'emoji😀'}, {'url':'a'*102400}]}]}]
        for value in fixtures:
            self.assertEqual(canonical_digest(value), hashlib.sha256(canonical(value).encode()).hexdigest())
        for invalid in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):canonical_digest({'x':invalid})

    def test_media_heavy_multilingual_hash_does_not_join_request(self):
        # Distinct strings ensure a realistic per-part bound, not shared mock data.
        payload = {'text':'😀', 'images':['a'*(2*1024*1024)+str(i) for i in range(8)]}
        expected = hashlib.sha256(canonical(payload).encode()).hexdigest()
        tracemalloc.start()
        try:
            actual = canonical_digest(payload)
            peak = tracemalloc.get_traced_memory()[1]
        finally:tracemalloc.stop()
        self.assertEqual(actual, expected)
        self.assertLess(peak, 10*1024*1024)
