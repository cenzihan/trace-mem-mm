import base64
import io
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from PIL import Image
from agent_memory import protocol, request_media, caption_image


def uri(color='red'):
    b=io.BytesIO()
    with Image.new('RGB',(32,32),color) as image:image.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


class RequestMediaTests(unittest.TestCase):
    def test_bytes_and_preview_identical_and_returned_parts_are_independent(self):
        u=uri();part={'type':'image_url','image_url':{'url':u}}
        raw=protocol.image_bytes(u);preview=caption_image.caption_part(part)
        with patch.object(protocol,'_decode_image_bytes',wraps=protocol._decode_image_bytes) as decode:
            with patch.object(caption_image,'_caption_uri',wraps=caption_image._caption_uri) as render:
                with request_media.media_request_cache() as state:
                    for _ in range(6):
                        self.assertEqual(protocol.image_bytes(u),raw)
                        value=caption_image.caption_part(part);self.assertEqual(value,preview)
                        value['image_url']['url']='mutated consumer copy'
                    self.assertEqual(decode.call_count,1);self.assertEqual(render.call_count,1)
                self.assertEqual(state,{'values':{},'bytes':0})
            protocol.image_bytes(u);self.assertEqual(decode.call_count,2)

    def test_invalid_and_mutated_input_cannot_reuse_valid_result(self):
        good=uri();part={'type':'image_url','image_url':{'url':good}}
        with request_media.media_request_cache():
            protocol.content_parts([part]);part['image_url']['url']='data:image/png;base64,AAAA'
            with self.assertRaises(protocol.ContractError):protocol.content_parts([part])
            with self.assertRaises(protocol.ContractError):protocol.image_bytes([])
            with self.assertRaises(protocol.ContractError):protocol.image_bytes('data:image/jpeg;base64,'+good.split(',')[1])

    def test_duplicate_images_still_count_toward_request_limit(self):
        raw=b'\x89PNG\r\n\x1a\n'+b'x'*(8*1024*1024)
        u='data:image/png;base64,'+base64.b64encode(raw).decode()
        p=dict(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=u)) for _ in range(4)])])
        with request_media.media_request_cache():
            with self.assertRaisesRegex(protocol.ContractError,'30 MiB'):protocol.validate_add(p)

    def test_budget_and_entry_limit_bound_retained_results(self):
        with patch.object(request_media,'MAX_BYTES',10),patch.object(request_media,'MAX_ENTRIES',2):
            with request_media.media_request_cache() as state:
                for i in range(100):request_media.media_result('decoded',str(i),lambda:(b'abc','png'))
                self.assertEqual(len(state['values']),2);self.assertEqual(state['bytes'],6)
                self.assertEqual(request_media.media_result('decoded','large',lambda:(b'x'*100,'png'))[0],b'x'*100)
                self.assertEqual(state['bytes'],6)

    def test_concurrent_nested_and_failed_scopes_release_without_cross_request_reuse(self):
        barrier=threading.Barrier(2)
        def call(color):
            with self.assertRaises(RuntimeError):
                with request_media.media_request_cache() as outer:
                    u=uri(color);protocol.image_bytes(u);barrier.wait(3)
                    with request_media.media_request_cache() as inner:
                        self.assertEqual(inner['values'],{});protocol.image_bytes(u)
                    self.assertEqual(inner['values'],{});self.assertEqual(len(outer['values']),1)
                    raise RuntimeError('synthetic failure')
            self.assertEqual(outer['values'],{});self.assertIsNone(request_media._current.get())
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(call,['red','blue']))

    def test_corrupt_pixels_still_fail_caption_validation_after_signature_cache(self):
        u='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\ncorrupt').decode()
        with request_media.media_request_cache():
            protocol.image_bytes(u)
            for _ in range(2):
                with self.assertRaises(protocol.ContractError):caption_image.caption_part(dict(image_url=dict(url=u)))
