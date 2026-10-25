"""Actual byte-boundary checks, not model/competition evaluation."""
import base64
import binascii
import struct
import tempfile
import unittest
from pathlib import Path

from agent_memory.protocol import IMAGE_LIMIT,MEDIA_LIMIT,ContractError,validate_add
from agent_memory.system import AgentMemorySystem


def padded_png(size):
    # Preserve a valid 8x8 PNG, adding a valid private ancillary chunk before IEND.
    root=Path(__file__).resolve().parent.parent
    source=(root/'fixtures/red.png').read_bytes()
    body=b'\x00'*(size-len(source)-12)
    kind=b'paDd'
    chunk=struct.pack('>I',len(body))+kind+body+struct.pack('>I',binascii.crc32(kind+body)&0xffffffff)
    value=source[:-12]+chunk+source[-12:]
    assert len(value)==size
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(value).decode()}}


class MediaBoundaryTests(unittest.TestCase):
    def payload(self,parts):return dict(request_id='r',user_id='u',session_id='s',messages=[dict(role='user',content=parts)])

    def test_real_single_image_10mib_boundary(self):
        validate_add(self.payload([padded_png(IMAGE_LIMIT)]))
        self.assertRaises(ContractError,validate_add,self.payload([padded_png(IMAGE_LIMIT+1)]))

    def test_real_30mib_add_and_response_boundaries(self):
        part=padded_png(IMAGE_LIMIT)
        validate_add(self.payload([part,part,part]))
        self.assertRaises(ContractError,validate_add,self.payload([part,part,part,padded_png(128)]))
        with tempfile.TemporaryDirectory() as tmp:
            store=AgentMemorySystem(Path(tmp)/'db.sqlite')
            for i in range(4):
                payload=self.payload([{'type':'text','text':'image evidence '+str(i)},part]);payload['request_id']='r'+str(i)
                store.add(payload)
            self.assertEqual(len(store.search(dict(query='image evidence',user_id='u',top_k=3))['data']),3)
            result,trace=store.retrieve(dict(query='image evidence',user_id='u',top_k=4))
            self.assertEqual(len(result['data']),3)
            self.assertEqual(trace['media_budget_omitted_records'],1)
            for memory in result['data']:
                self.assertEqual(memory['content'][1],part)
