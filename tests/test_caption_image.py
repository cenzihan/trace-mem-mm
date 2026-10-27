import base64
import copy
import io
import unittest

from PIL import Image
from agent_memory.caption_image import caption_part
from agent_memory.protocol import ContractError,image_bytes


def part(image,fmt='PNG',**kwargs):
    buffer=io.BytesIO();image.save(buffer,format=fmt,**kwargs)
    mime={'PNG':'png','JPEG':'jpeg','WEBP':'webp'}[fmt]
    return {'type':'image_url','image_url':{'url':'data:image/'+mime+';base64,'+base64.b64encode(buffer.getvalue()).decode()}}


class CaptionImageTests(unittest.TestCase):
    def test_large_original_not_mutated_and_small_wire_view(self):
        original=part(Image.effect_noise((2200,1600),100).convert('RGB'))
        before=copy.deepcopy(original)
        result=caption_part(original)
        self.assertEqual(original,before)
        raw,mime=image_bytes(result['image_url']['url'])
        self.assertEqual(mime,'jpeg');self.assertLess(len(raw),256*1024)
        self.assertLess(len(raw),len(image_bytes(original['image_url']['url'])[0])//10)
        with Image.open(io.BytesIO(raw)) as view:
            self.assertEqual(view.size,(512,372))
            self.assertNotIn('exif',view.info)

    def test_exif_orientation_is_applied_and_metadata_removed(self):
        image=Image.new('RGB',(30,10),'red');exif=Image.Exif();exif[274]=6
        raw,_=image_bytes(caption_part(part(image,'JPEG',exif=exif))['image_url']['url'])
        with Image.open(io.BytesIO(raw)) as view:
            self.assertEqual(view.size,(10,30));self.assertFalse(view.getexif())

    def test_transparency_becomes_white_without_enlarging(self):
        raw,_=image_bytes(caption_part(part(Image.new('RGBA',(20,10),(0,0,0,0))))['image_url']['url'])
        with Image.open(io.BytesIO(raw)) as view:
            self.assertEqual(view.size,(20,10));self.assertGreater(min(view.getpixel((0,0))),250)

    def test_decoded_pixel_guard_and_corrupt_payload(self):
        import unittest.mock
        source=Image.new('RGB',(2,2))
        with unittest.mock.patch('agent_memory.caption_image.MAX_DECODE_PIXELS',3):
            with self.assertRaises(ContractError):caption_part(part(source))
        broken={'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\ninvalid').decode()}}
        with self.assertRaises(ContractError):caption_part(broken)

    def test_animated_webp_is_not_silently_flattened(self):
        first=Image.new('RGB',(10,10),'red');second=Image.new('RGB',(10,10),'blue')
        original=part(first,'WEBP',save_all=True,append_images=[second],duration=100,loop=0)
        with self.assertRaises(ContractError):caption_part(original)
