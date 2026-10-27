"""Bounded visual derivatives for low-detail captions; originals stay untouched."""
import base64
import io

from .protocol import ContractError, image_bytes

MAX_DECODE_PIXELS = 24000000
PREVIEW_EDGE = 512
PREVIEW_BYTES = 256 * 1024


def caption_part(part):
    """Return a separate JPEG view, without editing the archived API source."""
    raw, _ = image_bytes(part['image_url']['url'])
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP') or getattr(source, 'is_animated', False):
                raise ContractError('static supported image required')
            if source.width * source.height > MAX_DECODE_PIXELS:
                raise ContractError('local decoded image capacity exceeded', 413)
            source.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
            view = ImageOps.exif_transpose(source).convert('RGBA')
            background = Image.new('RGBA', view.size, 'white')
            background.alpha_composite(view)
            rgb = background.convert('RGB')
            buffer = io.BytesIO()
            rgb.save(buffer, format='JPEG', quality=85, subsampling=2, optimize=False)
            encoded = buffer.getvalue()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ContractError('invalid or undecodable image') from None
    if len(encoded) > PREVIEW_BYTES:
        raise ContractError('local caption preview byte capacity exceeded', 413)
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')}}
