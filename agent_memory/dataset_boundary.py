"""Local dataset provenance boundary; never downloads or turns gold into memories."""
import hashlib
from pathlib import Path
from urllib.parse import urlsplit


class DatasetBoundaryError(ValueError):pass


def validate_manifest(manifest):
    if not isinstance(manifest,dict):raise DatasetBoundaryError('manifest must be an object')
    required={'dataset','native_task','source_url','split','license','review_status','files',
              'intended_use','evaluation_material','source_overlap_review'}
    if not required<=manifest.keys():raise DatasetBoundaryError('incomplete provenance manifest')
    if any(not isinstance(manifest[k],str) or not manifest[k].strip()
           for k in ('dataset','native_task','source_url','split','license','review_status','intended_use','source_overlap_review')):
        raise DatasetBoundaryError('explicit provenance strings required')
    url=urlsplit(manifest['source_url'])
    if url.scheme!='https' or not url.hostname or url.username or url.password:
        raise DatasetBoundaryError('HTTPS provenance source required')
    if type(manifest['evaluation_material']) is not bool:raise DatasetBoundaryError('explicit evaluation material flag required')
    if manifest['intended_use']!='native_offline_evaluation':
        raise DatasetBoundaryError('this adapter does not authorize training/tuning or AML service ingestion')
    if not isinstance(manifest['files'],list):raise DatasetBoundaryError('explicit local file inventory required')
    paths=set()
    for item in manifest['files']:
        if not isinstance(item,dict) or set(item)!={'path','role','sha256'}:
            raise DatasetBoundaryError('file inventory requires path/role/sha256 only')
        path=Path(item['path'])
        if path.is_absolute() or '..' in path.parts or item['path'] in paths:
            raise DatasetBoundaryError('ambiguous or escaping dataset path')
        if item['role'] not in ('evidence','questions','gold'):
            raise DatasetBoundaryError('native dataset roles must be explicit')
        if not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise DatasetBoundaryError('verified lowercase SHA256 required')
        paths.add(item['path'])
    return manifest


def read_verified_evidence(root,manifest,relative_path):
    """Native adapter gets evidence only; Q/A remain solely in a separate harness.

    A reviewed native-use license is not a claim of AML-approved development data.
    No automatic conversion into dialogues, captions, or Add payloads is provided.
    """
    validate_manifest(manifest)
    if manifest['review_status']!='native_use_reviewed' or manifest['license']=='unverified':
        raise DatasetBoundaryError('license/native use review pending')
    if manifest['source_overlap_review']!='completed':
        raise DatasetBoundaryError('prior-project/source benchmark overlap review pending')
    if manifest['evaluation_material']:
        raise DatasetBoundaryError('protected evaluation material may not enter this reader')
    item=next((item for item in manifest['files'] if item['path']==relative_path),None)
    if item is None or item['role']!='evidence':
        raise DatasetBoundaryError('questions/gold are harness-only and cannot enter evidence reader')
    base=Path(root).resolve();path=(base/relative_path).resolve()
    if not path.is_relative_to(base):raise DatasetBoundaryError('resolved path escapes dataset root')
    with path.open('rb') as source:
        value=source.read(30*1024*1024+1)
    if len(value)>30*1024*1024:raise DatasetBoundaryError('local evidence admission cap exceeded')
    if hashlib.sha256(value).hexdigest()!=item['sha256']:raise DatasetBoundaryError('dataset file hash mismatch')
    return value
