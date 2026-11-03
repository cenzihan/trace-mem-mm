"""Offline, bounded migration of our own trusted Qdrant SQLite persistence.

Run only after the old writer is stopped, against a preserved backup. Never
accept uploaded/untrusted pickle databases. No provider calls or raw logging.
The source is read-only; a partial destination is never activated or overwritten.
"""
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3

from qdrant_client import models
from .disk_vectors import DiskClient


def migrate(source_root, destination):
    source_root=Path(source_root);destination=Path(destination)
    if destination.exists():raise ValueError('migration destination must be new')
    meta=json.loads((source_root/'meta.json').read_text())
    client=DiskClient(destination)
    result={'collections':{},'source_modified':False,'model_calls':0}
    with client.connect() as target:
        for name,config in meta['collections'].items():
            cfg=models.CreateCollection.model_validate(config)
            vectors=cfg.vectors
            if isinstance(vectors,dict):
                if set(vectors)!={''}:raise ValueError('unexpected named vectors')
                vectors=vectors['']
            if vectors.distance!=models.Distance.COSINE or vectors.multivector_config:
                raise ValueError('unsupported source vector configuration')
            target.execute('INSERT INTO collections VALUES(?,?)',(name,vectors.size))
            src=source_root/'collection'/name/'storage.sqlite'
            if not src.exists():raise ValueError('missing source collection')
            original=sqlite3.connect(src.resolve().as_uri()+'?mode=ro',uri=True)
            try:
                original.execute('PRAGMA cache_size=-2048')
                # Same newest-wins duplicate-ID policy as pinned Qdrant load().
                count=0
                for raw, in original.execute('SELECT point FROM points ORDER BY rowid DESC'):
                    point=pickle.loads(raw)  # trusted local persistence only
                    if not isinstance(point,models.PointStruct):raise ValueError('invalid persisted point')
                    payload=point.payload or {}
                    key=client.key(point.id);encoded=client.encoded(point.vector,vectors.size)
                    data=json.dumps(payload,ensure_ascii=False,allow_nan=False)
                    count+=target.execute('INSERT OR IGNORE INTO vectors VALUES(?,?,?,?,?)',
                        (name,key,encoded,data,payload.get('user_id'))).rowcount
                # A second streaming pass validates every newest source record
                # against the destination without materializing a corpus ID set.
                target.execute('CREATE TEMP TABLE checked(id TEXT PRIMARY KEY)')
                digest=hashlib.sha256();verified=0
                for raw, in original.execute('SELECT point FROM points ORDER BY rowid DESC'):
                    point=pickle.loads(raw);key=client.key(point.id)
                    if target.execute('INSERT OR IGNORE INTO checked VALUES(?)',(key,)).rowcount==0:continue
                    got=target.execute('SELECT vector,payload FROM vectors WHERE collection=? AND id=?',(name,key)).fetchone()
                    if got is None or got[0]!=client.encoded(point.vector,vectors.size) or json.loads(got[1])!=(point.payload or {}):
                        raise ValueError('migration verification mismatch')
                    digest.update(key.encode());digest.update(got[0]);digest.update(got[1].encode());verified+=1
                target.execute('DROP TABLE checked')
                if verified!=count:raise ValueError('migration count mismatch')
                result['collections'][name]={'count':count,'verified':verified,'logical_sha256':digest.hexdigest()}
            finally:original.close()
        target.execute('INSERT INTO migration VALUES(?,?)',('verified',json.dumps(result)))
    with client.connect() as target:target.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trusted_source');parser.add_argument('new_destination')
    args=parser.parse_args()
    print(json.dumps(migrate(args.trusted_source,args.new_destination)))
