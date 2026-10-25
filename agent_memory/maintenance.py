"""Offline source-only migration, integrity and non-overwriting SQLite backups."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path

from .cards import derive
from .protocol import canonical,text_of
from .system import AgentMemorySystem


def integrity(path):
    path=Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        checks=[r[0] for r in db.execute('PRAGMA integrity_check')]
        foreign=list(db.execute('PRAGMA foreign_key_check'))
        version=db.execute('PRAGMA user_version').fetchone()[0]
        kind=db.execute('PRAGMA application_id').fetchone()[0]
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        scope_errors=0
        if {'cards','memories'}<=tables:
            scope_errors=db.execute('SELECT count(*) FROM cards c JOIN memories m ON m.id=c.memory_id WHERE c.user_id!=m.user_id OR c.session_id!=m.session_id').fetchone()[0]
        return dict(integrity_ok=checks==['ok'],foreign_key_errors=len(foreign),scope_errors=scope_errors,
                    schema_version=version,application_id=kind)


def backup_database(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('backup/restore never overwrites an existing file')
    result=integrity(source)
    if not result['integrity_ok'] or result['foreign_key_errors'] or result['scope_errors']:raise ValueError('source integrity failed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb'):pass
    src=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True);dst=sqlite3.connect(destination)
    try:src.backup(dst);dst.commit()
    finally:src.close();dst.close()
    verified=integrity(destination)
    if not verified['integrity_ok']:raise ValueError('backup integrity failed; preserve for diagnosis')
    return dict(destination=str(destination),bytes=destination.stat().st_size,integrity=verified)


def repair_missing_indexes(path,user_id=None):
    store=AgentMemorySystem(path);repaired=0
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        pending="SELECT 1 FROM add_operations WHERE state!='complete'"
        pending_args=()
        if user_id is not None:pending+=' AND user_id=?';pending_args=(user_id,)
        if db.execute(pending+' LIMIT 1',pending_args).fetchone():
            raise ValueError('unresolved model Add must be inspected before index repair')
        query='SELECT m.id,m.user_id,m.session_id,m.content FROM memories m LEFT JOIN cards c ON c.memory_id=m.id WHERE c.id IS NULL'
        params=()
        if user_id is not None:query+=' AND m.user_id=?';params=(user_id,)
        for mid,uid,sid,raw in db.execute(query,params).fetchall():
            source=text_of(json.loads(raw));card=derive(source)
            indexed=source+' '+' '.join(' '.join(f[k] for k in ('entity','attribute','value')) for f in card['facts'])
            cid='card_'+mid[4:]
            db.execute('INSERT INTO cards VALUES(?,?,?,?,?,?,NULL,NULL)',(cid,mid,uid,sid,canonical(card),indexed))
            for i,f in enumerate(card['facts']):
                datetime.fromisoformat(f['event'].replace('Z','+00:00'))
                db.execute('INSERT INTO facts VALUES(?,?,?,?,?,?,?,?)',(cid+':'+str(i),cid,uid,f['entity'].casefold(),f['attribute'].casefold(),f['value'],f['event'],datetime.now(timezone.utc).isoformat()))
            for link in card['links']:db.execute('INSERT INTO entity_links VALUES(?,?,?,?)',(cid,uid,link['entity'].casefold(),link['related'].casefold()))
            repaired+=1
    return dict(repaired=repaired,model_calls=0,source_changed=False,integrity=integrity(path))
