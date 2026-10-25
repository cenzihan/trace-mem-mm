"""Durable bounded development runs; counters contain no prompts or credentials."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from .budget import micro_cny


class DevelopmentRun:
    def __init__(self,path,*,create=False,max_requests=32,max_reservation_cny='10'):
        self.path=Path(path)
        if create:
            if not 1<=max_requests<=50 or not 0<micro_cny(max_reservation_cny)<=10000000:
                raise ValueError('development run exceeds approved bounds')
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('xb'):pass
            with self.connect() as db:
                db.executescript('''CREATE TABLE run (id INTEGER PRIMARY KEY, max_requests INTEGER, cap INTEGER, state TEXT);
                CREATE TABLE calls (ordinal INTEGER PRIMARY KEY, model TEXT, held INTEGER, state TEXT,
                                    started_at TEXT, finished_at TEXT, returned_model TEXT);''')
                db.execute('INSERT INTO run VALUES(1,?,?,?)',(max_requests,micro_cny(max_reservation_cny),'active'))
        elif not self.path.exists():raise ValueError('development run missing')

    def connect(self):
        # sqlite context commits/rolls back; callers explicitly close below.
        return _Connection(self.path)

    def claim(self,model,reservation_cny):
        amount=micro_cny(reservation_cny)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            limit,cap,state=db.execute('SELECT max_requests,cap,state FROM run').fetchone()
            count,total=db.execute('SELECT COUNT(*),coalesce(SUM(held),0) FROM calls').fetchone()
            if state!='active' or count>=limit or total+amount>cap:
                raise ValueError('durable development request/budget limit reached')
            if db.execute("SELECT 1 FROM calls WHERE state<>'complete' LIMIT 1").fetchone():
                raise ValueError('unfinished development attempt blocks automatic continuation')
            ordinal=count+1
            db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?)',
                       (ordinal,model,amount,'claimed',datetime.now(timezone.utc).isoformat(),None,None))
            return ordinal

    def finish(self,ordinal,model):
        with self.connect() as db:
            result=db.execute("UPDATE calls SET state='complete',finished_at=?,returned_model=? WHERE ordinal=? AND state='claimed'",
                              (datetime.now(timezone.utc).isoformat(),model,ordinal))
            if result.rowcount!=1:raise ValueError('development attempt already finalized')

    def close(self):
        with self.connect() as db:
            if db.execute("SELECT 1 FROM calls WHERE state<>'complete'").fetchone():
                raise ValueError('uncertain attempt retained; cannot declare completed')
            db.execute("UPDATE run SET state='complete'")

    def summary(self):
        with self.connect() as db:
            limit,cap,state=db.execute('SELECT max_requests,cap,state FROM run').fetchone()
            count,total=db.execute('SELECT COUNT(*),coalesce(SUM(held),0) FROM calls').fetchone()
            return dict(state=state,max_requests=limit,request_claims=count,
                        max_reservation_micro_cny=cap,cumulative_preflight_micro_cny=total,
                        calls_by_model=dict(db.execute('SELECT model,COUNT(*) FROM calls GROUP BY model')))


class _Connection:
    def __init__(self,path):self.path=path
    def __enter__(self):
        self.db=sqlite3.connect(self.path,timeout=10)
        self.db.execute('PRAGMA synchronous=FULL')
        return self.db
    def __exit__(self,kind,value,trace):
        try:
            if kind is None:self.db.commit()
            else:self.db.rollback()
        finally:self.db.close()


class ScopedTransport:
    def __init__(self,transport,run,reservation_cny):
        self.transport=transport;self.run=run;self.reservation_cny=reservation_cny
        self.mode=transport.mode
    def __call__(self,endpoint,payload):
        ordinal=self.run.claim(self.transport.model,self.reservation_cny)
        result=self.transport(endpoint,payload)
        self.run.finish(ordinal,result['model'])
        return result
