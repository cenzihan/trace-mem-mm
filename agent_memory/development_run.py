"""Durable bounded development runs; counters contain no prompts or credentials."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from .budget import micro_cny


class DevelopmentRun:
    def __init__(self,path,*,create=False,max_requests=32,max_reservation_cny='10',approval_scope='standard_50_10'):
        self.path=Path(path)
        if create:
            scopes={'standard_50_10':(50,10000000),'native_100_20':(100,20000000),
                    'atm_v5_embedding_1209_6045':(1209,60450000)}
            if approval_scope not in scopes:raise ValueError('unknown explicit approval scope')
            allowed_requests,allowed_money=scopes[approval_scope]
            if not 1<=max_requests<=allowed_requests or not 0<micro_cny(max_reservation_cny)<=allowed_money:
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
            if db.execute("SELECT 1 FROM calls WHERE state NOT IN ('complete','rejected_response') LIMIT 1").fetchone():
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
            if db.execute("SELECT 1 FROM calls WHERE state NOT IN ('complete','rejected_response')").fetchone():
                raise ValueError('uncertain attempt retained; cannot declare completed')
            db.execute("UPDATE run SET state='complete'")

    def tighten_limits(self,max_requests,max_reservation_cny):
        """Lower an existing run's caps without resetting claims or reservations."""
        amount=micro_cny(max_reservation_cny)
        if type(max_requests) is not int or max_requests<1 or amount<=0:raise ValueError('positive tighter limits required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old_requests,old_cap,state=db.execute('SELECT max_requests,cap,state FROM run').fetchone()
            count,held=db.execute('SELECT COUNT(*),coalesce(SUM(held),0) FROM calls').fetchone()
            if state!='active' or not count<=max_requests<=old_requests or not held<=amount<=old_cap:
                raise ValueError('limits may only decrease without hiding existing usage')
            db.execute('UPDATE run SET max_requests=?,cap=? WHERE id=1',(max_requests,amount))

    def acknowledge_known_contract_rejection(self,ordinal,ledger_path,attempt_id):
        """Explicit recovery only: a completed HTTP200 response rejected by schema.

        Never accepts a timeout, missing usage, mismatched model, or usage overflow.
        Preserves the rejected slot AND its full budget hold; does not dispatch,
        repeat, settle, release, or mark the provider request successful.
        """
        path=Path(ledger_path).resolve()
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as ledger:
            ledger.execute('PRAGMA query_only=ON')
            row=ledger.execute('''SELECT a.model,a.created_at,a.max_input,a.max_output,a.input_tokens,a.output_tokens,
                o.category,o.http_status,o.returned_model,o.observed_at
                FROM attempts a JOIN provider_observations o ON a.id=o.attempt_id WHERE a.id=?''',(attempt_id,)).fetchone()
        if not row:raise ValueError('no durable observed provider response')
        model,created,imax,omax,inp,out,category,status,returned,observed=row
        allowed={'gpt-4o-mini','gpt-4o-mini-2024-07-18'} if model=='gpt-4o-mini' else {'text-embedding-v4'}
        if (category!='invalid_contract' or status!=200 or returned not in allowed or
            type(inp) is not int or type(out) is not int or not 0<=inp<=imax or not 0<=out<=omax):
            raise ValueError('unknown/rejected provider outcome cannot be acknowledged as completed response')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            call=db.execute('SELECT model,state,started_at FROM calls WHERE ordinal=?',(ordinal,)).fetchone()
            if not call or call[0]!=model or call[1]!='claimed' or not call[2]<=created<=observed:
                raise ValueError('provider observation does not match the unfinished run slot')
            db.execute('''CREATE TABLE IF NOT EXISTS response_rejections(
                ordinal INTEGER PRIMARY KEY,attempt_id TEXT UNIQUE,category TEXT,http_status INTEGER,
                input_tokens INTEGER,output_tokens INTEGER,acknowledged_at TEXT)''')
            now=datetime.now(timezone.utc).isoformat()
            db.execute('INSERT INTO response_rejections VALUES(?,?,?,?,?,?,?)',
                       (ordinal,attempt_id,category,status,inp,out,now))
            db.execute("UPDATE calls SET state='rejected_response',finished_at=?,returned_model=? WHERE ordinal=?",
                       (now,returned,ordinal))

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
