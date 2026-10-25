"""Persistent authorization accounting only. No submission/network client exists."""
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path


class EvaluationLedger:
    def __init__(self,path,cap=5):
        if type(cap) is not int or not 1<=cap<=5:raise ValueError('Smoke user cap cannot exceed five')
        self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            kind=db.execute('PRAGMA application_id').fetchone()[0]
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if kind not in (0,0x414D4556) or tables & {'memories','budget'}:raise ValueError('wrong database type')
            db.executescript('''CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1),cap INTEGER NOT NULL);
              CREATE TABLE IF NOT EXISTS smoke_attempts(id TEXT PRIMARY KEY,claimed_at REAL NOT NULL,status TEXT NOT NULL,official_job TEXT);''')
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT cap FROM settings WHERE id=1').fetchone()
            if old and old[0]!=cap:raise ValueError('existing smoke cap cannot be silently changed')
            db.execute('INSERT OR IGNORE INTO settings VALUES(1,?)',(cap,))
            db.execute('PRAGMA application_id=1095583062');db.execute('PRAGMA user_version=1')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30);db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()

    def reserve_smoke(self,*,access_approved=False,credential_established=False,endpoints_authorized=False,submission_approved=False,now=None):
        if not all((access_approved,credential_established,endpoints_authorized,submission_approved)):
            raise ValueError('Smoke not ready/approved; no attempt reserved')
        timestamp=(now or datetime.now(timezone.utc)).timestamp()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cap=db.execute('SELECT cap FROM settings WHERE id=1').fetchone()[0]
            count,last=db.execute('SELECT count(*),max(claimed_at) FROM smoke_attempts').fetchone()
            if count>=cap:raise ValueError('whole-task smoke cap exhausted')
            if last is not None and timestamp-last<3600:raise ValueError('hourly smoke gate not elapsed')
            identifier='smoke_'+uuid.uuid4().hex
            db.execute('INSERT INTO smoke_attempts VALUES(?,?,?,NULL)',(identifier,timestamp,'reserved_before_submission'))
        return identifier

    def record_result(self,identifier,status,official_job=None):
        if status not in ('complete','failed','uncertain'):raise ValueError('unsupported state')
        if official_job is not None and not re.fullmatch(r'[A-Za-z0-9:_-]{1,128}',official_job):raise ValueError('invalid job reference')
        with self.connect() as db:
            changed=db.execute("UPDATE smoke_attempts SET status=?,official_job=? WHERE id=? AND status='reserved_before_submission'",(status,official_job,identifier)).rowcount
            if changed!=1:raise ValueError('attempt absent or already recorded')

    def reserve_full(self):raise ValueError('Full is forbidden')

    def summary(self):
        with self.connect() as db:
            cap=db.execute('SELECT cap FROM settings WHERE id=1').fetchone()[0]
            used,last=db.execute('SELECT count(*),max(claimed_at) FROM smoke_attempts').fetchone()
        return dict(user_cap=cap,attempts_including_uncertain=used,remaining=cap-used,
                    next_time_utc=datetime.fromtimestamp(last+3600,timezone.utc).isoformat() if last is not None else None,
                    full_allowed=False,network_submission_implemented=False)
