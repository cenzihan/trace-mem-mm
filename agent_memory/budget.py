"""Offline preflight ledger. No provider, credential, or network integration.

Reserve before every billable attempt, including retries. Uncertain attempts retain
their full reservation. Prices must be explicit verified CNY prices, not guesses.
"""
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path

CONSENT_CAP_MICRO_CNY = 500_000_000
MAX_SQLITE_INTEGER = 2**63-1
ALLOWED_MODELS = {"text-embedding-v4", "gpt-4o-mini"}


def micro_cny(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError("cost must be finite and non-negative")
        return int((number * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
    except InvalidOperation as exc:
        raise ValueError("invalid money value") from exc


def token_count(value):
    if type(value) is not int or value < 0:
        raise ValueError("token counts must be non-negative integers")
    return value


@dataclass(frozen=True)
class VerifiedRate:
    model: str
    input_cny_per_million: str
    output_cny_per_million: str
    price_source: str
    checked_at: str
    currency: str = "CNY"
    conversion_basis: str = "verified_fx"

    def validate(self):
        if self.model not in ALLOWED_MODELS:
            raise ValueError("only Academic embedding/LLM models are supported")
        if self.currency != "CNY":
            raise ValueError("verified CNY rates/FX required; no implicit currency conversion")
        if self.conversion_basis not in ("verified_fx", "budget_assumption_not_market_fx"):
            raise ValueError("unknown conversion basis")
        if not isinstance(self.price_source, str) or not self.price_source.strip():
            raise ValueError("verified price source required")
        if not isinstance(self.checked_at, str) or not self.checked_at.strip():
            raise ValueError("price verification timestamp required")
        micro_cny(self.input_cny_per_million)
        micro_cny(self.output_cny_per_million)

    def estimate(self, input_tokens, output_tokens):
        self.validate()
        token_count(input_tokens)
        token_count(output_tokens)
        # Rates are CNY per million tokens, so rate*tokens is micro-CNY.
        total = (Decimal(self.input_cny_per_million) * input_tokens +
                 Decimal(self.output_cny_per_million) * output_tokens)
        return int(total.to_integral_value(rounding=ROUND_CEILING))


class BudgetLedger:
    def __init__(self, path, cap_cny="500"):
        cap = micro_cny(cap_cny)
        if cap > CONSENT_CAP_MICRO_CNY or cap <= 0:
            raise ValueError("local development consent allows at most CNY 500")
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            kind=db.execute('PRAGMA application_id').fetchone()[0]
            if db.execute('PRAGMA user_version').fetchone()[0]>2:
                raise ValueError('future budget schema is not supported')
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if kind not in (0,0x414D4247) or tables & {'memories','smoke_attempts'}:
                raise ValueError('wrong database type for budget ledger')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, retry_of TEXT, model TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL,
                    max_input INTEGER NOT NULL, max_output INTEGER NOT NULL,
                    input_rate TEXT NOT NULL, output_rate TEXT NOT NULL,
                    price_source TEXT NOT NULL, price_checked_at TEXT NOT NULL,
                    reserved INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
                    accounted INTEGER NOT NULL, amount_kind TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS provider_observations (
                    attempt_id TEXT PRIMARY KEY REFERENCES attempts(id), observed_at TEXT NOT NULL,
                    mode TEXT NOT NULL, category TEXT NOT NULL, http_status INTEGER,
                    returned_model TEXT, input_tokens INTEGER, output_tokens INTEGER,
                    elapsed_ms INTEGER NOT NULL, estimated_usd TEXT);
            ''')
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
            if "conversion_basis" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN conversion_basis TEXT NOT NULL DEFAULT 'verified_fx'")
            db.execute('PRAGMA application_id=1095582279')
            db.execute('PRAGMA user_version=2')
            old = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()
            if old and old[0] != cap:
                raise ValueError("existing ledger budget cannot be silently changed")
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?)", (cap,))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, rate, max_input_tokens, max_output_tokens=0, retry_of=None, fixed_reserve_cny=None):
        if not isinstance(rate, VerifiedRate):
            raise ValueError("unknown/unverified price: calls are blocked")
        amount = rate.estimate(max_input_tokens, max_output_tokens)
        if fixed_reserve_cny is not None:
            fixed = micro_cny(fixed_reserve_cny)
            if fixed < amount:
                raise ValueError("fixed reservation does not cover priced token bounds")
            amount = fixed
        if rate.conversion_basis == "budget_assumption_not_market_fx" and fixed_reserve_cny is None:
            raise ValueError("assumed conversion requires an explicit fixed budget hold")
        attempt_id = "attempt_" + uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if retry_of and not db.execute("SELECT 1 FROM attempts WHERE id=?", (retry_of,)).fetchone():
                raise ValueError("retry must reference an existing attempt")
            if db.execute("SELECT 1 FROM attempts WHERE status='needs_reconciliation'").fetchone():
                raise ValueError("unreconciled usage anomaly blocks all further attempts")
            cap = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            used = db.execute("SELECT coalesce(sum(accounted),0) FROM attempts").fetchone()[0]
            if used + amount > cap:
                raise ValueError("hard budget exceeded; do not dispatch")
            db.execute("INSERT INTO attempts (id,retry_of,model,status,created_at,max_input,max_output,input_rate,output_rate,price_source,price_checked_at,reserved,input_tokens,output_tokens,accounted,amount_kind,conversion_basis) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (attempt_id, retry_of, rate.model, "reserved", datetime.now(timezone.utc).isoformat(),
                        max_input_tokens, max_output_tokens, rate.input_cny_per_million,
                        rate.output_cny_per_million, rate.price_source, rate.checked_at,
                        amount, None, None, amount,
                        "fixed_budget_hold" if fixed_reserve_cny is not None else "upper_bound_estimate",
                        rate.conversion_basis))
        return attempt_id

    def mark_dispatched(self, attempt_id):
        """Persist immediately before one request. A retry needs its own reservation."""
        with self.connect() as db:
            cursor = db.execute("UPDATE attempts SET status='dispatched' WHERE id=? AND status='reserved'", (attempt_id,))
            if cursor.rowcount != 1:
                raise ValueError("attempt is absent or already dispatched; never reuse it")

    def settle(self, attempt_id, input_tokens, output_tokens=0):
        token_count(input_tokens)
        token_count(output_tokens)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT model,max_input,max_output,input_rate,output_rate,price_source,price_checked_at,status,conversion_basis,reserved FROM attempts WHERE id=?",
                             (attempt_id,)).fetchone()
            if not row or row[7] != "dispatched":
                raise ValueError("only a dispatched attempt can settle once")
            if row[8] != "verified_fx":
                raise ValueError("budget assumption cannot settle a real CNY bill; retain fixed hold")
            rate = VerifiedRate(row[0], row[3], row[4], row[5], row[6])
            anomaly = input_tokens > row[1] or output_tokens > row[2]
            overflow=input_tokens>MAX_SQLITE_INTEGER or output_tokens>MAX_SQLITE_INTEGER
            amount=row[9] if overflow else rate.estimate(input_tokens,output_tokens)
            if amount>CONSENT_CAP_MICRO_CNY:amount=row[9]
            if overflow:input_tokens=output_tokens=None
            db.execute("UPDATE attempts SET status=?,input_tokens=?,output_tokens=?,accounted=?,amount_kind='usage_estimate' WHERE id=?",
                       ("needs_reconciliation" if anomaly else "settled", input_tokens, output_tokens, amount, attempt_id))
        if anomaly:
            # Persist the reported usage honestly and freeze future reservations.
            raise ValueError("usage exceeds preflight bounds; ledger frozen pending reconciliation")

    def record_usage_keep_hold(self, attempt_id, input_tokens, output_tokens):
        """Record observed tokens; keep the entire CNY hold pending real bill reconciliation."""
        token_count(input_tokens)
        token_count(output_tokens)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT max_input,max_output,status FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if not row or row[2] != "dispatched":
                raise ValueError("only a dispatched attempt can record usage once")
            anomaly = input_tokens > row[0] or output_tokens > row[1]
            if input_tokens>MAX_SQLITE_INTEGER or output_tokens>MAX_SQLITE_INTEGER:
                input_tokens=output_tokens=None
            db.execute("UPDATE attempts SET status=?,input_tokens=?,output_tokens=?,amount_kind='usage_observed_fixed_hold_pending_bill' WHERE id=?",
                       ("needs_reconciliation" if anomaly else "usage_recorded_awaiting_bill", input_tokens, output_tokens, attempt_id))
        if anomaly:
            raise ValueError("usage exceeds bounds; fixed hold retained and future reservations frozen")

    def totals(self):
        with self.connect() as db:
            cap = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            used = db.execute("SELECT coalesce(sum(accounted),0) FROM attempts").fetchone()[0]
            attempts = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
        return {"cap_micro_cny": cap, "accounted_micro_cny": used,
                "remaining_micro_cny": cap - used, "attempts": attempts,
                "scope": "local development only; excludes official evaluation/server costs"}

    def observe_provider(self,attempt_id,*,mode,category,http_status=None,returned_model=None,
                         input_tokens=None,output_tokens=None,elapsed_ms=0,estimated_usd=None):
        """Atomically persist redacted usage and accounting; never store source/response bodies."""
        categories={'success','rate_limited','server_error','authentication','permission_or_region',
                    'client_error','redirect','timeout','network_error','tls_error','response_limit',
                    'invalid_json','invalid_usage','model_mismatch','invalid_contract'}
        if mode not in ('offline_fake','live') or category not in categories:
            raise ValueError('invalid provider observation')
        if http_status is not None and (type(http_status) is not int or not 100<=http_status<=599):
            raise ValueError('invalid HTTP status')
        models={'gpt-4o-mini','gpt-4o-mini-2024-07-18','text-embedding-v4'}
        if returned_model is not None and returned_model not in models:
            raise ValueError('unapproved response model cannot be stored')
        token_count(elapsed_ms)
        if (input_tokens is None)!=(output_tokens is None):raise ValueError('incomplete usage')
        if input_tokens is not None:token_count(input_tokens);token_count(output_tokens)
        overflow=input_tokens is not None and (input_tokens>MAX_SQLITE_INTEGER or output_tokens>MAX_SQLITE_INTEGER)
        if overflow:
            input_tokens=output_tokens=None;estimated_usd=None;category='invalid_usage'
        if estimated_usd is not None:
            number=Decimal(str(estimated_usd))
            if not number.is_finite() or number<0:raise ValueError('invalid USD estimate')
            estimated_usd=str(number)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT max_input,max_output,reserved,input_rate,output_rate,status,conversion_basis FROM attempts WHERE id=?',(attempt_id,)).fetchone()
            if not row or row[5]!='dispatched':raise ValueError('observation requires one dispatched attempt')
            amount=row[2];status='uncertain';kind='retained_upper_bound'
            anomaly=category in ('model_mismatch','invalid_usage') or overflow
            if input_tokens is not None:
                observed=int((Decimal(row[3])*input_tokens+Decimal(row[4])*output_tokens).to_integral_value(rounding=ROUND_CEILING))
                anomaly=anomaly or input_tokens>row[0] or output_tokens>row[1]
                if row[6]=='verified_fx' and not anomaly:
                    amount=observed;status='settled' if category=='success' else 'failed_usage_recorded';kind='provider_usage_estimate'
                elif anomaly and observed<=CONSENT_CAP_MICRO_CNY:amount=max(amount,observed)
            if anomaly:status='needs_reconciliation'
            if overflow:kind='unrepresentable_usage_retained_hold'
            db.execute('UPDATE attempts SET status=?,accounted=?,input_tokens=?,output_tokens=?,amount_kind=? WHERE id=?',
                       (status,amount,input_tokens,output_tokens,kind,attempt_id))
            db.execute('INSERT INTO provider_observations VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (attempt_id,datetime.now(timezone.utc).isoformat(),mode,category,http_status,returned_model,input_tokens,output_tokens,elapsed_ms,estimated_usd))
