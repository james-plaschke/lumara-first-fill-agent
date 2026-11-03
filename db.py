"""
Data layer: a local SQLite database standing in for Calder's systems.

  data/hub_patients.csv                ->  patients        (Salesforce hub extract)
  data/pharmacy_status_YYYY-MM-DD.csv  ->  fill_status     (specialty pharmacy daily file)
  (written by the agent)               ->  safety_flags    (pharmacovigilance intake queue)
                                           callbacks       (hub work queue)
                                           opt_outs        (consent write-back)
                                           blocked_replies (audit of replies the outbound screen stopped)

The database is built from the CSVs on first use. Delete data/calder.db to reset it.
"""

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "calder.db"
SCHEMA_VERSION = 2  # bump when the tables change; an older calder.db is rebuilt from the CSVs

SCHEMA = """
CREATE TABLE patients (
    patient_code TEXT PRIMARY KEY, first_name TEXT, dob TEXT, timezone TEXT,
    preferred_language TEXT, consent_automated_outreach INTEGER,
    copay_card_status TEXT, prescriber_office TEXT
);
CREATE TABLE fill_status (
    patient_code TEXT PRIMARY KEY, stage TEXT, detail TEXT, pharmacy TEXT, file_date TEXT
);
CREATE TABLE safety_flags (
    id INTEGER PRIMARY KEY, kind TEXT, call_id TEXT, patient_code TEXT, reporter TEXT, product TEXT,
    event_verbatim TEXT, identity_verified INTEGER, detection TEXT, call_transcript TEXT,
    received_at_utc TEXT, pv_report_due_utc TEXT
);
CREATE TABLE callbacks (
    id INTEGER PRIMARY KEY, call_id TEXT, patient_code TEXT, team TEXT, preferred_time TEXT,
    note TEXT, source TEXT, call_transcript TEXT, booked_at_utc TEXT
);
CREATE TABLE opt_outs (patient_code TEXT, call_id TEXT, at_utc TEXT);
CREATE TABLE blocked_replies (call_id TEXT, patient_code TEXT, problems TEXT, reply TEXT, at_utc TEXT);
"""


def connect() -> sqlite3.Connection:
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as old:
            if old.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                DB_PATH.unlink()
    is_new = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if is_new:
        build(conn)
    return conn


@contextmanager
def open_db():
    """Open, commit on success, always close."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def build(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    with open(DATA_DIR / "hub_patients.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    conn.executemany(
        "INSERT INTO patients VALUES (:patient_code, :first_name, :dob, :timezone, :preferred_language, "
        ":consent_automated_outreach, :copay_card_status, :prescriber_office)", rows)

    # Load the most recent pharmacy file only; the file date is the "as of" for every status in it.
    latest = sorted(DATA_DIR.glob("pharmacy_status_*.csv"))[-1]
    file_date = latest.stem.replace("pharmacy_status_", "")
    with open(latest, newline="") as f:
        rows = [{**r, "file_date": file_date} for r in csv.DictReader(f)]
    conn.executemany(
        "INSERT INTO fill_status VALUES (:patient_code, :stage, :detail, :pharmacy, :file_date)", rows)
    conn.commit()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# --- reads ------------------------------------------------------------------

def get_patient(patient_code: str) -> Optional[dict]:
    with open_db() as conn:
        row = conn.execute("SELECT * FROM patients WHERE patient_code = ?", (patient_code,)).fetchone()
    return dict(row) if row else None


def get_fill_status(patient_code: str) -> Optional[dict]:
    """None means the patient is missing from the latest pharmacy file."""
    with open_db() as conn:
        row = conn.execute("SELECT * FROM fill_status WHERE patient_code = ?", (patient_code,)).fetchone()
    return dict(row) if row else None


def all_patient_codes() -> list:
    with open_db() as conn:
        return [r[0] for r in conn.execute("SELECT patient_code FROM patients ORDER BY patient_code")]


def read_table(table: str) -> list:
    """For tests and the call log review. Table names are fixed in code, never from input."""
    assert table in {"safety_flags", "callbacks", "opt_outs", "blocked_replies", "patients"}
    with open_db() as conn:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]

# --- writes -----------------------------------------------------------------

def insert_safety_flag(record: dict) -> None:
    with open_db() as conn:
        conn.execute(
            "INSERT INTO safety_flags (kind, call_id, patient_code, reporter, product, event_verbatim, "
            "identity_verified, detection, call_transcript, received_at_utc, pv_report_due_utc) VALUES (:kind, "
            ":call_id, :patient_code, :reporter, :product, :event_verbatim, :identity_verified, :detection, "
            ":call_transcript, :received_at_utc, :pv_report_due_utc)",
            {**record, "detection": json.dumps(record["detection"], ensure_ascii=False)})


def insert_callback(call_id: str, patient_code: str, team: str, preferred_time: str, note: str, source: str,
                    call_transcript: str = "") -> None:
    with open_db() as conn:
        conn.execute(
            "INSERT INTO callbacks (call_id, patient_code, team, preferred_time, note, source, call_transcript, "
            "booked_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (call_id, patient_code, team, preferred_time, note, source, call_transcript, now_utc()))


def record_opt_out(patient_code: str, call_id: str) -> None:
    """Writes the opt-out back to the patient record, so the next call is blocked by the consent check."""
    with open_db() as conn:
        conn.execute("INSERT INTO opt_outs VALUES (?, ?, ?)", (patient_code, call_id, now_utc()))
        conn.execute("UPDATE patients SET consent_automated_outreach = 0 WHERE patient_code = ?", (patient_code,))


def insert_blocked_reply(call_id: str, patient_code: str, problems: list, reply: str) -> None:
    with open_db() as conn:
        conn.execute("INSERT INTO blocked_replies VALUES (?, ?, ?, ?, ?)",
                     (call_id, patient_code, json.dumps(problems), reply, now_utc()))
