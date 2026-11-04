"""
Lumara first-fill nudge agent (prototype, fake data only).

An outbound call to a patient whose Lumara prescription was approved but who
has not received their first shipment. The agent verifies identity, finds the
blocker, explains the status, and books a human callback when needed.

Safety lives in code, not just the prompt:
  1. Pre-call gate:   only patients on the first-fill call list (approved or shipped, not yet in hand),
                      with consent to automated outreach, between 8am and 9pm patient local time.
  2. Inbound gate:    every patient message is screened BEFORE the agent model sees it
                      (keyword tripwire + separate classifier model). Possible adverse
                      event -> fixed script, flag record, nurse callback, call ends.
  3. Identity gate:   the status tool refuses to return anything until DOB is verified.
  4. Outbound gate:   every agent reply is screened BEFORE the patient hears it for
                      dosing numbers, off-label conditions, and medical advice.
  5. Fail closed:     if the screen or the agent model errors, escalate to a human and end the call.

Files: content.py holds every word the agent says or reads (scripts, prompts, word lists).
       chat.py and voice.py are front ends; both call handle_turn, so every gate always runs.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import anthropic
from dotenv import load_dotenv

import db
from content import (
    ADVICE_PHRASES, AE_TERMS, CLASSIFIER_PROMPT, CLASSIFIER_SCHEMA, DOSE_PATTERN, OFF_LABEL_TERMS,
    REWRITE_NOTE, SCRIPTS, STOP_CALLING_TERMS, SYSTEM_PROMPT, TOOLS, URGENT_TERMS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_MODEL = "claude-sonnet-5-5"       # the conversational agent
CLASSIFIER_MODEL = "claude-haiku-5-5"   # the separate adverse event screener
PRODUCT = "Lumara (lumaribart)"

QUIET_HOURS_START = 8    # no outreach before 8am patient local time
QUIET_HOURS_END = 21     # ...or at/after 9pm
AE_ESCALATE_AT = 0.3     # classifier's AE probability at or above this -> human, even if it said "no"
MAX_VERIFY_ATTEMPTS = 2
MAX_AGENT_STEPS = 5      # cap on model calls per patient turn (tool loops)
MAX_CALLBACKS_PER_CALL = 2  # callbacks the agent may book per call; safety callbacks are always booked
LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}

# ---------------------------------------------------------------------------
# Who this agent calls. Patient data lives in db.py (hub extract + pharmacy daily file).
# ---------------------------------------------------------------------------

# Approved or shipped, but not yet in the patient's hands. Anyone still in prior auth is
# not called: their blocker is the prescriber's office, which is a different agent's job.
FIRST_FILL_STAGES = {
    "approved_awaiting_patient_contact",
    "approved_patient_declined",
    "approved_patient_postponed",
    "approved_benefits_reverification",
    "shipped",
}


# ---------------------------------------------------------------------------
# Call state
# ---------------------------------------------------------------------------

@dataclass
class Session:
    patient_code: str
    patient: dict
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: list = field(default_factory=list)
    language: str = "en"
    verified: bool = False
    verify_attempts: int = 0
    callbacks_booked: int = 0
    ended: bool = False


@dataclass
class TurnResult:
    reply: str
    ended: bool = False
    events: list = field(default_factory=list)  # what happened under the hood, for logs and tests


client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global client
    if client is None:
        load_dotenv()
        client = anthropic.Anthropic(timeout=30, max_retries=2)
    return client

# ---------------------------------------------------------------------------
# Records (db.py tables stand in for the PV intake queue and the hub work queue)
# ---------------------------------------------------------------------------

def flag_safety_event(session: Session, kind: str, patient_text: str, screen: dict) -> dict:
    """One record per possible adverse event. Holds the four minimum elements a safety
    case needs: identifiable patient, reporter, suspect product, event (verbatim)."""
    now = datetime.now(timezone.utc)
    record = {
        "kind": kind,
        "call_id": session.call_id,
        "patient_code": session.patient_code,
        "reporter": "patient",
        "product": PRODUCT,
        "event_verbatim": patient_text,
        "identity_verified": session.verified,  # the reporter may not be the patient yet
        "detection": screen,
        "call_transcript": call_transcript(session, patient_text),
        "received_at_utc": now.isoformat(timespec="seconds"),
        "pv_report_due_utc": (now + timedelta(hours=24)).isoformat(timespec="seconds"),
    }
    db.insert_safety_flag(record)
    return record


def book_callback(session: Session, team: str, preferred_time: str = "", note: str = "", source: str = "agent",
                  transcript: str = "") -> None:
    db.insert_callback(session.call_id, session.patient_code, team, preferred_time, note, source, transcript)


def record_blocked_reply(session: Session, events: list, problems: list, reply: str) -> None:
    events.append({"outbound_blocked": problems, "original_reply": reply})
    db.insert_blocked_reply(session.call_id, session.patient_code, problems, reply)


def call_transcript(session: Session, latest_patient_text: str = "") -> str:
    """The conversation so far, so whoever calls back sees everything the patient asked for,
    not just the line that ended the call."""
    lines = []
    for message in session.messages:
        if message["role"] != "system" and isinstance(message["content"], str) and message["content"] != "(call connected)":
            speaker = "Agent" if message["role"] == "assistant" else "Patient"
            lines.append(f"{speaker}: {message['content']}")
    if latest_patient_text:
        lines.append(f"Patient: {latest_patient_text}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Gate 1: pre-call (call list + consent + quiet hours)
# ---------------------------------------------------------------------------

def can_call(patient: dict, status: Optional[dict], now_utc: datetime) -> tuple:
    if not patient["consent_automated_outreach"]:
        return False, "no consent to automated outreach on file"
    if status is None:
        return False, "missing from the latest pharmacy file, needs data review"
    if status["stage"] not in FIRST_FILL_STAGES:
        return False, f"not on the first-fill list (stage: {status['stage']})"
    local = now_utc.astimezone(ZoneInfo(patient["timezone"]))
    if not (QUIET_HOURS_START <= local.hour < QUIET_HOURS_END):
        return False, f"outside calling hours ({local:%I:%M %p} patient local time)"
    return True, "ok"

# ---------------------------------------------------------------------------
# Gate 2: inbound screen (runs on every patient message before the agent sees it)
# ---------------------------------------------------------------------------

def find_terms(text: str, terms: list) -> list:
    """Whole-word, case-insensitive matches, so 'red' does not fire inside 'covered'."""
    return [t for t in terms if re.search(rf"\b{re.escape(t)}\b", text, re.I)]


def classify_message(text: str, agent_asked: str = "") -> dict:
    # The agent's last line is context only, so a short answer like "nov 2 79" reads as the birthday it is.
    context = f"The agent just said:\n<agent>{agent_asked}</agent>\n\n" if agent_asked else ""
    response = get_client().messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=1024,
        system=CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": f"{context}Patient message:\n<message>{text}</message>"}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": CLASSIFIER_SCHEMA}},
    )
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def screen_inbound(text: str, agent_asked: str = "") -> dict:
    """Keyword tripwire OR classifier -> escalate. The keywords cannot be argued with;
    the classifier catches paraphrase and Spanish. Either one alone is enough."""
    urgent_hits = find_terms(text, URGENT_TERMS)
    ae_hits = find_terms(text, AE_TERMS)
    try:
        verdict = classify_message(text, agent_asked)
    except Exception as e:
        # Fail closed, but keep what the keywords found: "heart attack" during an outage
        # still gets the 911 script, not the 24-hour nurse one. Language stays as it was.
        return {"urgent": bool(urgent_hits), "adverse_event": True,
                "keyword_hits": urgent_hits + ae_hits, "error": f"{type(e).__name__}: {e}"}

    maybe = verdict["ae_probability"] >= AE_ESCALATE_AT  # a "no" that isn't a confident no
    return {
        "urgent": bool(urgent_hits) or verdict["urgent"],
        "adverse_event": bool(urgent_hits or ae_hits) or verdict["possible_adverse_event"] or maybe,
        "keyword_hits": urgent_hits + ae_hits,
        "classifier": verdict,
        "language": verdict["language"],
    }

# ---------------------------------------------------------------------------
# Gate 4: outbound screen (runs on every agent reply before the patient hears it)
# ---------------------------------------------------------------------------

def screen_outbound(reply: str) -> list:
    problems = []
    if DOSE_PATTERN.search(reply):
        problems.append("dose_number")
    problems += [f"off_label:{t}" for t in OFF_LABEL_TERMS if t in reply.lower()]
    problems += [f"advice:{p}" for p in ADVICE_PHRASES if p in reply.lower()]
    return problems

# ---------------------------------------------------------------------------
# Tools (Gate 3, identity, is enforced inside get_first_fill_status)
# ---------------------------------------------------------------------------

def tool_verify_identity(session: Session, dob: str) -> dict:
    if session.verify_attempts >= MAX_VERIFY_ATTEMPTS:
        return {"verified": False, "message": "Too many attempts. Offer a callback from a person."}
    session.verify_attempts += 1
    session.verified = dob.strip() == session.patient["dob"]
    return {"verified": session.verified}


def tool_get_first_fill_status(session: Session) -> dict:
    if not session.verified:
        return {"error": "Identity not verified. Ask for date of birth first."}
    status = db.get_fill_status(session.patient_code)
    if status is None:
        return {"error": "No status in the latest pharmacy file. Offer a callback from the pharmacy team."}
    return {
        "stage": status["stage"], "detail": status["detail"], "pharmacy": status["pharmacy"],
        "copay_card": session.patient["copay_card_status"],
        "as_of": f"the pharmacy's daily update from {status['file_date']}",
    }


def tool_schedule_callback(session: Session, team: str, preferred_time: str = "", note: str = "") -> dict:
    if session.callbacks_booked >= MAX_CALLBACKS_PER_CALL:  # stops one call from flooding the hub's queue
        return {"booked": False, "reason": "Callback limit for this call reached. Tell the patient the teams "
                                           "already booked will help with anything else."}
    session.callbacks_booked += 1
    book_callback(session, team, preferred_time, note)
    return {
        "booked": True, "team": team, "patient_preferred_time": preferred_time or "none given",
        "what_to_tell_patient": "The team calls back during business hours, Monday to Friday. "
                                "Their preferred time is noted but not guaranteed.",
    }


TOOL_FUNCTIONS = {
    "verify_identity": tool_verify_identity,
    "get_first_fill_status": tool_get_first_fill_status,
    "schedule_callback": tool_schedule_callback,
}


def run_tool(session: Session, name: str, args: dict) -> tuple:
    """Returns (result_json, is_error). A failing tool tells the model it is unavailable
    instead of crashing the call; the prompt already says to offer a person."""
    try:
        return json.dumps(TOOL_FUNCTIONS[name](session, **args)), False
    except Exception as e:
        return json.dumps({"error": f"{name} is unavailable right now ({type(e).__name__})."}), True

# ---------------------------------------------------------------------------
# The turn: screen in -> agent (with tools) -> screen out
# ---------------------------------------------------------------------------

def start_call(patient_code: str, now_utc: Optional[datetime] = None) -> tuple:
    """Returns (session, opening_line), or (None, reason) if the call must not happen."""
    patient = db.get_patient(patient_code)
    if patient is None:
        return None, f"unknown patient {patient_code}"
    ok, reason = can_call(patient, db.get_fill_status(patient_code), now_utc or datetime.now(timezone.utc))
    if not ok:
        return None, f"call blocked: {reason}"
    session = Session(patient_code, patient, language=patient["preferred_language"])
    opening = SCRIPTS["greeting"][session.language].format(first_name=patient["first_name"])
    session.messages.append({"role": "user", "content": "(call connected)"})
    session.messages.append({"role": "assistant", "content": opening})
    return session, opening


def end_with(session: Session, script: str, events: list) -> TurnResult:
    session.ended = True
    return TurnResult(SCRIPTS[script][session.language], ended=True, events=events)


def handle_turn(session: Session, patient_text: str) -> TurnResult:
    if session.ended:
        return TurnResult("", ended=True)
    events = []

    # Gate 2: inbound screen, on every message, opt-outs included, so a side effect said in the
    # same breath as "stop calling me" is never lost. If the screen itself fails, assume the worst.
    agent_asked = next((m["content"] for m in reversed(session.messages)
                        if m["role"] == "assistant" and isinstance(m["content"], str)), "")
    try:
        screen = screen_inbound(patient_text, agent_asked)
        session.language = screen.get("language", session.language)
    except Exception as e:
        screen = {"urgent": False, "adverse_event": True, "error": f"{type(e).__name__}: {e}"}
    events.append({"inbound_screen": screen})

    # Opt-out is honored in code. A side effect still gets its nurse callback: that's a person, not this agent.
    opted_out = bool(find_terms(patient_text, STOP_CALLING_TERMS))
    if opted_out:
        db.record_opt_out(session.patient_code, session.call_id)
        events.append("opt_out recorded")

    if screen["urgent"] or screen["adverse_event"]:
        kind = "urgent_adverse_event" if screen["urgent"] else "possible_adverse_event"
        flag_safety_event(session, kind, patient_text, screen)
        note = f"{kind}. Read the full call: other requests may be open."
        if opted_out:
            note += " Patient also opted out of automated calls."
        book_callback(session, "pv_nurse", "within 24 hours", note,
                      source="safety_gate", transcript=call_transcript(session, patient_text))
        events.append(f"FLAGGED {kind} -> nurse callback booked, call ended")
        return end_with(session, "urgent" if screen["urgent"] else "adverse_event", events)
    if opted_out:
        return end_with(session, "stop_calling", events)

    # Agent loop: the model may call tools a few times before answering.
    session.messages.append({"role": "user", "content": patient_text})
    # Language is enforced per turn from the screen's detection, not left to the prompt.
    session.messages.append({"role": "system", "content": f"Reply only in {LANGUAGE_NAMES[session.language]}."})
    try:
        reply = run_agent(session, events)
    except Exception as e:
        events.append(f"agent error {type(e).__name__}: {e}")
        book_callback(session, "general", note="automated call failed", source="fail_closed",
                      transcript=call_transcript(session))
        return end_with(session, "system_error", events)

    # Gate 4: outbound screen. The patient never hears a blocked reply.
    problems = screen_outbound(reply)
    if problems:
        record_blocked_reply(session, events, problems, reply)
        # One rewrite: tell the model why it was stopped, ask again, and screen the new reply too.
        session.messages.append({"role": "system", "content": REWRITE_NOTE.format(problems=", ".join(problems))})
        events.append("asked the model for one rewrite")
        try:
            reply = run_agent(session, events)
            problems = screen_outbound(reply)
            if problems:
                record_blocked_reply(session, events, problems, reply)
        except Exception as e:
            events.append(f"rewrite failed: {type(e).__name__}")
        if problems:  # still blocked, or the rewrite failed: the fixed line, and a nurse follows up
            book_callback(session, "injection_nurse", note="medical question deferred", source="outbound_gate")
            reply = SCRIPTS["medical_deferral"][session.language]

    session.messages.append({"role": "assistant", "content": reply})
    return TurnResult(reply, events=events)


def run_agent(session: Session, events: list) -> str:
    system = SYSTEM_PROMPT.format(first_name=session.patient["first_name"])
    for _ in range(MAX_AGENT_STEPS):
        response = get_client().messages.create(
            model=AGENT_MODEL,
            max_tokens=2048,
            system=system,
            tools=TOOLS,
            messages=session.messages,
            output_config={"effort": "low"},
        )
        if response.stop_reason in ("refusal", "max_tokens"):  # declined, or cut off mid-reply: fail closed
            raise RuntimeError(f"model stopped early: {response.stop_reason}")
        if response.stop_reason != "tool_use":
            # Only the text goes into history; handle_turn appends it after the outbound screen.
            return " ".join(b.text for b in response.content if b.type == "text").strip()

        session.messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output, is_error = run_tool(session, block.name, block.input)
                events.append({"tool": block.name, "input": block.input, "output": json.loads(output)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
        session.messages.append({"role": "user", "content": results})
    raise RuntimeError("agent did not finish within MAX_AGENT_STEPS")
