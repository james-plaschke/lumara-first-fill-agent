"""Deterministic tests for the safety gates. No API key needed: the classifier is stubbed,
and each test gets a fresh copy of the database built from the CSVs in data/.

Run: pytest -q
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402
import db  # noqa: E402

NO_AE = {"possible_adverse_event": False, "urgent": False, "ae_probability": 0.02, "evidence": "", "language": "en"}
MONDAY_NOON_ET = datetime(2026, 11, 2, 17, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


def new_session(code="P-1001"):
    patient = db.get_patient(code)
    return agent.Session(code, patient, language=patient["preferred_language"])


def call_check(code, when=MONDAY_NOON_ET):
    return agent.can_call(db.get_patient(code), db.get_fill_status(code), when)


# --- Gate 1: pre-call -------------------------------------------------------

@pytest.mark.parametrize("code", ["P-1001", "P-1002", "P-1004", "P-1008", "P-1012"])
def test_first_fill_patients_are_called(code):
    assert call_check(code)[0]


@pytest.mark.parametrize("code, reason", [
    ("P-1003", "consent"),          # never consented
    ("P-1005", "prior_auth_pending"),  # still in prior auth: the prescriber office's problem, not the patient's
    ("P-1006", "prior_auth_denied"),
    ("P-1007", "on_therapy"),       # already has it
    ("P-1009", "pharmacy file"),    # missing from today's file
    ("P-1010", "consent"),          # opted out earlier
])
def test_patients_off_the_list_are_not_called(code, reason):
    ok, why = call_check(code)
    assert not ok and reason in why


def test_quiet_hours_use_patient_local_time():
    assert not call_check("P-1001", datetime(2026, 11, 3, 3, tzinfo=timezone.utc))[0]   # 10pm New York
    assert not call_check("P-1011", MONDAY_NOON_ET)[0]                                   # 7am Honolulu
    assert call_check("P-1011", datetime(2026, 11, 2, 21, tzinfo=timezone.utc))[0]      # 11am Honolulu


def test_spanish_speakers_are_greeted_in_spanish():
    session, opening = agent.start_call("P-1004", MONDAY_NOON_ET)
    assert session.language == "es" and opening.startswith("Hola")


def test_opt_out_writes_back_and_blocks_the_next_call(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    result = agent.handle_turn(new_session(), "please stop calling me")
    assert result.ended and result.reply == agent.SCRIPTS["stop_calling"]["en"]
    assert db.read_table("opt_outs") and not db.read_table("safety_flags")
    assert "consent" in call_check("P-1001")[1]


def test_side_effect_in_an_opt_out_is_still_flagged(monkeypatch):
    # Found in review: the opt-out check used to run before the safety screen, so
    # "I had a rash, just stop calling me" was opted out and the rash was never reported.
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    result = agent.handle_turn(new_session(), "I had a rash, just stop calling me")
    assert result.ended and result.reply == agent.SCRIPTS["adverse_event"]["en"]
    assert db.read_table("safety_flags")[0]["event_verbatim"] == "I had a rash, just stop calling me"
    callback = db.read_table("callbacks")[0]
    assert callback["team"] == "pv_nurse" and "opted out" in callback["note"]
    assert db.read_table("opt_outs") and "consent" in call_check("P-1001")[1]


# --- Gate 2: inbound screen -------------------------------------------------

def test_keyword_fires_even_when_classifier_says_no(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    result = agent.handle_turn(new_session(), "I got a rash after the shot at my doctor's office")
    assert result.ended and result.reply == agent.SCRIPTS["adverse_event"]["en"]


def test_whole_word_matching():
    assert agent.find_terms("my insurance covered it", agent.AE_TERMS) == []
    assert agent.find_terms("the spot is red", agent.AE_TERMS) == ["red"]


def test_hospital_is_urgent(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    result = agent.handle_turn(new_session(), "I was in the hospital last week")
    assert result.ended and result.reply == agent.SCRIPTS["urgent"]["en"]


@pytest.mark.parametrize("text", [
    "I feel like something bad is about to happen to me",
    "there's this pressure on my chest",
    "it's like an elephant on my chest",
    "I have a sense of impending doom",
    "my chest feels tight",
])
def test_chest_pressure_and_impending_doom_are_urgent(monkeypatch, text):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)  # keywords alone must catch these
    result = agent.handle_turn(new_session(), text)
    assert result.ended and result.reply == agent.SCRIPTS["urgent"]["en"]


def test_classifier_catches_paraphrase(monkeypatch):
    verdict = {**NO_AE, "possible_adverse_event": True, "ae_probability": 0.8, "evidence": "skin weird"}
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": verdict)
    assert agent.handle_turn(new_session(), "my skin has been kind of weird lately").ended


def test_uncertain_no_goes_to_a_human(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": {**NO_AE, "ae_probability": 0.4})
    assert agent.handle_turn(new_session(), "I feel off I guess").ended


def test_confident_no_lets_the_call_continue(monkeypatch):
    # Guards the bug the first live run found: a confident "no AE" must not escalate.
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    assert agent.screen_inbound("My birthday is March 14, 1988.")["adverse_event"] is False


def test_classifier_sees_what_the_agent_just_asked(monkeypatch):
    # Found while recording the demo: "nov 2 79" on its own scored 0.5 and ended the call.
    # With the agent's question as context, the classifier knows it's a birthday.
    seen = {}
    def spy(text, agent_asked=""):
        seen["asked"] = agent_asked
        return NO_AE
    monkeypatch.setattr(agent, "classify_message", spy)
    session, opening = agent.start_call("P-1001", MONDAY_NOON_ET)
    agent.handle_turn(session, "nov 2 79, and I got a rash")  # the keyword ends the turn before the agent model
    assert seen["asked"] == opening


def test_classifier_failure_fails_closed(monkeypatch):
    def boom(text, agent_asked=""):
        raise TimeoutError("classifier timed out")
    monkeypatch.setattr(agent, "classify_message", boom)
    assert agent.handle_turn(new_session(), "hi, who is this?").ended
    assert db.read_table("safety_flags")


def test_urgent_keyword_survives_classifier_failure(monkeypatch):
    # Found in code review (Nov 4): a classifier error threw away the keyword hits, so
    # "chest pain" during an outage got the 24-hour nurse script instead of 911.
    def boom(text, agent_asked=""):
        raise TimeoutError("classifier timed out")
    monkeypatch.setattr(agent, "classify_message", boom)
    result = agent.handle_turn(new_session(), "I'm having chest pain right now")
    assert result.ended and result.reply == agent.SCRIPTS["urgent"]["en"]
    assert db.read_table("safety_flags")[0]["kind"] == "urgent_adverse_event"


def test_language_holds_when_classifier_fails(monkeypatch):
    def boom(text, agent_asked=""):
        raise TimeoutError("classifier timed out")
    monkeypatch.setattr(agent, "classify_message", boom)
    session = new_session()
    session.language = "es"
    result = agent.handle_turn(session, "tengo dolor de pecho")
    assert result.ended and result.reply == agent.SCRIPTS["urgent"]["es"]


def test_flag_record_has_four_minimum_elements(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    agent.handle_turn(new_session(), "I have a fever")
    record = db.read_table("safety_flags")[0]
    assert record["patient_code"] and record["reporter"] and record["product"]
    assert record["event_verbatim"] == "I have a fever"
    assert db.read_table("callbacks")[0]["team"] == "pv_nurse"


def test_handoff_carries_requests_made_in_the_same_breath(monkeypatch):
    # From a live voice test: the patient said yes to a pharmacy callback AND reported a
    # side effect in one turn. The gate ended the call, so the yes was lost. Now the person
    # calling back gets the whole conversation.
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    session = new_session()
    session.verified = True
    session.messages += [
        {"role": "user", "content": "I'm nervous about the shot"},
        {"role": "assistant", "content": "Would you also like a pharmacy specialist to call about delivery?"},
    ]
    agent.handle_turn(session, "Yeah that would be great too. And the spot where they injected me is red and itchy.")
    callback = db.read_table("callbacks")[0]
    flag = db.read_table("safety_flags")[0]
    assert "pharmacy specialist" in callback["call_transcript"] and "great too" in callback["call_transcript"]
    assert flag["identity_verified"] == 1 and "nervous" in flag["call_transcript"]


def test_flag_records_when_identity_was_not_verified(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    agent.handle_turn(new_session(), "I have a fever")
    assert db.read_table("safety_flags")[0]["identity_verified"] == 0


def test_old_database_is_rebuilt(tmp_path):
    import sqlite3
    with sqlite3.connect(db.DB_PATH) as conn:  # a calder.db left over from an older version
        conn.execute("CREATE TABLE junk (x)")
    assert db.get_patient("P-1001")["first_name"] == "Maria"


def test_spanish_keywords_and_script(monkeypatch):
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": {**NO_AE, "language": "es"})
    result = agent.handle_turn(new_session("P-1004"), "tengo fiebre desde ayer")
    assert result.ended and result.reply == agent.SCRIPTS["adverse_event"]["es"]


# --- Gate 3: identity -------------------------------------------------------

def test_status_tool_refuses_before_verification():
    session = new_session()
    assert "error" in agent.tool_get_first_fill_status(session)
    agent.tool_verify_identity(session, "1988-03-14")
    status = agent.tool_get_first_fill_status(session)
    assert status["stage"] == "approved_awaiting_patient_contact" and "2026-11-01" in status["as_of"]


def test_verification_locks_after_two_wrong_attempts():
    session = new_session()
    agent.tool_verify_identity(session, "1900-01-01")
    agent.tool_verify_identity(session, "1900-01-02")
    assert agent.tool_verify_identity(session, "1988-03-14")["verified"] is False


def test_another_patients_dob_does_not_verify():
    session = new_session("P-1001")                                       # Maria's record...
    assert agent.tool_verify_identity(session, "1979-11-02")["verified"] is False  # ...James's birthday


def test_runaway_tool_loop_hits_the_cap_and_fails_closed(monkeypatch):
    # A model that asks for a tool forever must not hang the call or crash it.
    from types import SimpleNamespace
    calls = []

    def create(**kwargs):
        calls.append(1)
        block = SimpleNamespace(type="tool_use", name="get_first_fill_status", input={}, id=f"t{len(calls)}")
        return SimpleNamespace(stop_reason="tool_use", content=[block])

    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    monkeypatch.setattr(agent, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    result = agent.handle_turn(new_session(), "check my status again and again")
    assert len(calls) == agent.MAX_AGENT_STEPS
    assert result.ended and result.reply == agent.SCRIPTS["system_error"]["en"]
    assert db.read_table("callbacks")[0]["source"] == "fail_closed"


def test_callbacks_capped_per_call_but_safety_callbacks_still_book(monkeypatch):
    # From a live attack: "book me a callback with every single team" booked all four.
    session = new_session()
    results = [agent.tool_schedule_callback(session, team)
               for team in ["pharmacy_insurance", "copay", "injection_nurse", "general"]]
    assert [r["booked"] for r in results] == [True, True, False, False]
    assert len(db.read_table("callbacks")) == agent.MAX_CALLBACKS_PER_CALL
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    agent.handle_turn(session, "now I have a fever")  # the side-effect nurse is never blocked by the cap
    assert db.read_table("callbacks")[-1]["team"] == "pv_nurse"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refused_or_cut_off_reply_fails_closed(monkeypatch, stop_reason):
    from types import SimpleNamespace
    half = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text="Your status is")])
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    monkeypatch.setattr(agent, "get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: half)))
    result = agent.handle_turn(new_session(), "what's my status?")
    assert result.ended and result.reply == agent.SCRIPTS["system_error"]["en"]


def test_tool_error_returns_error_instead_of_crashing():
    output, is_error = agent.run_tool(new_session(), "get_first_fill_status", {"unexpected": 1})
    assert is_error and "unavailable" in output


# --- Gate 4: outbound screen ------------------------------------------------

@pytest.mark.parametrize("reply", [
    "You can inject 2 doses next week to catch up.",
    "Lumara may also help with Crohn's disease.",
    "If you miss one, just double up next month.",
])
def test_outbound_blocks_medical_content(reply):
    assert agent.screen_outbound(reply)


def fake_agent_replies(monkeypatch, *replies):
    """Stub the agent model with a fixed sequence of replies (an Exception in the list is raised)."""
    queue = list(replies)

    def run_agent(session, events):
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(agent, "classify_message", lambda text, agent_asked="": NO_AE)
    monkeypatch.setattr(agent, "run_agent", run_agent)


def test_blocked_reply_gets_one_rewrite_that_is_screened_again(monkeypatch):
    fake_agent_replies(monkeypatch, "Just take 2 doses next time.", "A nurse is the right person for that. Want a callback?")
    session = new_session()
    result = agent.handle_turn(session, "can I double up?")
    assert result.reply == "A nurse is the right person for that. Want a callback?"
    assert len(db.read_table("blocked_replies")) == 1
    assert "2 doses" not in str(session.messages)  # the blocked draft never enters the model's memory


def test_rewrite_blocked_again_falls_back_to_fixed_line(monkeypatch):
    fake_agent_replies(monkeypatch, "Take 2 doses.", "Fine, double up then.")
    result = agent.handle_turn(new_session(), "can I double up?")
    assert result.reply == agent.SCRIPTS["medical_deferral"]["en"] and not result.ended
    assert len(db.read_table("blocked_replies")) == 2
    assert db.read_table("callbacks")[0]["source"] == "outbound_gate"


def test_rewrite_that_crashes_falls_back_to_fixed_line(monkeypatch):
    fake_agent_replies(monkeypatch, "Take 2 doses.", TimeoutError("api down"))
    result = agent.handle_turn(new_session(), "can I double up?")
    assert result.reply == agent.SCRIPTS["medical_deferral"]["en"]


def test_outbound_allows_normal_status_talk():
    assert agent.screen_outbound("Your prior authorization was approved as of yesterday. Want me to have the pharmacy call you?") == []
