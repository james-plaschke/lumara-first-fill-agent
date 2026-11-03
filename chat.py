"""
Talk to the agent in the terminal.

Run:  python chat.py [PATIENT_CODE]     talk to the agent as that patient (it prints their DOB)
      python chat.py --call-list      who would be called right now, and why not
"""

import json
import sys
from datetime import datetime, timezone
from typing import Optional

import agent
import db


def describe(event) -> str:
    """One readable line per event, so the demo shows what the code did."""
    if isinstance(event, str):
        return event
    if "inbound_screen" in event:
        s = event["inbound_screen"]
        if "error" in s:
            return f"screen ERROR, failing closed: {s['error']}"
        c = s["classifier"]
        return (f"screen: keywords={s['keyword_hits'] or 'none'} | classifier ae={c['possible_adverse_event']} "
                f"urgent={c['urgent']} p={c['ae_probability']} lang={c['language']}")
    if "tool" in event:
        return f"tool {event['tool']}({json.dumps(event['input'], ensure_ascii=False)}) -> {json.dumps(event['output'], ensure_ascii=False)}"
    if "outbound_blocked" in event:
        return f"OUTBOUND BLOCKED {event['outbound_blocked']}: model wanted to say: {event['original_reply']!r}"
    return json.dumps(event, ensure_ascii=False)


def print_call_list(now_utc: Optional[datetime] = None) -> None:
    now_utc = now_utc or datetime.now(timezone.utc)
    print(f"First-fill call list at {now_utc:%Y-%m-%d %H:%M} UTC\n")
    for code in db.all_patient_codes():
        patient, status = db.get_patient(code), db.get_fill_status(code)
        ok, reason = agent.can_call(patient, status, now_utc)
        print(f"  {code}  {patient['first_name']:<8} {'CALL' if ok else 'skip':<5} {'' if ok else reason}")


def main() -> None:
    if sys.argv[1:] == ["--call-list"]:
        print_call_list()
        return
    patient_code = sys.argv[1] if len(sys.argv) > 1 else "P-1001"
    session, opening = agent.start_call(patient_code)
    if session is None:
        print(f"[not calling] {opening}")
        return
    print(f"[call {session.call_id} to {patient_code}. Type 'quit' to hang up.]")
    print(f"[demo only: you are {session.patient['first_name']}, DOB {session.patient['dob']}]\n")
    print(f"Agent: {opening}\n")
    while not session.ended:
        try:
            text = input("Patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "bye"}:
            break
        if not text:
            continue
        result = agent.handle_turn(session, text)
        for event in result.events:
            print(f"   [{describe(event)}]")
        print(f"\nAgent: {result.reply}\n")
    print("[call ended]")


if __name__ == "__main__":
    main()
