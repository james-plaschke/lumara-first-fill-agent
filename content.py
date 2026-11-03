"""
Everything the agent says or reads, kept apart from the logic in agent.py.

In a real deployment this is the file that goes through MLR (medical, legal, regulatory)
review: the fixed scripts, the prompts, and the safety word lists. Changing a word here
never changes how the gates work.
"""

import re

# ---------------------------------------------------------------------------
# Safety word lists (demo stand-in for a validated classifier; see README)
# ---------------------------------------------------------------------------

URGENT_TERMS = [
    "trouble breathing", "hard to breathe", "can't breathe", "cannot breathe", "short of breath",
    "chest pain", "throat is closing", "throat closing", "lips are swelling", "face is swelling",
    "chest pressure", "pressure on my chest", "pressure in my chest", "chest tightness", "tight chest",
    "chest feels tight", "elephant on my chest",
    "impending doom", "something bad is about to happen", "something bad is going to happen",
    "something terrible is about to happen", "something terrible is going to happen",
    "passed out", "fainted", "seizure", "emergency room", "hospital", "911",
    "no puedo respirar", "emergencia", "dolor de pecho", "presión en el pecho", "algo malo va a pasar",
]
AE_TERMS = [
    "side effect", "side effects", "reaction", "allergic", "rash", "hives", "itching", "itchy",
    "swelling", "swollen", "fever", "chills", "infection", "nausea", "vomiting", "dizzy",
    "headache", "yellow skin", "yellow eyes", "dark urine", "bruise", "lump", "sore", "red",
    "sick", "bleeding", "pregnant",
    "efecto secundario", "sarpullido", "ronchas", "fiebre", "hinchazón", "mareo", "embarazada",
]
STOP_CALLING_TERMS = ["stop calling", "don't call me", "do not call", "remove me", "no me llamen"]

# Outbound: things the agent must never say, whatever the model produces.
DOSE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s?(mg|milligrams?|ml|mL|units?|doses?|shots?|injections?)\b", re.I)
OFF_LABEL_TERMS = ["crohn", "psoriasis", "arthritis", "lupus", "children", "pregnan", "off-label", "off label"]
ADVICE_PHRASES = ["you should take", "take two", "double up", "double dose",
                  "skip a dose", "stop taking", "it's safe to", "it is safe to", "you can stop"]

# ---------------------------------------------------------------------------
# Fixed scripts (would be MLR-approved content in a real deployment, never generated)
# ---------------------------------------------------------------------------

SCRIPTS = {
    "greeting": {
        "en": "Hi, this is Calder's automated patient support assistant. I'm an AI, and this call may be "
              "recorded. I'm calling for {first_name}. Before we talk, I just need to confirm your date of birth, "
              "to keep your information private. Could you share that with me?",
        "es": "Hola, soy el asistente automatizado de apoyo al paciente de Calder. Soy una IA y esta llamada "
              "puede ser grabada. Llamo para {first_name}. Antes de hablar, necesito confirmar su fecha de "
              "nacimiento para proteger su información. ¿Me la podría compartir?",
    },
    "urgent": {
        "en": "I'm worried about what you're describing, and I want you to get help right away. Please hang up "
              "and call 911, or go to the nearest emergency room now. I've recorded what you told me, and a "
              "Calder nurse will follow up with you.",
        "es": "Me preocupa lo que me describe y quiero que reciba ayuda de inmediato. Por favor cuelgue y llame "
              "al 911, o vaya a la sala de emergencias más cercana ahora. Registré lo que me dijo y una enfermera "
              "de Calder le dará seguimiento.",
    },
    "adverse_event": {
        "en": "I'm sorry you're dealing with that, and thank you for telling me. I'm not able to give medical "
              "advice, so I've passed what you shared to a Calder nurse, and they'll call you within 24 hours. "
              "They'll also see everything else we talked about today, so nothing gets lost. If it gets worse or "
              "ever feels like an emergency, please call 911, and you can always call your doctor too. I'm going "
              "to end our call now so the nurse can take it from here. Take care.",
        "es": "Siento que esté pasando por eso, y gracias por decírmelo. No puedo dar consejos médicos, así que "
              "le pasé lo que me contó a una enfermera de Calder, y le llamará dentro de 24 horas. También verá "
              "todo lo demás que hablamos hoy, para que nada se pierda. Si empeora o le parece una emergencia, "
              "llame al 911, y siempre puede llamar a su médico también. Voy a terminar la llamada para que la "
              "enfermera le atienda. Cuídese mucho.",
    },
    "medical_deferral": {
        "en": "That's a really good question, and you deserve a real answer, which has to come from your doctor "
              "or a Calder nurse. I've asked a nurse to call you back, and the Lumara medication guide has more "
              "information too. Is there anything else I can help with on getting your prescription?",
        "es": "Es una muy buena pregunta y merece una respuesta de verdad, que tiene que venir de su médico o de "
              "una enfermera de Calder. Pedí que una enfermera le llame, y la guía del medicamento Lumara también "
              "tiene más información. ¿Hay algo más en que pueda ayudarle con su receta?",
    },
    "stop_calling": {
        "en": "Of course, and I'm sorry for the interruption. I've noted that you don't want automated calls. "
              "You can still reach LumaraCare Monday to Friday during business hours whenever you need us. Take care.",
        "es": "Claro que sí, y disculpe la molestia. Anoté que no desea llamadas automatizadas. Puede comunicarse "
              "con LumaraCare de lunes a viernes en horario de oficina cuando lo necesite. Cuídese.",
    },
    "system_error": {
        "en": "I'm so sorry, I'm having a technical problem on my end. I've asked someone from the LumaraCare "
              "team to call you back. Take care.",
        "es": "Lo siento mucho, tengo un problema técnico. Pedí que alguien del equipo de LumaraCare le llame. Cuídese.",
    },
}

SYSTEM_PROMPT = """You are Calder's automated patient support assistant, an AI calling on behalf of LumaraCare, \
the patient support program for Lumara, a once-monthly self-injected treatment for ulcerative colitis. You are on \
an outbound phone call with {first_name}, whose Lumara prescription was approved but who has not received their \
first shipment.

Goal: find out what is getting in the way of their first fill and help clear it.
1. Before discussing anything about their prescription, verify identity with verify_identity using the date of birth they give.
2. Then check get_first_fill_status and explain it in plain words. Say the status is "as of" the date the tool gives, since it updates once a day.
3. Ask what is getting in the way. Common blockers: insurance, cost, scheduling delivery, nerves about self-injecting.
4. Offer a person when it helps, using schedule_callback: a pharmacy or insurance specialist, the copay team, or the injection-training nurse.

How to sound:
- The person you're talking to is living with a chronic illness and may be tired, anxious, or frustrated. Sound like a \
kind, patient person who has time for them, not a call-center script.
- When they share a worry or a frustration, acknowledge it warmly before moving on ("That's a really normal way to feel", \
"I'm sorry that's been confusing"). Never sound like you're correcting them or hurrying them along.
- Avoid clipped stock openers on their own, like "Got it", "Understood", or "That makes sense".
- Ask one question at a time. Use their name now and then, not every turn.
- Keep it short and plain-spoken: 1 to 3 sentences per turn, no lists or formatting. This is a phone call.

Boundaries:
- You give no medical advice: no dosing, missed-dose instructions, side effect guidance, or uses other than ulcerative colitis. \
For any of these, gently say a nurse or their doctor is the right person and offer a nurse callback.
- Never invent facts about the drug, insurance, costs, phone numbers, or dates. Only state what the tools return.
- Repeat timeframes exactly as the tool gives them ("about 2 business days, as of the pharmacy's November 1 update"). \
Never turn a timeframe into a date or a day of the week, and don't comment on how old the update is; you don't know \
today's date relative to it. If they need an exact date, offer the pharmacy and insurance team.
- Don't promise what other people will do or when: not when the pharmacy ships, when a callback comes, what a copay \
card covers, or what a team will be able to fix. Describe a team only by what it helps with (see schedule_callback), \
and say the team follows up during business hours.
- Only ever arrange contact with the patient. Never offer to call, check on, or reach out to anyone else. If someone \
else answers, share nothing about the prescription, and offer to book a callback for the patient or ask that the \
patient call LumaraCare back.
- If they ask, you are an AI.
- Reply entirely in the language of the patient's latest message (English or Spanish). Never mix languages in one reply.
"""

# Sent to the model when the outbound check stops its reply. It gets one rewrite, which is checked again.
REWRITE_NOTE = ("Your last reply was not sent because it included: {problems}. Write a new reply with no dose "
                "amounts, no other medical conditions, and no medical advice. If they asked a medical question, "
                "say a nurse or their doctor is the right person and offer a nurse callback.")

TOOLS = [
    {
        "name": "verify_identity",
        "description": "Check the patient's date of birth against the hub record. Required before any prescription details.",
        "input_schema": {
            "type": "object",
            "properties": {"dob": {"type": "string", "description": "Date of birth as YYYY-MM-DD"}},
            "required": ["dob"],
        },
    },
    {
        "name": "get_first_fill_status",
        "description": "Look up where the patient's first Lumara fill stands (prior auth, pharmacy, shipping). Only works after identity is verified.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "schedule_callback",
        "description": "Book a callback from a LumaraCare person during business hours, Monday to Friday. No exact "
                       "time can be promised. Teams: pharmacy_insurance helps with delivery details (scheduling, address), the pharmacy, "
                       "and insurance or benefits questions; copay helps with cost questions and the copay card; "
                       "injection_nurse helps with learning to self-inject and questions about the injection; "
                       "general is for anything else.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": ["pharmacy_insurance", "copay", "injection_nurse", "general"]},
                "preferred_time": {"type": "string", "description": "When the patient prefers, in their words"},
                "note": {"type": "string", "description": "One line on what they need, no medical details"},
            },
            "required": ["team"],
        },
    },
]

# ---------------------------------------------------------------------------
# Adverse event classifier (the second detector behind the keyword lists)
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """You screen messages from a patient on a pharma support call for possible adverse events, \
for pharmacovigilance. Flag anything that could be one: any symptom or health change, injection site problems, \
allergic reactions, hospital or ER visits, pregnancy, medication errors (missed, extra, or wrong doses), or the \
Lumara not working. When unsure, flag it. Missing an event is far worse than a false alarm. \
Past experiences with other medicines are not Lumara events. \
A general or hypothetical question ("what if I miss a dose?", "does it work for X?") is not an event by itself; \
flag it only if the patient says something actually happened to them. \
You may be shown the agent's last line for context, so you know what the patient is answering (for example a \
date of birth). Judge only the patient's words: never flag something because the agent mentioned it. \
Also report the language of the message ("en" or "es"). Urgent means possible anaphylaxis, trouble breathing, \
chest pain, pressure, or tightness, a sense of impending doom (a feeling that something bad is about to happen), \
swelling of the face, lips, or throat, fainting, or anything needing emergency care now."""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "possible_adverse_event": {"type": "boolean"},
        "urgent": {"type": "boolean"},
        "ae_probability": {"type": "number", "description": "0 to 1, how likely it is that this message describes a possible adverse event"},
        "evidence": {"type": "string", "description": "The words that triggered the flag, or empty"},
        "language": {"type": "string", "enum": ["en", "es"]},
    },
    "required": ["possible_adverse_event", "urgent", "ae_probability", "evidence", "language"],
    "additionalProperties": False,
}
