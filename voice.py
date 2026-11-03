"""
Voice front end for the Lumara agent.

Voice is only the input and output. Every patient turn still goes through
agent.handle_turn, so every safety gate runs exactly as in the text version:
    microphone -> ElevenLabs speech-to-text -> handle_turn -> ElevenLabs text-to-speech -> speaker

New failure mode to know about: the safety screen reads the transcript, not the audio,
so a transcription mistake can hide a side effect. The transcript is printed every turn.

Run:  python voice.py [PATIENT_CODE]
      Press Enter, talk, press Enter. Or type a message instead (works if the mic or network fails).
Needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env (key needs text_to_speech and speech_to_text).
"""

import io
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv

import agent
from chat import describe

ELEVENLABS_URL = "https://api.elevenlabs.io/v1"
TTS_MODEL = "eleven_multilingual_v2"  # one voice for English and Spanish
STT_MODEL = "scribe_v2"
SAMPLE_RATE = 16000

# Calm, even delivery. Defaults (stability 0.5) let a cloned voice swing in tone, which
# can come across as clipped or sassy on a support call.
VOICE_SETTINGS = {
    "stability": 0.75,         # higher = steadier tone between sentences
    "similarity_boost": 0.75,  # how closely to follow the cloned voice
    "style": 0.0,              # no exaggerated speaking style
    "speed": 0.92,             # a little slower than normal
    "use_speaker_boost": True,
}


def api_headers() -> dict:
    return {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}


def record_until_enter() -> bytes:
    """Record from the default mic until Enter is pressed. Returns WAV bytes."""
    chunks = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        callback=lambda data, frames, time, status: chunks.append(data.copy())):
        input()
    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="int16")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    return buffer.getvalue()


def transcribe(wav_bytes: bytes) -> str:
    # No language set, so ElevenLabs detects English or Spanish on its own.
    response = requests.post(
        f"{ELEVENLABS_URL}/speech-to-text", headers=api_headers(), timeout=30,
        data={"model_id": STT_MODEL}, files={"file": ("turn.wav", wav_bytes, "audio/wav")})
    response.raise_for_status()
    return response.json()["text"].strip()


def speak(text: str) -> None:
    """Say it out loud. If voice fails, the text is already on screen, so the call goes on."""
    try:
        response = requests.post(
            f"{ELEVENLABS_URL}/text-to-speech/{os.environ['ELEVENLABS_VOICE_ID']}",
            headers=api_headers(), json={"text": text, "model_id": TTS_MODEL, "voice_settings": VOICE_SETTINGS}, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(response.content)
            f.flush()
            subprocess.run(["afplay", f.name], check=True)
    except Exception as e:
        print(f"   [voice unavailable, text only: {type(e).__name__}]")


def get_patient_turn() -> str:
    """Enter on an empty line records speech; anything typed is used as-is."""
    typed = input("Press Enter to talk, or type: ").strip()
    if typed:
        return typed
    print("   [listening... press Enter when done]")
    wav = record_until_enter()
    try:
        return transcribe(wav)
    except Exception as e:
        print(f"   [couldn't transcribe ({type(e).__name__}), please type instead]")
        return ""


def main() -> None:
    load_dotenv()
    patient_code = sys.argv[1] if len(sys.argv) > 1 else "P-1001"
    session, opening = agent.start_call(patient_code)
    if session is None:
        print(f"[not calling] {opening}")
        return
    print(f"[voice call {session.call_id} to {patient_code}. Say or type 'quit' to hang up.]")
    print(f"[demo only: you are {session.patient['first_name']}, DOB {session.patient['dob']}]\n")
    print(f"Agent: {opening}\n")
    speak(opening)

    while not session.ended:
        try:
            text = get_patient_turn()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        print(f"Patient: {text}")  # what the safety screen will actually read
        if text.lower().strip(" .!") in {"quit", "exit", "bye", "goodbye"}:
            break
        result = agent.handle_turn(session, text)
        for event in result.events:
            print(f"   [{describe(event)}]")
        print(f"\nAgent: {result.reply}\n")
        speak(result.reply)
    print("[call ended]")


if __name__ == "__main__":
    main()
