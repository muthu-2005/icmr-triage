#!/usr/bin/env python3
"""
OpdDoc MedGemma — AI Medical Triage Assistant
==============================================

Production pipeline with two output modes:
  1. Doctor Assessment  → SOAP note (clinical language)
  2. Patient Summary    → 6-section plain-English explanation (≤250 chars each)

Architecture:
  Flask + SocketIO real-time consultation flow:
    start_consultation → request_questions → submit_answer (×3-5) →
    generate_soap | generate_patient_summary → soap_generated event

Voice Pipeline:
  Whisper Medium GGML (Vulkan GPU via whisper-cli.exe) → Tamil/English STT
  VAD (webrtcvad) → speech-frame detection + silence trimming
  ffmpeg (C:\ffmpeg) → audio format conversion (webm → 16kHz mono WAV)
  TranslateGemma 4B GGUF (llama-server port 8080) → Tamil↔English translation

GPU Backend:
  llama-server (llama.cpp) running MedGemma Q4_K_M via Vulkan on AMD Radeon 860M

  Start BOTH servers before running this app:
    Terminal 1 — TranslateGemma (translation):
      E:\llama-gpu\llama-server.exe -m E:\llama-gpu\translategemma-4b-it.Q4_K_M.gguf -ngl 34 --host 127.0.0.1 --port 8080 --ctx-size 4096
    Terminal 2 — MedGemma (medical AI):
      E:\llama-gpu\llama-server.exe -m E:\llama-gpu\medgemma-4b-Q4_K_M.gguf -ngl 34 --host 127.0.0.1 --port 8081 --ctx-size 4096
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT & IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CHROMA_TELEMETRY"] = "False"

import gc
import re
import sys
import uuid
import wave
import struct
import socket
import tempfile
import subprocess
import requests
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import math
import chromadb
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration — single place to tune all parameters."""

    # Server
    HOST = "0.0.0.0"
    PORT = 5010
    SECRET_KEY = "opddoc-medgemma-2026"

    # Model
    MODEL_ID = "google/medgemma-4b-it"

    # Llama GPU Server — MedGemma (medical AI)
    LLAMA_SERVER_URL = "http://127.0.0.1:8081"
    LLAMA_TIMEOUT = 300  # seconds

    # Llama GPU Server — TranslateGemma (Tamil ↔ English translation)
    TRANSLATE_SERVER_URL = "http://127.0.0.1:8080"
    TRANSLATE_TIMEOUT = 120  # seconds
    TRANSLATE_MAX_TOKENS = 512

    # ── Whisper Vulkan GPU ─────────────────────────────────────────────────
    WHISPER_EXE = r"E:\whisper-vulkan\whisper-cli.exe"
    MODEL       = r"E:\whisper-vulkan\ggml-medium.bin"
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

    # Whisper CLI options
    WHISPER_THREADS = 4
    WHISPER_LANGUAGE = "auto"          # auto-detect Tamil/English

    # VAD settings (energy-based, pure Python — no C build tools needed)
    VAD_FRAME_MS = 30                  # frame duration in ms
    VAD_SAMPLE_RATE = 16000            # 16kHz for Whisper
    VAD_PADDING_FRAMES = 10            # speech-boundary padding (frames)
    VAD_MIN_SPEECH_FRAMES = 15         # minimum speech frames to keep (~450ms)
    VAD_ENERGY_THRESHOLD = 0.015       # RMS threshold relative to max possible (auto-tuned)
    VAD_DYNAMIC_FACTOR = 1.5           # multiplier above noise floor for speech detection

    # RAG
    CHROMA_PATH = r"E:\triage-engine\chroma-db"
    CHROMA_COLLECTION = "icmr_dataset_collection"
    EMBEDDER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RAG_TOP_K = 5
    RAG_MAX_TOKENS_QUESTIONS = 350
    RAG_MAX_TOKENS_PATIENT = 200

    # ── Generation — SOAP (doctor) ────────────────────────────────────────────
    SOAP_MAX_TOKENS = 900
    SOAP_DO_SAMPLE = False
    SOAP_REP_PENALTY = 1.3

    # ── Generation — Patient summary ──────────────────────────────────────────
    PATIENT_MAX_TOKENS = 500
    PATIENT_DO_SAMPLE = True
    PATIENT_TEMPERATURE = 0.3
    PATIENT_TOP_P = 0.9
    PATIENT_REP_PENALTY = 1.5

    # ── Generation — Patient retry (pass 2) ───────────────────────────────────
    RETRY_TEMPERATURE = 0.4
    RETRY_REP_PENALTY = 1.6

    # ── Output limits ─────────────────────────────────────────────────────────
    MAX_SECTION_CHARS = 300
    MIN_MODEL_SECTIONS = 2

    # Input limits
    MAX_PERSONA_CHARS = 500
    MAX_SYMPTOMS_CHARS = 1000

    # Follow-up questions
    MIN_QUESTIONS = 3
    MAX_QUESTIONS = 5


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def squash_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def clean_markdown(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"(.)\1{5,}", "", text)
    text = re.sub(r"(\b\w\b\s+){10,}", "", text)
    out_lines = []
    for line in text.split("\n"):
        words = line.strip().split()
        if len(words) > 5:
            top = Counter(words).most_common(1)[0][1]
            if top / len(words) > 0.6:
                continue
        out_lines.append(line)
    return "\n".join(out_lines).strip().rstrip(".,;: \n")


def cap_text(text: str, limit: int = Config.MAX_SECTION_CHARS) -> str:
    text = squash_ws(text)
    if len(text) <= limit:
        return text
    trunc = text[:limit]
    for sep in [". ", "! ", "? "]:
        idx = trunc.rfind(sep)
        if idx > limit // 2:
            return trunc[: idx + 1].strip()
    idx = trunc.rfind(" ")
    if idx > limit // 2:
        return trunc[:idx].strip() + "."
    return trunc.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RAG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RagHit:
    rank: int
    distance: float
    metadata: Dict[str, Any]
    text: str


def rag_retrieve(collection, embedder, query: str, k: int) -> List[RagHit]:
    q_emb = embedder.encode([squash_ws(query)], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb, n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return [
        RagHit(rank=i + 1, distance=float(dists[i]),
               metadata=metas[i] or {}, text=docs[i] or "")
        for i in range(min(len(docs), len(metas), len(dists)))
    ]


def format_rag_context(hits: List[RagHit]) -> Tuple[str, List[Dict[str, Any]]]:
    sources, blocks = [], []
    for h in hits:
        meta = h.metadata or {}
        src = (meta.get("source_file") or meta.get("file_name")
               or meta.get("source") or meta.get("file") or "")
        sources.append({
            "rank": h.rank, "distance": h.distance,
            "file_name": meta.get("file_name") or (Path(str(src)).name if src else ""),
            "page": meta.get("page"), "source_file": src,
            "chunk": meta.get("chunk"),
        })
        hdr = f"[RAG {h.rank}] {Path(str(src)).name if src else 'source'}"
        if meta.get("page") is not None:
            hdr += f" p.{meta['page']}"
        blocks.append(f"{hdr}\n{h.text}".strip())
    return "\n\n".join(blocks).strip(), sources


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    if not text.strip():
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def apply_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )


def generate_text(
    chat_text: str, *,
    max_new_tokens: int, do_sample: bool = False,
    temperature: float = 0.0, top_p: float = 1.0,
    forbid_pad: bool = True, repetition_penalty: float = 1.3,
) -> Tuple[str, str]:
    """Call llama-server GPU API and return (clean_text, raw_text)."""
    payload = {
        "model": "medgemma",
        "messages": [{"role": "user", "content": chat_text}],
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature) if do_sample else 0.0,
        "top_p": float(top_p) if do_sample else 1.0,
        "repeat_penalty": float(repetition_penalty),
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{Config.LLAMA_SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=Config.LLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text, text
    except Exception as e:
        log(f"LLaMA server error: {e}")
        return "", ""


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_followup_prompt(persona: str, symptoms: str, context: str) -> str:
   return f"""You are a clinical triage assistant.

Your task is to ask between 3 and 5 follow-up questions to better understand
severity, pain level, red-flag danger signs, and any chronic complexity for safe triage.

Use the following rule:
- If symptoms sound clearly mild and low-risk → ask 3 questions.
- If there is moderate pain, impact on function, or some risk factors → ask 4 questions.
- If there is severe pain, red-flag–like features, or important chronic comorbidities → ask 5 questions.

Guidelines for questions:
- Each question must be clinically meaningful and non-redundant.
- Prioritise questions that help distinguish between:
  * mild/self-limited illness that can be observed at home
  * illness needing routine in-person review
  * illness needing urgent / emergency evaluation
- Ask only ONE question per line.
- Do NOT include bullets or words like "Question 1:". Use numeric prefixes only.
- Always end each line with a '?'.

Output format:
- 3 to 5 lines.
- Each line: "<number>. <question text>?"
- Example:
  1. How many days have you had the fever?
  2. Are you able to drink and keep fluids down?
  3. Do you have any difficulty breathing?

Patient persona (may contain typos):

{persona}

Patient symptoms (may contain typos):

{symptoms}

Guideline excerpt:

{context}

Now output ONLY the numbered questions, one per line, no extra commentary.
"""


def parse_followup_questions(text: str, min_q: int = 3, max_q: int = 5) -> List[str]:
    lines = [squash_ws(l) for l in (text or "").splitlines()]
    qs: List[str] = []
    for ln in lines:
        if not ln:
            continue
        ln = re.sub(r"^(q\s*\d+[:.)-]\s*|\d+[:.)-]\s*)", "", ln, flags=re.I).strip()
        if not ln:
            continue
        if "?" in ln:
            ln = ln.split("?")[0].strip() + "?"
        else:
            ln = ln.strip()
            if not ln.endswith("?"):
                ln += "?"
        if ln not in qs:
            qs.append(ln)
        if len(qs) >= max_q:
            break
    if len(qs) < min_q:
        fallback = [
            "How long have these symptoms been going on?",
            "Are you able to drink fluids and pass urine normally?",
            "Do you have any severe pain, blood, trouble breathing, or high fever?",
            "Do you have any chronic medical conditions or are you pregnant?",
            "Have your symptoms been getting better, worse, or staying the same?",
        ]
        while len(qs) < min_q and len(qs) < max_q:
            cand = fallback[len(qs) % len(fallback)]
            if cand not in qs:
                qs.append(cand if cand.endswith("?") else cand + "?")
    return qs[:max_q]


def build_soap_prompt(persona: str, symptoms: str,
                      followup_block: str, context: str) -> str:
    return f"""You are a clinical AI assistant. Based on this case, write a detailed medical assessment.

PATIENT:
{persona}

CHIEF COMPLAINT:
{symptoms}

FOLLOW-UP DETAILS:
{followup_block}

CLINICAL GUIDELINE REFERENCE (may contain multiple conditions, some unrelated):
{context}

CRITICAL INSTRUCTIONS:
- Use ONLY the parts of the guideline that are clearly relevant to THIS patient's current complaint.
- Completely ignore any sections about other diseases, body parts, or treatments that do not match this case.
- If the guideline discusses varicose veins, chronic venous insufficiency, or compression stockings and the patient has no leg symptoms, IGNORE those sections entirely and DO NOT mention them in the assessment or plan.

TASK:
Write a SOAP note tailored to this specific patient.

Use this exact format:

S: [3–5 sentences summarizing the patient's symptoms and follow-up answers in clear medical language.]

O: [State that this is remote triage with no vitals or physical examination. Note any relevant comorbidities or risk factors if mentioned.]

A: [200–300 words. Name the most likely diagnosis explicitly (e.g., "acne vulgaris", "upper respiratory tract infection", etc.). Explain why it fits this presentation. Mention 1–3 key differentials and why they are more or less likely. State that this is a working clinical impression, not a confirmed diagnosis.]

P: [200–300 words. Provide a safe, guideline-consistent plan focused ONLY on the current complaint. Include home care, medications if appropriate, when to seek in-person review, and specific danger signs that should trigger urgent/emergency care. Do NOT mention leg procedures or venous interventions unless the current complaint involves the legs or veins.]

RED_FLAGS: [Yes or No]

CONFIDENCE: [decimal between 0.7 and 0.9]

Write naturally and directly. Do not use markdown formatting like ** or ##.
"""


def parse_soap(narrative: str, persona: str, symptoms: str,
               followup_block: str) -> Dict[str, Any]:

    def _extract(patterns: List[str], text: str) -> str:
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.I)
            if m:
                return clean_markdown(squash_ws(m.group(1)))
        return ""

    s = _extract([r"S:\s*(.*?)(?=\nO:|$)",
                   r"SUBJECTIVE:\s*(.*?)(?=\nOBJECTIVE:|$)"], narrative)
    o = _extract([r"O:\s*(.*?)(?=\nA:|$)",
                   r"OBJECTIVE:\s*(.*?)(?=\nASSESSMENT:|$)"], narrative)
    a = _extract([r"A:\s*(.*?)(?=\nP:|$)",
                   r"ASSESSMENT:\s*(.*?)(?=\nPLAN:|$)"], narrative)
    p = _extract([r"P:\s*(.*?)(?=\nRED_FLAGS:|RED FLAGS:|CONFIDENCE:|$)",
                   r"PLAN:\s*(.*?)(?=\nRED_FLAGS:|RED FLAGS:|CONFIDENCE:|$)"], narrative)

    rf_m = re.search(r"RED[_ ]FLAGS?:\s*(Yes|No)", narrative, re.I)
    red_flags = bool(rf_m and rf_m.group(1).lower() == "yes")

    conf = 0.75
    conf_m = re.search(r"CONFIDENCE:\s*(0\.\d+|1\.0|\d+%)", narrative, re.I)
    if conf_m:
        try:
            c = float(conf_m.group(1).replace("%", ""))
            conf = c / 100.0 if c > 1.0 else c
        except Exception:
            pass

    if not s:
        s = f"Patient reports: {symptoms}. Follow-up: {followup_block}"
    if not o:
        o = ("Remote triage: no vitals or physical exam available. "
             "Patient demographics and history as reported.")
    if not a:
        a = ("The clinical presentation suggests a condition requiring in-person "
             "evaluation. Several differential diagnoses are possible and formal "
             "examination with appropriate investigations is recommended.")
    if not p:
        p = ("Seek in-person medical evaluation for complete assessment. "
             "Follow general supportive care and monitor for worsening. "
             "Seek urgent care for severe pain, high fever, difficulty "
             "breathing, or rapidly worsening symptoms.")

    return {
        "S": s, "O": o, "A": a, "P": p,
        "red_flags": red_flags,
        "confidence": max(conf, 0.70),
        "references": {},
    }


def build_patient_prompt(persona: str, symptoms: str,
                         followup_block: str, context: str) -> str:
    return f"""You are a friendly doctor explaining a patient's condition simply.

PATIENT: {persona}
SYMPTOMS: {symptoms}
FOLLOW-UP Q&A:
{followup_block}
REFERENCE: {context}

RULES:
1. Simple everyday English. No jargon.
2. No markdown. No ** or ## or bullets. Plain sentences only.
3. Each section: exactly 2-3 short sentences. NEVER write "Not Available".
4. If reference lacks treatment info, give general advice (rest, fluids, see doctor).
5. Stop after [URGENCY].

FORMAT:

[DIAGNOSIS]
2-3 sentences about what the problem likely is.

[FINDINGS]
2-3 sentences about which symptoms point to this.

[HOW_FOUND]
2-3 sentences about how this was figured out.

[TREATMENT]
2-3 sentences about home care and when to see a doctor.

[RECOVERY]
2-3 sentences about how long recovery takes.

[URGENCY]
One word: Low, Moderate, or High. Then 1-2 sentences.
"""


def build_patient_retry_prompt(persona: str, symptoms: str,
                                followup_block: str) -> str:
    return f"""A patient needs a simple health explanation.

Patient: {persona}
Symptoms: {symptoms}
Answers: {followup_block}

Write 6 sections. Each: 2-3 plain sentences. No markdown. No bullets.

[DIAGNOSIS]
What is likely wrong.

[FINDINGS]
What symptoms suggest this.

[HOW_FOUND]
How you figured this out.

[TREATMENT]
What to do at home and when to see a doctor.

[RECOVERY]
How long to feel better.

[URGENCY]
Low, Moderate, or High. When to get urgent help.
"""


_PATIENT_SECTION_PATTERNS = [
    ("diagnosis", [r"\[DIAGNOSIS\]", r"Diagnosis\s*:", r"\*\*Diagnosis\*\*", r"^DIAGNOSIS\b"]),
    ("findings",  [r"\[FINDINGS\]",  r"Findings\s*:",  r"\*\*Findings\*\*",  r"^FINDINGS\b"]),
    ("how_found", [r"\[HOW_FOUND\]", r"How It Was Found\s*:", r"How it was found\s*:",
                   r"\*\*How It Was Found\*\*", r"^HOW.?IT.?WAS.?FOUND\b"]),
    ("treatment", [r"\[TREATMENT\]", r"Treatment Option\s*:", r"Treatment\s*:",
                   r"\*\*Treatment(?:\s*Option)?\*\*", r"^TREATMENT\b"]),
    ("recovery",  [r"\[RECOVERY\]",  r"Recovery Period\s*:", r"Recovery\s*:",
                   r"\*\*Recovery(?:\s*Period)?\*\*", r"^RECOVERY\b"]),
    ("urgency",   [r"\[URGENCY\]",   r"Urgency Level\s*:", r"Urgency\s*:",
                   r"\*\*Urgency(?:\s*Level)?\*\*", r"^URGENCY\b"]),
]

_NA_RE = re.compile(r"^(not available|n/?a|none|not specified|nil|-)$", re.I)

_FALLBACK_PREFIXES = [
    "Based on your symptoms of",
    "You reported",
    "This was determined",
    "Rest well and stay",
    "Most conditions with",
    "Moderate. See a doctor",
]


def _build_fallbacks(symptoms: str) -> Dict[str, str]:
    s = squash_ws(symptoms) if symptoms else "your symptoms"
    return {
        "diagnosis": cap_text(
            f"Based on your symptoms of {s}, you may have a condition that needs "
            f"further evaluation. This is a preliminary assessment and we recommend "
            f"seeing a doctor for a confirmed diagnosis and proper treatment plan."
        ),
        "findings": cap_text(
            f"You reported {s} as your main concern. These symptoms suggest your body "
            f"is responding to an underlying condition. The details from your follow-up "
            f"answers helped us better understand your current health situation."
        ),
        "how_found": cap_text(
            f"This was determined by reviewing your reported symptoms of {s} along with "
            f"your follow-up answers about severity and duration. Combining this "
            f"information with known medical patterns helped form this assessment."
        ),
        "treatment": cap_text(
            "Rest well and stay hydrated with water and clear fluids. You may take "
            "paracetamol for fever or pain if needed. Avoid self-medicating with "
            "antibiotics. See a doctor if symptoms worsen or do not improve in a few days."
        ),
        "recovery": cap_text(
            f"Most conditions with symptoms like {s} improve within a few days to one "
            f"week with proper rest and hydration. Monitor your progress daily and "
            f"consult a doctor for a specific timeline after examination."
        ),
        "urgency": cap_text(
            "Moderate. See a doctor within a day or two. Seek emergency care immediately "
            "if you have difficulty breathing, severe chest pain, high fever above 103F "
            "that does not respond to medicine, or sudden worsening of symptoms."
        ),
    }


def parse_patient_narrative(text: str, persona: str = "",
                            symptoms: str = "",
                            followup_block: str = "") -> Dict[str, str]:
    positions: List[Tuple[int, int, str]] = []
    for key, patterns in _PATIENT_SECTION_PATTERNS:
        for pat in patterns:
            m = re.search(pat, text, re.MULTILINE | re.IGNORECASE)
            if m:
                positions.append((m.start(), m.end(), key))
                break
    positions.sort(key=lambda x: x[0])

    result: Dict[str, str] = {}
    for i, (_, hdr_end, key) in enumerate(positions):
        nxt = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        raw = clean_markdown(text[hdr_end:nxt].strip())
        result[key] = re.sub(r"^[\s:;\-]+", "", raw).strip()

    fallbacks = _build_fallbacks(symptoms)
    for key in ["diagnosis", "findings", "how_found", "treatment", "recovery", "urgency"]:
        val = result.get(key, "")
        is_bad = (
            len(val) < 20
            or bool(_NA_RE.match(val.strip()))
            or (len(val) > 200 and len(set(val.lower().split())) < 15)
        )
        result[key] = fallbacks[key] if is_bad else cap_text(clean_markdown(val))

    return result


def count_model_sections(parsed: Dict[str, str]) -> int:
    return sum(
        1 for v in parsed.values()
        if not any(v.startswith(p) for p in _FALLBACK_PREFIXES) and len(v) >= 30
    )


def post_clean(text: str) -> str:
    for marker in ("\n\n\n", "---", "===", "Note:", "Disclaimer:", "Additional"):
        idx = text.find(marker)
        if idx > 100:
            return text[:idx].strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# 7. VOICE PIPELINE — VAD + WHISPER VULKAN GPU
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_to_wav16k(input_path: str, output_path: str) -> bool:
    """Convert any audio file to 16kHz mono 16-bit PCM WAV using ffmpeg."""
    try:
        cmd = [
            Config.FFMPEG_PATH,
            "-y",                     # overwrite output
            "-i", input_path,         # input file
            "-ar", "16000",           # 16kHz sample rate
            "-ac", "1",               # mono
            "-sample_fmt", "s16",     # 16-bit signed int
            "-f", "wav",              # output format
            output_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            log(f"ffmpeg error: {result.stderr.decode(errors='replace')[:300]}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 100
    except Exception as e:
        log(f"ffmpeg conversion failed: {e}")
        return False


def _read_wav_frames(wav_path: str, frame_ms: int = 30) -> Tuple[bytes, int, List[bytes]]:
    """Read a 16kHz mono WAV and split into frames for VAD."""
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1, "Must be mono"
        assert wf.getsampwidth() == 2, "Must be 16-bit"
        sample_rate = wf.getframerate()
        raw_data = wf.readframes(wf.getnframes())

    frame_size = int(sample_rate * frame_ms / 1000) * 2  # 2 bytes per sample (16-bit)
    frames = []
    for i in range(0, len(raw_data) - frame_size + 1, frame_size):
        frames.append(raw_data[i : i + frame_size])

    return raw_data, sample_rate, frames


def _frame_rms(frame_bytes: bytes) -> float:
    """Calculate RMS energy of a 16-bit PCM audio frame."""
    n_samples = len(frame_bytes) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", frame_bytes)
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / n_samples)


def _vad_filter_speech(wav_path: str) -> str:
    """
    Energy-based Voice Activity Detection (pure Python, no C build tools).
    Measures RMS energy per frame, auto-tunes threshold from the quietest
    10% of frames (noise floor), keeps speech frames + padding.
    Returns path to a new WAV with silence removed.
    """
    try:
        raw_data, sample_rate, frames = _read_wav_frames(wav_path, Config.VAD_FRAME_MS)

        if not frames:
            log("VAD: no frames to process, using original audio")
            return wav_path

        # Calculate RMS energy for each frame
        energies = [_frame_rms(f) for f in frames]

        # Auto-tune threshold: use the quietest 10% as noise floor
        sorted_energies = sorted(energies)
        noise_count = max(1, len(sorted_energies) // 10)
        noise_floor = sum(sorted_energies[:noise_count]) / noise_count

        # Speech threshold = noise_floor × dynamic_factor, with a minimum absolute floor
        # The absolute floor (500) handles near-silent recordings where noise_floor ≈ 0
        threshold = max(
            noise_floor * Config.VAD_DYNAMIC_FACTOR,
            500.0  # absolute minimum RMS for 16-bit audio (~1.5% of max)
        )

        log(f"VAD: noise_floor={noise_floor:.0f} threshold={threshold:.0f} "
            f"max_energy={max(energies):.0f}")

        # Classify each frame as speech or silence
        speech_flags = [e >= threshold for e in energies]

        # Ring buffer: pad speech regions to avoid clipping word boundaries
        padded = list(speech_flags)
        padding = Config.VAD_PADDING_FRAMES
        for i, flag in enumerate(speech_flags):
            if flag:
                for j in range(max(0, i - padding), min(len(padded), i + padding + 1)):
                    padded[j] = True

        # Collect speech frames
        speech_frames = [f for f, keep in zip(frames, padded) if keep]
        speech_count = len(speech_frames)
        total_count = len(frames)

        log(f"VAD: {speech_count}/{total_count} frames contain speech "
            f"({100 * speech_count / max(total_count, 1):.0f}%)")

        # If too few speech frames, use original audio (might be very short speech)
        if speech_count < Config.VAD_MIN_SPEECH_FRAMES:
            log("VAD: too few speech frames, using original audio")
            return wav_path

        # If almost all frames are speech, skip rewrite
        if speech_count >= total_count * 0.9:
            log("VAD: most frames are speech, using original audio")
            return wav_path

        # Write filtered WAV
        vad_out = wav_path.replace(".wav", "_vad.wav")
        with wave.open(vad_out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(speech_frames))

        log(f"VAD: trimmed audio written → {os.path.getsize(vad_out)} bytes")
        return vad_out

    except Exception as e:
        log(f"VAD processing error: {e}")
        return wav_path  # fallback to original


def _detect_language_from_output(text: str) -> str:
    """Heuristic language detection from Whisper output text."""
    if not text.strip():
        return "en"
    # Count Tamil Unicode characters (range: U+0B80 – U+0BFF)
    tamil_chars = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
    total_alpha = sum(1 for ch in text if ch.isalpha())
    if total_alpha == 0:
        return "en"
    tamil_ratio = tamil_chars / total_alpha
    return "ta" if tamil_ratio > 0.3 else "en"


def whisper_transcribe(audio_path: str) -> Tuple[str, str]:
    """
    Transcribe audio using whisper-cli.exe (Vulkan GPU).
    Returns (text, language).
    """
    tmp_dir = tempfile.gettempdir()
    wav_path = os.path.join(tmp_dir, f"whisper_{uuid.uuid4().hex[:8]}.wav")
    vad_path = None

    try:
        # Step 1: Convert to 16kHz mono WAV
        if not _convert_to_wav16k(audio_path, wav_path):
            log("Whisper: ffmpeg conversion failed")
            return "", "en"

        wav_size = os.path.getsize(wav_path)
        log(f"Whisper: WAV ready ({wav_size} bytes)")

        # Step 2: VAD — trim silence, keep speech frames only
        vad_path = _vad_filter_speech(wav_path)

        # Step 3: Run whisper-cli.exe (Vulkan GPU)
        cmd = [
            Config.WHISPER_EXE,
            "-m", Config.MODEL,
            "-f", vad_path,
            "-t", str(Config.WHISPER_THREADS),
            "-l", "auto",
            "--no-timestamps",
            "--print-progress", "false",
        ]

        log(f"Whisper: running → {Path(Config.WHISPER_EXE).name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            cwd=str(Path(Config.WHISPER_EXE).parent),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            log(f"Whisper error (rc={result.returncode}): {stderr[:300]}")
            return "", "en"

        # Step 4: Parse stdout — decode as UTF-8 (handles Tamil output)
        raw_output = (result.stdout or b"").decode("utf-8", errors="replace").strip()

        # whisper-cli prints some info lines before the actual text;
        # the transcription is typically after the last empty line or
        # after lines starting with "whisper_" or "system_info"
        lines = raw_output.split("\n")
        text_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip whisper info/debug lines
            if not stripped:
                continue
            if any(stripped.lower().startswith(p) for p in [
                "whisper_", "system_info", "main:", "log_mel",
                "processing", "output_", "encode", "decode",
                "sampling", "beam_", "translate"
            ]):
                continue
            # Skip timestamp lines like [00:00:00.000 --> 00:00:05.000]
            if re.match(r"^\[?\d{2}:\d{2}[:\.]", stripped):
                continue
            text_lines.append(stripped)

        text = " ".join(text_lines).strip()
        # Clean up common artifacts
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^\[.*?\]\s*", "", text)  # remove leading [timestamp]
        text = text.strip()

        # Step 5: Detect language from the transcribed text
        language = _detect_language_from_output(text)

        log(f"Whisper result: lang={language} | chars={len(text)} "
            f"| preview=\"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        return text, language

    except subprocess.TimeoutExpired:
        log("Whisper: process timed out (120s)")
        return "", "en"
    except Exception as e:
        log(f"Whisper transcription error: {e}")
        import traceback
        traceback.print_exc()
        return "", "en"
    finally:
        # Cleanup temp files
        for f in [wav_path, vad_path]:
            if f and os.path.exists(f) and f != audio_path:
                try:
                    os.unlink(f)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════════
# 8. APPLICATION STATE
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

STATE: Dict[str, Any] = {
    "tokenizer": None, "model": None, "device": None,
    "collection": None, "embedder": None,
    "whisper_ready": False,
    "models_loaded": False,
}

consultations: Dict[str, Dict[str, Any]] = {}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_models() -> bool:
    if STATE["models_loaded"]:
        return True
    try:
        log("=" * 60)
        log("INITIALIZING OPDDOC MEDGEMMA (GPU via llama-server)")

        # Test MedGemma llama-server connection (port 8081)
        resp = requests.get(f"{Config.LLAMA_SERVER_URL}/health", timeout=10)
        if resp.status_code != 200:
            raise Exception("MedGemma llama-server not responding on port 8081")
        log("✓ MedGemma llama-server connected (port 8081, AMD Radeon 860M via Vulkan)")

        # Test TranslateGemma llama-server connection (port 8080)
        resp = requests.get(f"{Config.TRANSLATE_SERVER_URL}/health", timeout=10)
        if resp.status_code != 200:
            raise Exception("TranslateGemma llama-server not responding on port 8080")
        log("✓ TranslateGemma llama-server connected (port 8080)")

        # Whisper Vulkan GPU — validate paths
        whisper_ok = True
        if not os.path.isfile(Config.WHISPER_EXE):
            log(f"✗ Whisper exe not found: {Config.WHISPER_EXE}")
            whisper_ok = False
        if not os.path.isfile(Config.MODEL):
            log(f"✗ Whisper model not found: {Config.MODEL}")
            whisper_ok = False
        if not os.path.isfile(Config.FFMPEG_PATH):
            log(f"✗ ffmpeg not found: {Config.FFMPEG_PATH}")
            whisper_ok = False

        if whisper_ok:
            STATE["whisper_ready"] = True
            log(f"✓ Whisper Vulkan GPU ready (whisper-cli.exe + ggml-medium.bin)")
            log(f"  EXE:   {Config.WHISPER_EXE}")
            log(f"  Model: {Config.MODEL}")
            log(f"  ffmpeg: {Config.FFMPEG_PATH}")
            log(f"  VAD:   energy-based (pure Python, threshold auto-tuned)")
        else:
            log("⚠ Whisper NOT available — voice input will be disabled")

        # ChromaDB
        client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        STATE["collection"] = client.get_collection(name=Config.CHROMA_COLLECTION)
        log(f"✓ ChromaDB: {STATE['collection'].count()} chunks")

        # Embedder
        STATE["embedder"] = SentenceTransformer(Config.EMBEDDER_ID)
        log("✓ Embedder ready")

        # Tokenizer only — model runs in llama-server (no vision processor needed)
        STATE["tokenizer"] = AutoTokenizer.from_pretrained(
            Config.MODEL_ID, use_fast=True
        )
        log("✓ Tokenizer loaded (lightweight — no vision processor)")

        STATE["device"] = "vulkan"   # informational — actual compute is in llama-server
        STATE["model"] = None        # model runs in llama-server GPU process
        STATE["models_loaded"] = True

        # Free any transient memory used during initialization
        gc.collect()

        log("✓✓✓ ALL SYSTEMS READY — MedGemma + TranslateGemma + Whisper Vulkan ✓✓✓")
        log("=" * 60)
        return True

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 10. HTTP ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ready" if STATE["models_loaded"] else "loading",
        "backend_medgemma": "llama-server (AMD Vulkan GPU) port 8081",
        "backend_translate": "llama-server (AMD Vulkan GPU) port 8080",
        "whisper": "whisper-cli.exe (Vulkan GPU) + VAD",
        "device": "AMD Radeon 860M",
    })


@app.route("/start_consultation", methods=["POST"])
def start_consultation():
    data = request.json
    cid = str(uuid.uuid4())
    persona  = data.get("persona",  "").strip()[: Config.MAX_PERSONA_CHARS]
    symptoms = data.get("symptoms", "").strip()[: Config.MAX_SYMPTOMS_CHARS]

    if not persona or not symptoms:
        return jsonify({"error": "persona and symptoms required"}), 400

    consultations[cid] = {
        "persona": persona, "symptoms": symptoms,
        "rag_context": None, "rag_sources": [],
        "questions": [], "answers": [],
        "current_q_index": 0, "max_questions": Config.MAX_QUESTIONS,
    }
    return jsonify({"consultation_id": cid,
                    "max_questions": Config.MAX_QUESTIONS,
                    "message": "Consultation started"})


# ── 10b. Whisper Vulkan Speech-to-Text ─────────────────────────────────────

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribe audio using Whisper Vulkan GPU (whisper-cli.exe).
    Pipeline: save upload → ffmpeg → VAD trim → whisper-cli → parse text."""
    if not STATE.get("whisper_ready"):
        return jsonify({"error": "Whisper not available"}), 503

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    tmp = None
    try:
        # Determine file extension from content type
        suffix = ".webm"
        content_type = audio_file.content_type or ""
        if "wav" in content_type:
            suffix = ".wav"
        elif "mp4" in content_type or "m4a" in content_type:
            suffix = ".m4a"
        elif "ogg" in content_type:
            suffix = ".ogg"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        audio_file.save(tmp)
        tmp.close()

        file_size = os.path.getsize(tmp.name)
        log(f"Whisper transcribing: {tmp.name} ({suffix}, {file_size} bytes)")

        # Run the full pipeline: ffmpeg convert → VAD → whisper-cli
        text, language = whisper_transcribe(tmp.name)

        if not text:
            return jsonify({"text": "", "language": "en"})

        return jsonify({"text": text, "language": language})

    except Exception as e:
        log(f"Whisper error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


# ── 10c. TranslateGemma Translation ──────────────────────────────────────────

def _call_translate_server(prompt_text: str) -> str:
    """Call TranslateGemma llama-server on port 8080."""
    payload = {
        "model": "translategemma",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": Config.TRANSLATE_MAX_TOKENS,
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.2,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{Config.TRANSLATE_SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=Config.TRANSLATE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log(f"TranslateGemma error: {e}")
        return ""


@app.route("/translate", methods=["POST"])
def translate_text():
    """Translate text between Tamil and English using TranslateGemma."""
    data = request.json
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    text = (data.get("text") or "").strip()
    source = (data.get("source") or "").strip().lower()
    target = (data.get("target") or "").strip().lower()

    if not text:
        return jsonify({"error": "No text to translate"}), 400
    if source not in ("ta", "en") or target not in ("ta", "en"):
        return jsonify({"error": "source and target must be 'ta' or 'en'"}), 400
    if source == target:
        return jsonify({"translated_text": text})

    lang_names = {"ta": "Tamil", "en": "English"}
    prompt = (
        f"Translate the following {lang_names[source]} text to {lang_names[target]}. "
        f"Output ONLY the complete translation in {lang_names[target]} script, nothing else.\n\n{text}"
    )

    log(f"Translate: {source}→{target} | chars={len(text)} "
        f"| preview=\"{text[:60]}{'...' if len(text) > 60 else ''}\"")

    translated = _call_translate_server(prompt)

    if not translated:
        return jsonify({"error": "Translation failed"}), 500

    log(f"Translated: chars={len(translated)} "
        f"| preview=\"{translated[:60]}{'...' if len(translated) > 60 else ''}\"")

    return jsonify({"translated_text": translated})


@app.route("/translate_ui_batch", methods=["POST"])
def translate_ui_batch():
    """Batch-translate an array of UI text strings using TranslateGemma.
    Accepts: { "texts": ["Hello", "Start"], "source": "en", "target": "ta" }
    Returns: { "translations": ["வணக்கம்", "தொடங்கு"] }
    This endpoint is optional — the frontend uses a client-side dictionary by default
    and only falls back here for dynamic text not in the dictionary.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    texts = data.get("texts", [])
    source = (data.get("source") or "en").strip().lower()
    target = (data.get("target") or "ta").strip().lower()

    if not texts:
        return jsonify({"translations": []})
    if source == target:
        return jsonify({"translations": texts})

    lang_names = {"ta": "Tamil", "en": "English"}
    if source not in lang_names or target not in lang_names:
        return jsonify({"error": "source and target must be 'ta' or 'en'"}), 400

    # Translate each text (could batch into one prompt for efficiency)
    translations = []
    for t in texts:
        if not t.strip():
            translations.append(t)
            continue
        prompt = (
            f"Translate the following {lang_names[source]} text to {lang_names[target]}. "
            f"Output ONLY the translation, nothing else.\n\n{t}"
        )
        result = _call_translate_server(prompt)
        translations.append(result if result else t)

    return jsonify({"translations": translations})


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SOCKET EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _followup_block(c: dict) -> str:
    return "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(c["questions"], c["answers"]))
    )


# ── 11a. Follow-up Questions ──────────────────────────────────────────────────

@socketio.on("request_questions")
def on_request_questions(data):
    cid = data["consultation_id"]
    if cid not in consultations:
        emit("error", {"message": "Invalid consultation ID"}); return
    c = consultations[cid]

    log(f"[{cid[:8]}] ── Questions output generating...")

    query = squash_ws(f"{c['persona']} {c['symptoms']}")
    hits = rag_retrieve(STATE["collection"], STATE["embedder"], query, Config.RAG_TOP_K)
    ctx_full, src = format_rag_context(hits)
    ctx = truncate_to_tokens(ctx_full, STATE["tokenizer"],
                             Config.RAG_MAX_TOKENS_QUESTIONS)
    c["rag_context"] = ctx
    c["rag_sources"] = src

    prompt = build_followup_prompt(c["persona"], c["symptoms"], ctx)
    chat = apply_chat(STATE["tokenizer"], prompt)
    raw, _ = generate_text(
        chat,
        max_new_tokens=150, forbid_pad=True,
    )
    questions = parse_followup_questions(raw)
    c["questions"] = questions
    c["max_questions"] = len(questions)

    log(f"[{cid[:8]}] ── Questions generated | count={len(questions)} "
        f"| total_chars={sum(len(q) for q in questions)}")

    emit("questions_ready", {"questions": questions, "total": len(questions)})


# ── 11b. Answer Intake ────────────────────────────────────────────────────────

@socketio.on("submit_answer")
def on_submit_answer(data):
    cid = data["consultation_id"]
    if cid not in consultations:
        emit("error", {"message": "Invalid consultation ID"}); return
    c = consultations[cid]
    answer = data.get("answer", "").strip()
    c["answers"].append(answer)
    c["current_q_index"] += 1

    log(f"[{cid[:8]}] Answer {c['current_q_index']}/{c['max_questions']} "
        f"| chars={len(answer)} | preview=\"{answer[:60]}{'...' if len(answer) > 60 else ''}\"")

    if c["current_q_index"] >= c["max_questions"]:
        emit("all_answers_complete", {"message": "Ready to generate SOAP"})
    else:
        emit("answer_received", {"next_index": c["current_q_index"]})


# ── 11c. Doctor Assessment (SOAP) ────────────────────────────────────────────

@socketio.on("generate_soap")
def on_generate_soap(data):
    cid = data["consultation_id"]
    if cid not in consultations:
        emit("error", {"message": "Invalid consultation ID"}); return
    c = consultations[cid]
    fb = _followup_block(c)

    emit("soap_progress", {"message": "Generating clinical assessment..."})
    log(f"[{cid[:8]}] ── SOAP generating...")

    t0 = datetime.now()
    prompt = build_soap_prompt(c["persona"], c["symptoms"], fb, c["rag_context"])
    chat = apply_chat(STATE["tokenizer"], prompt)
    text, _ = generate_text(
        chat,
        max_new_tokens=Config.SOAP_MAX_TOKENS,
        do_sample=Config.SOAP_DO_SAMPLE,
        forbid_pad=True,
        repetition_penalty=Config.SOAP_REP_PENALTY,
    )
    parsed = parse_soap(text, c["persona"], c["symptoms"], fb)
    elapsed = (datetime.now() - t0).seconds

    log(f"[{cid[:8]}] ── SOAP generated | elapsed={elapsed}s"
        f" | confidence={parsed['confidence']:.0%}"
        f" | red_flags={parsed['red_flags']}")

    emit("soap_generated", {
        "reported_issue": parsed["S"],
        "key_findings":   parsed["O"],
        "soap": {
            "A": parsed["A"],
            "P": parsed["P"],
            "red_flags":  "Yes" if parsed["red_flags"] else "No",
            "confidence": f"{int(parsed['confidence'] * 100)}%",
        },
        "diagnosis": "", "findings": "", "how_found": "",
        "treatment": "", "recovery": "", "urgency": "",
    })


# ── 11d. Patient Summary (two-pass) ──────────────────────────────────────────

@socketio.on("generate_patient_summary")
def on_generate_patient_summary(data):
    cid = data["consultation_id"]
    if cid not in consultations:
        emit("error", {"message": "Invalid consultation ID"}); return
    c = consultations[cid]
    fb = _followup_block(c)

    emit("soap_progress", {"message": "Generating patient-friendly summary..."})
    log(f"[{cid[:8]}] ── Patient summary generating...")

    t0 = datetime.now()
    patient_ctx = truncate_to_tokens(
        c["rag_context"], STATE["tokenizer"],
        Config.RAG_MAX_TOKENS_PATIENT,
    )

    # Pass 1
    log(f"[{cid[:8]}]   PASS1 generating...")
    prompt = build_patient_prompt(c["persona"], c["symptoms"], fb, patient_ctx)
    chat = apply_chat(STATE["tokenizer"], prompt)
    raw1, _ = generate_text(
        chat,
        max_new_tokens=Config.PATIENT_MAX_TOKENS,
        do_sample=Config.PATIENT_DO_SAMPLE,
        temperature=Config.PATIENT_TEMPERATURE,
        top_p=Config.PATIENT_TOP_P,
        forbid_pad=True,
        repetition_penalty=Config.PATIENT_REP_PENALTY,
    )
    raw1 = post_clean(raw1)
    parsed = parse_patient_narrative(raw1, c["persona"], c["symptoms"], fb)
    n1 = count_model_sections(parsed)
    log(f"[{cid[:8]}]   PASS1 done | model_sections={n1}/6 | raw_chars={len(raw1)}")

    # Pass 2 (retry if too few model sections)
    if n1 < Config.MIN_MODEL_SECTIONS:
        log(f"[{cid[:8]}]   PASS2 generating (only {n1}/6 sections, retrying)...")
        retry_prompt = build_patient_retry_prompt(c["persona"], c["symptoms"], fb)
        retry_chat = apply_chat(STATE["tokenizer"], retry_prompt)
        raw2, _ = generate_text(
            retry_chat,
            max_new_tokens=Config.PATIENT_MAX_TOKENS,
            do_sample=True,
            temperature=Config.RETRY_TEMPERATURE,
            top_p=Config.PATIENT_TOP_P,
            forbid_pad=True,
            repetition_penalty=Config.RETRY_REP_PENALTY,
        )
        raw2 = post_clean(raw2)
        parsed2 = parse_patient_narrative(raw2, c["persona"], c["symptoms"], fb)
        n2 = count_model_sections(parsed2)
        log(f"[{cid[:8]}]   PASS2 done | model_sections={n2}/6 | raw_chars={len(raw2)}")
        if n2 > n1:
            parsed = parsed2
            log(f"[{cid[:8]}]   Using PASS2 result")
        else:
            log(f"[{cid[:8]}]   Keeping PASS1 result")

    elapsed = (datetime.now() - t0).seconds
    log(f"[{cid[:8]}] ── Patient summary done | elapsed={elapsed}s")
   

    emit("soap_generated", {
        "reported_issue": c["symptoms"],
        "key_findings": "Remote triage — no vitals or physical exam available.",
        "soap": {"A": "", "P": "", "red_flags": "No", "confidence": "N/A"},
        "diagnosis": parsed["diagnosis"],
        "findings":  parsed["findings"],
        "how_found": parsed["how_found"],
        "treatment": parsed["treatment"],
        "recovery":  parsed["recovery"],
        "urgency":   parsed["urgency"],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 12. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("  OPDDOC MEDGEMMA — AI MEDICAL TRIAGE")
    print("  Backend: AMD Radeon 860M (Vulkan via llama-server)")
    print("  Voice:   Whisper Vulkan GPU (whisper-cli.exe) + VAD")
    print(f"{'=' * 60}\n")
    print("  Make sure BOTH llama-servers are running:\n")
    print("  Terminal 1 — TranslateGemma (translation):")
    print("  E:\\llama-gpu\\llama-server.exe -m E:\\llama-gpu\\translategemma-4b-it.Q4_K_M.gguf")
    print("  -ngl 34 --host 127.0.0.1 --port 8080 --ctx-size 4096\n")
    print("  Terminal 2 — MedGemma (medical AI):")
    print("  E:\\llama-gpu\\llama-server.exe -m E:\\llama-gpu\\medgemma-4b-Q4_K_M.gguf")
    print(f"  -ngl 34 --host 127.0.0.1 --port 8081 --ctx-size 4096\n")

    if not initialize_models():
        print("\n✗ INITIALIZATION FAILED")
        print("  Are both llama-servers running?")
        print("  Check http://127.0.0.1:8080/health (TranslateGemma)")
        print("  Check http://127.0.0.1:8081/health (MedGemma)")
        sys.exit(1)

    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n✓ SERVER READY!")
    print(f"✓ Local:  http://127.0.0.1:{Config.PORT}")
    print(f"✓ Mobile: http://{local_ip}:{Config.PORT}\n")

    socketio.run(app, host=Config.HOST, port=Config.PORT,
                 allow_unsafe_werkzeug=True, debug=False)