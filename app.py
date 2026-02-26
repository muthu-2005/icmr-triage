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

GPU Backend:
  llama-server (llama.cpp) running MedGemma Q4_K_M via Vulkan on AMD Radeon 860M
  Start server before running this app:
    E:\llama-gpu\llama-server.exe -m E:\llama-gpu\medgemma-4b-Q4_K_M.gguf -ngl 34 --host 127.0.0.1 --port 8080 --ctx-size 4096
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
import socket
import requests
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import chromadb
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor


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

    # Llama GPU Server
    LLAMA_SERVER_URL = "http://127.0.0.1:8080"
    LLAMA_TIMEOUT = 300  # seconds

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


def apply_chat(processor, user_text: str) -> str:
    return processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )


def generate_text(
    processor, model, dev, chat_text: str, *,
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
# 7. APPLICATION STATE
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

STATE: Dict[str, Any] = {
    "processor": None, "model": None, "device": None,
    "collection": None, "embedder": None, "models_loaded": False,
}

consultations: Dict[str, Dict[str, Any]] = {}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_models() -> bool:
    if STATE["models_loaded"]:
        return True
    try:
        log("=" * 60)
        log("INITIALIZING OPDDOC MEDGEMMA (GPU via llama-server)")

        # Test llama server connection
        resp = requests.get(f"{Config.LLAMA_SERVER_URL}/health", timeout=10)
        if resp.status_code != 200:
            raise Exception("llama-server not responding")
        log("✓ llama-server GPU backend connected (AMD Radeon 860M via Vulkan)")

        # ChromaDB
        client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        STATE["collection"] = client.get_collection(name=Config.CHROMA_COLLECTION)
        log(f"✓ ChromaDB: {STATE['collection'].count()} chunks")

        # Embedder
        STATE["embedder"] = SentenceTransformer(Config.EMBEDDER_ID)
        log("✓ Embedder ready")

        # Processor (tokenizer only — model runs in llama-server)
        STATE["processor"] = AutoProcessor.from_pretrained(
            Config.MODEL_ID, use_fast=True
        )
        log("✓ Processor/tokenizer loaded")

        STATE["device"] = torch.device("cpu")
        STATE["model"] = None  # model runs in llama-server GPU process
        STATE["models_loaded"] = True

        log("✓✓✓ ALL SYSTEMS READY — MedGemma running on AMD GPU ✓✓✓")
        log("=" * 60)
        return True

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HTTP ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ready" if STATE["models_loaded"] else "loading",
        "backend": "llama-server (AMD Vulkan GPU)",
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


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SOCKET EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _followup_block(c: dict) -> str:
    return "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(c["questions"], c["answers"]))
    )


# ── 10a. Follow-up Questions ──────────────────────────────────────────────────

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
    ctx = truncate_to_tokens(ctx_full, STATE["processor"].tokenizer,
                             Config.RAG_MAX_TOKENS_QUESTIONS)
    c["rag_context"] = ctx
    c["rag_sources"] = src

    prompt = build_followup_prompt(c["persona"], c["symptoms"], ctx)
    chat = apply_chat(STATE["processor"], prompt)
    raw, _ = generate_text(
        STATE["processor"], STATE["model"], STATE["device"], chat,
        max_new_tokens=150, forbid_pad=True,
    )
    questions = parse_followup_questions(raw)
    c["questions"] = questions
    c["max_questions"] = len(questions)

    log(f"[{cid[:8]}] ── Questions generated | count={len(questions)} "
        f"| total_chars={sum(len(q) for q in questions)}")

    emit("questions_ready", {"questions": questions, "total": len(questions)})


# ── 10b. Answer Intake ────────────────────────────────────────────────────────

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


# ── 10c. Doctor Assessment (SOAP) ────────────────────────────────────────────

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
    chat = apply_chat(STATE["processor"], prompt)
    text, _ = generate_text(
        STATE["processor"], STATE["model"], STATE["device"], chat,
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


# ── 10d. Patient Summary (two-pass) ──────────────────────────────────────────

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
        c["rag_context"], STATE["processor"].tokenizer,
        Config.RAG_MAX_TOKENS_PATIENT,
    )

    # Pass 1
    log(f"[{cid[:8]}]   PASS1 generating...")
    prompt = build_patient_prompt(c["persona"], c["symptoms"], fb, patient_ctx)
    chat = apply_chat(STATE["processor"], prompt)
    raw1, _ = generate_text(
        STATE["processor"], STATE["model"], STATE["device"], chat,
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
        retry_chat = apply_chat(STATE["processor"], retry_prompt)
        raw2, _ = generate_text(
            STATE["processor"], STATE["model"], STATE["device"], retry_chat,
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
# 11. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("  OPDDOC MEDGEMMA — AI MEDICAL TRIAGE")
    print("  Backend: AMD Radeon 860M (Vulkan via llama-server)")
    print(f"{'=' * 60}\n")
    print("  Make sure llama-server is running:")
    print("  E:\\llama-gpu\\llama-server.exe -m E:\\llama-gpu\\medgemma-4b-Q4_K_M.gguf")
    print(f"  -ngl 34 --host 127.0.0.1 --port 8080 --ctx-size 4096\n")

    if not initialize_models():
        print("\n✗ INITIALIZATION FAILED")
        print("  Is llama-server running? Check http://127.0.0.1:8080/health")
        sys.exit(1)

    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n✓ SERVER READY!")
    print(f"✓ Local:  http://127.0.0.1:{Config.PORT}")
    print(f"✓ Mobile: http://{local_ip}:{Config.PORT}\n")

    socketio.run(app, host=Config.HOST, port=Config.PORT,
                 allow_unsafe_werkzeug=True, debug=False)