"""How a System One question becomes the student's input: one sequence per question, options scored at markers.

    <bos> TAG [question text] <eos> OPT option 1 OPT option 2 ... <eos> state <eos>

- Atelier's fixed questions (needsYou, agentClicks, ...) are recognised and given a one-token TAG and their option
  KEYS only (``none``, ``purchase`` ...): the student learns what they mean from the teacher, so it never re-reads
  the 250-token criteria Atelier sends. That is most of the student's speed-up over reading the full question.
- Any other question (label_items, routines, packs) keeps its instructions and option descriptions, with the
  generic OPEN tag, so the same model still answers open questions.
- OPT is a marker token before each option; the head scores the marker's hidden state. A ``noul`` is the two
  options ``false``/``true``, a ``score`` its levels in order.

This module is the contract between training (Python) and the runtime (runtime/), which re-implements it.
"""

from __future__ import annotations

import json
import re

# mmBERT's tokenizer has 100 unused tokens (<unused0> = id 7 ...); they become markers and use tags.
OPT = "<unused0>"
OPEN = "<unused1>"
TAGS = {  # question key -> tag token (Atelier's own decisions)
    "waits": "<unused2>",        # needsYou
    "dictation": "<unused3>",    # dictation
    "kind": "<unused4>",         # agentClicks
    "intent": "<unused5>",       # searchOrAsk
    "target": "<unused6>",       # voiceTarget
    "effort": "<unused7>",       # effort
    "lasting": "<unused8>",      # memory
    "place": "<unused9>",        # tabs
    "skill": "<unused10>",       # skills (one question per skill)
    "techLevel": "<unused11>",   # setup
    "replyStyle": "<unused12>",  # setup
    "interest": "<unused13>",    # setup (interest_<x>)
}
# Instructions that identify Atelier's fixed questions (a key alone is not enough: packs may reuse names).
SIGNATURES = {
    "waits": "This is the end of an assistant’s reply.",
    "dictation": "Someone spoke into a text field.",
    "kind": "An assistant is about to activate this control",
    "intent": "Someone typed this into their browser’s search box.",
    "target": "Someone spoke while a text field was focused",
    "effort": "Someone sent this message to their AI assistant. How much careful thinking",
    "lasting": "This is a conversation between a person and their AI assistant. Does it tell the assistant anything worth",
    "place": "Someone is sorting their open browser tabs into groups.",
    "skill": "Someone sent this first message to their AI assistant. Would the assistant’s skill",
    "techLevel": "From this setup chat with their new browser assistant",
    "replyStyle": "Did the person say how they like the assistant’s answers?",
    "interest": "Does the person want help with, or do a lot of,",
}
SKILL_RE = re.compile(r"skill “/([^”]+)” help with it\? The skill: (.*)$", re.S)
GROUP_RE = re.compile(r"Their own group “(.*)”: anything that belongs with it\.$", re.S)
HEAD_TAIL = {"waits", "lasting"}  # Atelier already sends the END of these; keep the end when cutting
# Decisions over whole chats run in the background with seconds to spare: they may read more.
LONG_LIMITS = {"lasting": 1536, "techLevel": 2048, "replyStyle": 2048, "interest": 2048}


def option_labels(question: dict) -> list[str]:
    """Option keys in answer order (as pplx_decider.protocol.options)."""
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def fixed_kind(key: str, question: dict) -> str | None:
    text = question.get("instructions") or ""
    for name, signature in SIGNATURES.items():
        if text.startswith(signature) or (name in ("skill", "interest") and signature in text):
            return name
    return None


def state_text(state) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in state.items())
    return json.dumps(state, ensure_ascii=False)


def question_parts(key: str, question: dict) -> tuple[str, list[str], str | None]:
    """(question text with its tag, option texts, fixed kind or None)."""
    kind = fixed_kind(key, question)
    labels = option_labels(question)
    if kind == "skill":
        m = SKILL_RE.search(question.get("instructions") or "")
        name, description = (m.group(1), m.group(2)) if m else ("", question.get("instructions") or "")
        return f"{TAGS['skill']} /{name}: {description}", ["false", "true"], kind
    if kind == "interest":
        topic = (question.get("instructions") or "").rsplit(",", 1)[-1].strip(" ?")
        return f"{TAGS['interest']} {topic}", ["false", "true"], kind
    if kind == "place":
        options = []
        for label in labels:
            description = question["criteria"][label] or ""
            g = GROUP_RE.match(description)
            options.append(f"group: {g.group(1)}" if g else label)
        return TAGS["place"], options, kind
    if kind is not None:
        if question["type"] == "score":
            return TAGS[kind], [f"level {i}" for i in range(len(labels))], kind
        return TAGS[kind], labels, kind
    # An open question: full instructions and descriptions.
    instructions = question.get("instructions") or "Choose the best matching option."
    if question["type"] == "choice":
        options = [k if v is None else f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                   for k, v in question["criteria"].items()]
    elif question["type"] == "score":
        options = [str(v) for v in question["criteria"]]
    else:
        criteria = question.get("criteria") or {}
        options = [f"no: {criteria['false']}" if criteria.get("false") else "no",
                   f"yes: {criteria['true']}" if criteria.get("true") else "yes"]
    return f"{OPEN} {instructions}", options, None


def encode(tokenizer, state, key: str, question: dict, max_length: int = 512) -> dict:
    """Token ids of one question with its state, and the positions of its option markers."""
    head, options, kind = question_parts(key, question)
    opt_id = tokenizer.convert_tokens_to_ids(OPT)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    ids = [bos, *tokenizer.encode(head, add_special_tokens=False)[:160], eos]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids += [opt_id, *tokenizer.encode(option, add_special_tokens=False)[:24]]
    ids.append(eos)
    limit = max(max_length, LONG_LIMITS.get(kind or "", 0))
    budget = max(16, limit - len(ids) - 1)
    state_ids = tokenizer.encode(state_text(state), add_special_tokens=False)
    if len(state_ids) > budget:
        state_ids = state_ids[-budget:] if kind in HEAD_TAIL else state_ids[:budget]
    ids += [*state_ids, eos]
    return {"input_ids": ids, "markers": markers, "kind": kind or "open"}
