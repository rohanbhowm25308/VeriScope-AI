"""
groq_client.py
---------------
Thin wrapper around Groq's OpenAI-compatible chat completions endpoint.

Used for two OPTIONAL features:
  1. The in-app chatbot (ask questions about a claim / the tool).
  2. "AI Review" -- an LLM-assisted second opinion on a claim, useful because
     the offline heuristic/ML baseline has no access to real-world facts
     beyond what's in its lexicon and whatever context the user supplies.

Degrades gracefully: if GROQ_API_KEY is not set, callers get a clear,
non-crashing message instead of a stack trace. This keeps the core
ML pipeline (claim_analyzer.py) fully usable at zero cost, per the
project's resource boundary -- Groq is an enhancement, not a dependency.
"""

import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are the assistant embedded inside ClaimGuard, a claim-auditing research "
    "tool. Your job is to help the user reason about whether a factual claim "
    "requires external verification -- NOT to declare claims true or false. "
    "When asked to review a claim, comment on: what would need to be checked, "
    "why it is or isn't time-sensitive, and what kind of source could resolve it. "
    "Be concise, calibrated, and explicit about your own uncertainty. Never "
    "state a claim is definitely true or false without citing what evidence "
    "that would require."
)


def is_configured():
    return bool(os.environ.get("GROQ_API_KEY"))


def _call_groq(messages, temperature=0.3, max_tokens=600):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY is not set. Add it to backend/.env to enable AI chat / AI review."
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": DEFAULT_MODEL, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content, None
    except requests.exceptions.RequestException as e:
        return None, f"Groq API request failed: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return None, f"Unexpected Groq API response: {e}"


def chat(user_message, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-10:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": user_message})
    return _call_groq(messages)


def ai_review(claim_text, heuristic_summary):
    prompt = (
        f"Claim: \"{claim_text}\"\n\n"
        f"Our heuristic/ML system's assessment: {heuristic_summary}\n\n"
        "Give a short second opinion (4-6 sentences): does this claim plausibly "
        "require external verification? What specifically should be checked? "
        "Do not assert whether the claim is true or false."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    return _call_groq(messages, temperature=0.2, max_tokens=350)
