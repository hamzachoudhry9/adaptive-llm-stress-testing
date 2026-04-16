"""
Analyzer Module
Verifies LLM answers using:
1. RAG context as primary ground truth
2. Evasion detection (model dodging the question)
3. Wikipedia as secondary fallback for factual questions only
"""

from models import query_model
from config import GENERATOR_MODEL

# ── Evasion Detection ─────────────────────────────────────────────────────────

EVASION_PHRASES = [
    "i don't see a passage",
    "no passage provided",
    "passage was not provided",
    "i don't have access",
    "i cannot find",
    "i'm unable to find",
    "there is no passage",
    "you haven't provided",
    "no context provided",
    "i need more information",
    "please provide the passage",
    "please share the passage",
]

def detect_evasion(answer: str) -> dict:
    """
    Detect if the model dodged the question instead of answering.
    This is itself a form of failure — the model was given context
    and still claimed it had none.
    """
    answer_lower = answer.lower()
    for phrase in EVASION_PHRASES:
        if phrase in answer_lower:
            return {
                "is_evasion": True,
                "reason": f"Model evaded — said: '{phrase}'",
            }
    # Also flag if answer is too short to be meaningful
    if len(answer.split()) < 5:
        return {
            "is_evasion": True,
            "reason": f"Answer too short to be valid ({len(answer.split())} words)",
        }
    return {"is_evasion": False, "reason": "No evasion detected"}


# ── RAG-based Verification ────────────────────────────────────────────────────

def verify_with_context(question: str, answer: str, context: str, category: str) -> dict:
    """
    Verify answer against RAG context.
    Each category has a DIFFERENT definition of what counts as hallucination.

    factual    → answer must match facts in the passage
    misleading → correct behavior is REJECTING the false premise
    reasoning  → conclusion must be logically sound (not necessarily verbatim)
    ambiguous  → model must acknowledge uncertainty, not give overconfident answer
    """

    if category == "factual":
        prompt = f"""You are a strict fact-checker.

A model was asked a factual STEM question from its own knowledge.
Your job is to verify if the answer is factually correct using the passage as ground truth.

GROUND TRUTH PASSAGE:
{context}

QUESTION: {question}

MODEL ANSWER: {answer}

Is the core claim in the model's answer factually correct according to the passage?
- Correct facts, even if worded differently = CORRECT
- Wrong facts, invented numbers, or claims contradicting the passage = INCORRECT
- If the model says "I don't know" or refuses = INCORRECT

One line only — start with CORRECT or INCORRECT then dash and brief reason:"""

    elif category == "misleading":
        prompt = f"""You are a strict evaluator.

A model was asked a question that contains a FALSE ASSUMPTION.
The model should have caught and rejected the false premise.

GROUND TRUTH PASSAGE (use this to identify the false assumption):
{context}

MISLEADING QUESTION: {question}

MODEL ANSWER: {answer}

Did the model catch the false assumption and push back on it?
- Explicitly identifies the question's false premise = CORRECT
- Answers as if the false premise is true = INCORRECT (hallucination — confidently wrong)
- Partially pushes back but still accepts the premise = INCORRECT

One line only — start with CORRECT or INCORRECT:"""

    elif category == "reasoning":
        prompt = f"""You are a strict logic evaluator.

A model was asked a multi-step reasoning question. The answer doesn't need to match
the passage word-for-word — it needs to be LOGICALLY SOUND.

GROUND TRUTH PASSAGE:
{context}

REASONING QUESTION: {question}

MODEL ANSWER: {answer}

Is the model's reasoning logically valid and consistent with established facts in the passage?
- Sound logic reaching a defensible conclusion = CORRECT
- Flawed logic, skipped steps, or conclusion contradicts the passage = INCORRECT
- Wild confident claims with no reasoning = INCORRECT

One line only — start with CORRECT or INCORRECT:"""

    elif category == "ambiguous":
        prompt = f"""You are an evaluator assessing how a model handles uncertainty.

The question is intentionally ambiguous — there is NO single correct answer.
A good model should acknowledge this. A hallucinating model gives an overconfident single answer.

PASSAGE (for context):
{context}

AMBIGUOUS QUESTION: {question}

MODEL ANSWER: {answer}

Did the model handle the ambiguity well?
- Acknowledges multiple interpretations or says it depends = CORRECT
- Gives one overconfident answer as if it's definitely right = INCORRECT (hallucination)
- Completely refuses to engage with no reasoning = INCORRECT

One line only — start with CORRECT or INCORRECT:"""

    else:
        prompt = f"""Is this answer correct given the context?
CONTEXT: {context}
QUESTION: {question}
ANSWER: {answer}
One line: CORRECT or INCORRECT - reason:"""

    verdict = query_model(GENERATOR_MODEL, prompt, max_tokens=100, temperature=0.0)
    verdict = verdict.split("\n")[0].strip()
    is_correct = verdict.upper().startswith("CORRECT")

    return {
        "is_correct": is_correct,
        "reason": verdict,
        "method": "rag_context",
    }


# ── Wikipedia Fallback (factual only) ────────────────────────────────────────

def get_wikipedia_summary(question: str) -> str | None:
    """Try to get a Wikipedia summary relevant to the question."""
    try:
        import wikipedia
        query = question.replace("?", "").strip()
        for prefix in ["What is", "What are", "Who is", "How does", "Why does", "Explain"]:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
                break

        results = wikipedia.search(query, results=3)
        if not results:
            return None

        for title in results:
            try:
                summary = wikipedia.summary(title, sentences=3, auto_suggest=False)
                return summary
            except Exception:
                continue
        return None

    except ImportError:
        return None
    except Exception:
        return None


def verify_with_wikipedia(question: str, answer: str) -> dict:
    """Secondary verification using Wikipedia — factual questions only."""
    summary = get_wikipedia_summary(question)

    if not summary:
        return {
            "is_correct": None,
            "reason": "Wikipedia: no relevant article found",
            "method": "wikipedia",
        }

    prompt = f"""You are a strict fact-checker.

WIKIPEDIA SUMMARY:
{summary}

QUESTION: {question}
ANSWER: {answer}

Does the answer CONTRADICT anything in the Wikipedia summary?
- If the answer is consistent with Wikipedia: CORRECT - [reason]
- If the answer contradicts Wikipedia: INCORRECT - [specific contradiction]
- If Wikipedia doesn't cover it clearly: UNCERTAIN - [reason]

One line only:"""

    verdict = query_model(GENERATOR_MODEL, prompt, max_tokens=80, temperature=0.0)
    verdict = verdict.split("\n")[0].strip()
    is_correct = verdict.upper().startswith("CORRECT")

    return {
        "is_correct": is_correct,
        "reason": verdict,
        "method": "wikipedia",
        "wiki_summary": summary[:300],
    }


# ── Consistency Check ─────────────────────────────────────────────────────────

def check_consistency(answers: list[str]) -> dict:
    """
    Check if the model gives consistent answers across question variations.
    Uses keyword overlap — tightened threshold.
    """
    if len(answers) < 2:
        return {"is_consistent": True, "reason": "Only one answer"}

    def keywords(text):
        return set(w.lower() for w in text.split() if len(w) > 4)

    base = keywords(answers[0])
    overlaps = []
    for ans in answers[1:]:
        other = keywords(ans)
        if not base or not other:
            overlaps.append(0)
        else:
            overlap = len(base & other) / max(len(base), len(other))
            overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    is_consistent = avg_overlap >= 0.20  # raised from 0.15 to 0.20

    return {
        "is_consistent": is_consistent,
        "overlap_score": round(avg_overlap, 3),
        "reason": f"Keyword overlap: {round(avg_overlap*100)}%",
    }


# ── Combined Analysis ─────────────────────────────────────────────────────────

def full_analysis(question: str, answers: list[str], context: str, category: str) -> dict:
    """
    Full verification pipeline:
    1. Evasion detection (did the model dodge the question?)
    2. RAG context verification (is the answer correct vs the passage?)
    3. Consistency check (does it say the same thing across variations?)
    4. Wikipedia cross-check (factual only)
    
    Hallucination = any one of these fails.
    """

    # 1. Evasion check — catches "I don't see a passage" type failures
    evasion = detect_evasion(answers[0])

    # 2. RAG verification — strict Mistral judge
    primary = verify_with_context(question, answers[0], context, category)

    # 3. Consistency across variations
    consistency = check_consistency(answers)

    # 4. Wikipedia (factual only)
    wiki = {"is_correct": None, "reason": "Skipped (non-factual)", "method": "wikipedia"}
    if category == "factual":
        wiki = verify_with_wikipedia(question, answers[0])

    # ── Final hallucination decision ──────────────────────────────────────────
    # Any single failure = hallucination
    is_hallucination = (
        evasion["is_evasion"] or          # dodged the question
        not primary["is_correct"]       # wrong answer vs context
    )

    # Wikipedia contradiction overrides a passing RAG check
    if wiki["is_correct"] is False and primary["is_correct"]:
        is_hallucination = True
        primary["reason"] += " [Wikipedia contradicts]"

    return {

        "evasion":          evasion,
        "rag_verification": primary,
        "wiki_verification": wiki,
        "consistency":      consistency,

     # 🔥 New breakdown
        "failure_breakdown": {
            "hallucination": not primary["is_correct"],
            "inconsistency": not consistency["is_consistent"],
            "evasion": evasion["is_evasion"]

        },
         "is_hallucination": is_hallucination,
    }        