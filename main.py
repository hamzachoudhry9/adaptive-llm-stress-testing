"""
LLM Stress Tester — Main Entry Point
-------------------------------------
Flow:
  1. Load RAG JSONL (your domain knowledge base)
  2. Bandit selects category to test
  3. Retrieve relevant context from RAG
  4. Mistral generates a grounded question from context
  5. Llama3 answers the question (+ 2 variations for consistency)
  6. Analyzer verifies answer against context + Wikipedia
  7. Bandit updates based on hallucination result
  8. Results saved to results.jsonl
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from config import (
    CURRENT_DOMAIN, RAG_FILE, TEST_MODEL,
    NUM_STEPS, RAG_TOP_K, REPORT_FILE
)
from rag import RAG
from generator import generate_question, generate_variations
from analyzer import full_analysis
from bandit import UCB1Bandit
from models import query_model


def print_header():
    print("\n" + "="*60)
    print(f"  LLM STRESS TESTER")
    print(f"  Domain  : {CURRENT_DOMAIN}")
    print(f"  RAG     : {RAG_FILE}")
    print(f"  Testing : {TEST_MODEL}")
    print(f"  Steps   : {NUM_STEPS}")
    print("="*60 + "\n")


def save_result(record: dict, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def confidence_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    import math
    z = 1.96
    p = successes / total
    center = (p + z**2 / (2*total)) / (1 + z**2 / total)
    margin = (z * math.sqrt(p*(1-p)/total + z**2/(4*total**2))) / (1 + z**2/total)
    return (round(max(0, center - margin) * 100, 1), round(min(1, center + margin) * 100, 1))


def print_summary(bandit: UCB1Bandit, records: list, total: int, hallucinations: int):
    from config import TEST_MODEL, CURRENT_DOMAIN, CATEGORIES

    print("\n" + "="*60)
    print(f"  HALLUCINATION REPORT")
    print(f"  Model   : {TEST_MODEL}")
    print(f"  Domain  : {CURRENT_DOMAIN}")
    print(f"  Steps   : {total}")
    print("="*60)

    overall_rate = round(hallucinations / total * 100, 1) if total else 0
    lo, hi = confidence_interval(hallucinations, total)

    print(f"\n  Overall hallucination rate : {overall_rate}%")
    print(f"  95% confidence interval    : [{lo}% — {hi}%]")

    # ─────────────────────────────────────────────
    # 🔥 NEW: FAILURE BREAKDOWN
    # ─────────────────────────────────────────────
    h_count = 0
    i_count = 0
    e_count = 0

    for r in records:
        fb = r["analysis"]["failure_breakdown"]
        if fb["hallucination"]:
            h_count += 1
        if fb["inconsistency"]:
            i_count += 1
        if fb["evasion"]:
            e_count += 1

    print(f"\n  Failure Breakdown:")
    print(f"    Hallucination : {h_count} ({round(h_count/total*100,1)}%)")
    print(f"    Inconsistency : {i_count} ({round(i_count/total*100,1)}%)")
    print(f"    Evasion       : {e_count} ({round(e_count/total*100,1)}%)")

    # ─────────────────────────────────────────────

    print(f"\n  {'Category':<14} {'Rate':>6}  {'CI (95%)':<18}  {'Samples':>7}  Verdict")
    print(f"  {'─'*14} {'─'*6}  {'─'*18}  {'─'*7}  {'─'*20}")

    for cat in CATEGORIES:
        subset = [r for r in records if r["category"] == cat]
        n = len(subset)
        if n == 0:
            continue
        h = sum(1 for r in subset if r["is_hallucination"])
        rate = round(h / n * 100, 1)
        lo, hi = confidence_interval(h, n)
        ci = f"[{lo}% — {hi}%]"

        if rate >= 50:
            verdict = "🚨 High risk"
        elif rate >= 25:
            verdict = "⚠️  Moderate risk"
        elif rate >= 10:
            verdict = "🟡 Low risk"
        else:
            verdict = "✅ Reliable"

        print(f"  {cat:<14} {rate:>5}%  {ci:<18}  {n:>7}  {verdict}")

    print(f"\n  {'─'*56}")
    print(f"  CONCLUSION")

    if overall_rate >= 40:
        print(f"  ⛔ Do NOT use {TEST_MODEL} for {CURRENT_DOMAIN} content.")
    elif overall_rate >= 20:
        print(f"  ⚠️  Use {TEST_MODEL} with caution.")
    elif overall_rate >= 10:
        print(f"  🟡 Mostly reliable — verify important facts.")
    else:
        print(f"  ✅ Reliable for {CURRENT_DOMAIN} content.")

    # Weakest category
    cat_rates = []
    for cat in CATEGORIES:
        subset = [r for r in records if r["category"] == cat]
        if subset:
            h = sum(1 for r in subset if r["is_hallucination"])
            cat_rates.append((cat, round(h/len(subset)*100, 1), len(subset)))

    if cat_rates:
        worst = max(cat_rates, key=lambda x: x[1])
        print(f"\n  Weakest area : {worst[0]} ({worst[1]}%)")

    print(f"\n  Full results → {REPORT_FILE}")
    print("="*60 + "\n")


def main():
    print_header()

    try:
        rag = RAG(RAG_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    bandit = UCB1Bandit()
    hallucination_count = 0
    all_records = []

    Path(REPORT_FILE).unlink(missing_ok=True)

    for step in range(1, NUM_STEPS + 1):
        print(f"\n{'─'*60}")
        print(f"Step {step}/{NUM_STEPS}")

        domain, category = bandit.select()
        print(f"Category: {category.upper()}")

        rag_record = rag.random_record()
        context    = rag_record["context"]
        topic      = rag_record.get("topic", "STEM")
        field      = rag_record.get("field", domain)
        concept    = rag_record.get("concept", "")

        print(f"Topic   : {field} > {topic}")

        try:
            question = generate_question(category, context, topic)
        except Exception as e:
            print(f"[SKIP] {e}")
            continue

        print(f"Question: {question}")

        variations = generate_variations(question, context)

        answers = []
        for i, q in enumerate(variations):
            ans = query_model(TEST_MODEL, q)
            answers.append(ans)
            print(f"\n[{i}] {q}")
            print(f"Answer: {ans[:150]}")

        analysis = full_analysis(question, answers, context, category)

        if analysis["is_hallucination"]:
            hallucination_count += 1

        fb = analysis["failure_breakdown"]
        print(f"\nBreakdown: H={fb['hallucination']} I={fb['inconsistency']} E={fb['evasion']}")

        bandit.update(domain, category, analysis["is_hallucination"])

        record = {
            "step": step,
            "category": category,
            "question": question,
            "answers": answers,
            "analysis": analysis,
            "is_hallucination": analysis["is_hallucination"],
        }

        save_result(record, REPORT_FILE)
        all_records.append(record)

    print_summary(bandit, all_records, NUM_STEPS, hallucination_count)


if __name__ == "__main__":
    main()