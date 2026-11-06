"""support_triage.py -- Part C: Support Ticket Triage with Guardrails.
Implements input guardrails, PII redaction, zero-shot and few-shot classification,
resilient calls with backoff/jitter, response validation, token budget, and routing.
"""
import os
import re
import json
import time
import sys
from pathlib import Path

# Add repository root to sys.path so mock_llm and local modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

SYSTEM_ZERO_SHOT = """You are an internal customer support triage assistant. Your task is to CLASSIFY the customer support ticket into exactly one of four allowed categories: billing, delivery, technical, fraud.
Reply strictly with a JSON object in this format: {"category": "<category>", "reason": "<short explanation>"}. Do not include any other text."""

SYSTEM_FEW_SHOT = """You are an internal customer support triage assistant. Your task is to CLASSIFY the customer support ticket into exactly one of four allowed categories: billing, delivery, technical, fraud.
Reply strictly with a JSON object in this format: {"category": "<category>", "reason": "<short explanation>"}. Do not include any other text.
EXAMPLES:
Ticket: "There is an unauthorized transaction on my card linked to my order."
Reply: {"category": "fraud", "reason": "unauthorized card activity"}
Ticket: "The delivery rider never showed up and the order is marked completed."
Reply: {"category": "delivery", "reason": "missing delivery"}"""

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")


def redact_pii(text):
    """Redacts email, phone numbers, and order IDs from ticket text."""
    t, n_email = EMAIL_PATTERN.subn("[EMAIL]", text)
    t, n_phone = PHONE_PATTERN.subn("[PHONE]", t)
    t, n_order = ORDER_PATTERN.subn("[ORDER_ID]", t)
    return t, n_email, n_phone, n_order


def call_mock_llm_with_retry(messages, ticket_id):
    """Wraps model call with max 3 retries, exponential backoff, and jitter."""
    BASE_DELAY = 0.05
    for n in range(4):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError as e:
            if n < 3:
                wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
                print(f"[RETRY LOG] Ticket {ticket_id}: retry {n}, waiting {wait:.4f}s ({e})")
                time.sleep(wait)
            else:
                raise e


def validate_response(content):
    """Parses response as JSON or falls back to word extraction."""
    # Attempt 1: Strict JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = data.get("category", "").lower()
            if cat in ALLOWED_CATEGORIES:
                return "classified_json", cat
            else:
                return "flagged_invalid_output", None
    except Exception:
        pass

    # Attempt 2: Keyword extraction
    words = set(re.findall(r"[a-z]+", (content or "").lower()))
    matches = [c for c in ALLOWED_CATEGORIES if c in words]
    if len(matches) == 1:
        return "classified_keyword", matches[0]

    return "flagged_invalid_output", None


def run_triage_pipeline(prompt_type="zero_shot", token_budget=None, tickets_path="data/raw/tickets.csv"):
    mock_llm.reset()
    random.seed(42)  # Fixed seed for reproducible jitter
    tickets = pd.read_csv(tickets_path, keep_default_na=False)

    system_prompt = SYSTEM_FEW_SHOT if prompt_type == "few_shot" else SYSTEM_ZERO_SHOT
    results = []
    total_tokens_used = 0

    for _, row in tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]

        # 1. Input guardrail: empty or whitespace
        if not raw_text.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # 2. Input guardrail: long text
        if mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # 3. Budget guardrail: check running tokens before call
        if token_budget is not None and total_tokens_used >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # 4. Redact PII
        redacted, _, _, _ = redact_pii(raw_text)

        # Assert no leftover PII in prompt
        assert len(EMAIL_PATTERN.findall(redacted)) == 0, f"Leftover email in {tid}"
        assert len(PHONE_PATTERN.findall(redacted)) == 0, f"Leftover phone in {tid}"
        assert len(ORDER_PATTERN.findall(redacted)) == 0, f"Leftover order in {tid}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted}
        ]

        resp = call_mock_llm_with_retry(messages, tid)
        tokens = resp.usage.total_tokens
        total_tokens_used += tokens

        status, pred_cat = validate_response(resp.choices[0].message.content)
        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": pred_cat if pred_cat else "", "true_category": true_cat,
            "tokens_used": tokens
        })

    df = pd.DataFrame(results)
    return df, total_tokens_used


def apply_routing_and_save():
    os.makedirs("outputs", exist_ok=True)
    seg_df = pd.read_csv("outputs/customer_segments.csv")
    cust_seg_map = seg_df.set_index("customer_id").to_dict("index")

    def route_ticket(status, category, customer_id):
        cust_info = cust_seg_map.get(customer_id, {})
        is_prem = cust_info.get("is_premium", False)
        seg_name = cust_info.get("segment_name", "")

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            return "human_review", "NORMAL", "", ""
        elif category == "fraud":
            return "security_team", "HIGH", category, seg_name
        elif is_prem:
            return "premium_desk", "HIGH", category, seg_name
        else:
            return "support_queue", "NORMAL", category, seg_name

    # 1. Budget Demo
    print("\n--- Running Budget Demo (TOKEN_BUDGET = 6000) ---")
    df_budget, tok_budget = run_triage_pipeline("zero_shot", token_budget=6000)
    rows_budget = []
    for _, r in df_budget.iterrows():
        route, prio, cat_out, sname_out = route_ticket(r["status"], r["category"], r["customer_id"])
        rows_budget.append({
            "ticket_id": r["ticket_id"],
            "customer_id": r["customer_id"],
            "status": r["status"],
            "category": cat_out,
            "segment_name": sname_out,
            "priority": prio,
            "route": route,
            "tokens_used": r["tokens_used"]
        })
    df_budget_out = pd.DataFrame(rows_budget)
    df_budget_out.to_csv("outputs/triage_budget_demo.csv", index=False)
    exhausted_count = (df_budget_out["status"] == "budget_exhausted").sum()
    print(f"Saved outputs/triage_budget_demo.csv ({exhausted_count} tickets ended budget_exhausted)")

    # 2. Final Triage Results (Few-Shot, Budget 100,000)
    print("\n--- Running Final Few-Shot Triage (TOKEN_BUDGET = 100000) ---")
    df_final, tok_final = run_triage_pipeline("few_shot", token_budget=100000)
    rows_final = []
    for _, r in df_final.iterrows():
        route, prio, cat_out, sname_out = route_ticket(r["status"], r["category"], r["customer_id"])
        rows_final.append({
            "ticket_id": r["ticket_id"],
            "customer_id": r["customer_id"],
            "status": r["status"],
            "category": cat_out,
            "segment_name": sname_out,
            "priority": prio,
            "route": route,
            "tokens_used": r["tokens_used"]
        })
    df_final_out = pd.DataFrame(rows_final)
    df_final_out.to_csv("outputs/triage_results.csv", index=False)
    print(f"Saved outputs/triage_results.csv ({len(df_final_out)} tickets)")


if __name__ == "__main__":
    print("=== PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS ===")
    # 1. Audit PII
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    e_tot, p_tot, o_tot = 0, 0, 0
    for t in tickets["text"]:
        _, ne, np_, no = redact_pii(t)
        e_tot += ne; p_tot += np_; o_tot += no
    print(f"PII Substitutions across {len(tickets)} tickets: Email={e_tot}, Phone={p_tot}, OrderID={o_tot}, Total={e_tot+p_tot+o_tot}")

    # 2. Compare Zero-Shot vs Few-Shot
    print("\n--- Running Zero-Shot Evaluation (no budget) ---")
    df_zero, tok_zero = run_triage_pipeline("zero_shot")
    clf_zero = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])]
    acc_zero = (clf_zero["category"] == clf_zero["true_category"]).mean()
    print(f"Zero-shot classified: {len(clf_zero)}/156, Accuracy: {acc_zero:.4f}, Total tokens: {tok_zero}, Tokens/classified: {tok_zero/len(clf_zero):.1f}")
    print("Zero-shot confusion table:")
    print(pd.crosstab(clf_zero["true_category"], clf_zero["category"]))

    print("\n--- Running Few-Shot Evaluation (no budget) ---")
    df_few, tok_few = run_triage_pipeline("few_shot")
    clf_few = df_few[df_few["status"].isin(["classified_json", "classified_keyword"])]
    acc_few = (clf_few["category"] == clf_few["true_category"]).mean()
    print(f"Few-shot classified: {len(clf_few)}/156, Accuracy: {acc_few:.4f}, Total tokens: {tok_few}, Tokens/classified: {tok_few/len(clf_few):.1f}")

    # 3. Budget demo and routing outputs
    apply_routing_and_save()
