"""test_agent.py -- Comprehensive test suite for Instamart Agent, Tools, RAG, and Triage.
Executable via:  python test_agent.py  or  python -m unittest test_agent.py
"""
import os
import re
import json
import sqlite3
import unittest
import pandas as pd
import numpy as np

import mock_llm
from src.policy_rag_agent import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    chunk_text,
    load_and_chunk_policies,
    RAGRetriever,
    answer,
    run_agent,
    resume_agent,
    save_checkpoint,
    TOOL_MAP,
    TOOLS_SPEC,
)
from src.support_triage import (
    redact_pii,
    validate_response,
    call_mock_llm_with_retry,
    EMAIL_PATTERN,
    PHONE_PATTERN,
    ORDER_PATTERN,
)


class TestAgentTools(unittest.TestCase):
    """Unit tests for the 5 ReAct Agent tools."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = "data/instamart.db"
        assert os.path.exists(cls.db_path), "Database data/instamart.db missing"

    def test_get_order_completed(self):
        # Known completed order with late delivery
        res = get_order("ORD-100048", db_path=self.db_path)
        self.assertNotIn("error", res)
        self.assertEqual(res["order_id"], "ORD-100048")
        self.assertEqual(res["status"], "completed")
        self.assertAlmostEqual(res["order_amount"], 128.25, places=2)
        self.assertGreater(res["delivery_minutes"], 45)

    def test_get_order_cancelled(self):
        # Find a cancelled order
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT order_id FROM orders WHERE status = 'cancelled' LIMIT 1")
        canc_oid = cur.fetchone()[0]
        conn.close()

        res = get_order(canc_oid, db_path=self.db_path)
        self.assertNotIn("error", res)
        self.assertEqual(res["status"], "cancelled")

    def test_get_order_not_found(self):
        res = get_order("ORD-999999", db_path=self.db_path)
        self.assertIn("error", res)
        self.assertEqual(res["error"], "order not found")

    def test_calculate_refund_late_delivery(self):
        res = calculate_refund(order_amount=250.50, delivery_minutes=55)
        self.assertEqual(res["refund_inr"], 250.50)

    def test_calculate_refund_on_time_delivery(self):
        res = calculate_refund(order_amount=250.50, delivery_minutes=30)
        self.assertEqual(res["refund_inr"], 0.0)

    def test_issue_refund(self):
        tmp_ledger = "outputs/test_refund_ledger.jsonl"
        res = issue_refund("ORD-TEST01", 150.0)
        self.assertEqual(res["refund_inr"], 150.0)

    def test_escalate_to_human(self):
        res = escalate_to_human("ORD-TEST02", 1500.0)
        self.assertEqual(res["refund_inr"], 1500.0)


class TestPolicyRAG(unittest.TestCase):
    """Unit tests for document chunking, TF-IDF vectorization, and retrieval."""

    def test_chunking_overlap_10(self):
        chunks = load_and_chunk_policies(overlap=10)
        self.assertEqual(len(chunks), 15, f"Expected 15 chunks with overlap 10, got {len(chunks)}")

    def test_chunking_overlap_0(self):
        chunks = load_and_chunk_policies(overlap=0)
        self.assertEqual(len(chunks), 13, f"Expected 13 chunks with overlap 0, got {len(chunks)}")

    def test_tfidf_retrieval(self):
        chunks = load_and_chunk_policies(overlap=10)
        retriever = RAGRetriever(chunks)
        top3 = retriever.retrieve("refund policy late delivery", k=3)
        self.assertEqual(len(top3), 3)
        for r in top3:
            self.assertEqual(r["source"], "refund_policy.txt")
            self.assertGreater(r["score"], 0.20)

    def test_grounded_answer_and_fallback(self):
        chunks = load_and_chunk_policies(overlap=10)
        retriever = RAGRetriever(chunks)

        # Answerable query
        ans, src = answer("What delivery time makes a completed order eligible for a full refund?", retriever)
        self.assertIn("refund_policy.txt", src)
        self.assertTrue("45 minutes" in ans or "more than 45" in ans)

        # Unanswerable query
        ans_unans, _ = answer("What is the favourite colour of the company mascot?", retriever)
        self.assertTrue(ans_unans.startswith("I could not find"))


class TestReActRefundAgent(unittest.TestCase):
    """Integration tests for all 7 ReAct refund agent scenarios and guardrails."""

    def setUp(self):
        mock_llm.reset()

    def test_scenario_s1_full_refund(self):
        res = run_agent("My order ORD-100048 reached me very late. I would like a refund.", "S1_test")
        self.assertEqual(res["status"], "REFUND")
        self.assertIn("128.25", res["final_answer"])
        self.assertEqual(res["llm_calls"], 5)

    def test_scenario_s2_full_refund(self):
        res = run_agent("I was delivered order ORD-100067 after a massive delay. Please issue my refund.", "S2_test")
        self.assertEqual(res["status"], "REFUND")
        self.assertIn("135", res["final_answer"])
        self.assertEqual(res["llm_calls"], 5)

    def test_scenario_s3_escalate_human(self):
        res = run_agent("My electronics order ORD-100145 was delayed over an hour. Where is my refund?", "S3_test")
        self.assertEqual(res["status"], "ESCALATE")
        self.assertEqual(res["llm_calls"], 5)

    def test_scenario_s4_denied_ontime(self):
        res = run_agent("Please refund order ORD-100001, I did not like the delivery.", "S4_test")
        self.assertEqual(res["status"], "DENIED")
        self.assertIn("within 45 minutes", res["final_answer"])

    def test_scenario_s5_denied_cancelled(self):
        res = run_agent("I want my money back for order ORD-100025.", "S5_test")
        self.assertEqual(res["status"], "DENIED")
        self.assertIn("only completed orders", res["final_answer"])

    def test_scenario_s6_not_found(self):
        res = run_agent("Please refund order ORD-999999.", "S6_test")
        self.assertEqual(res["status"], "NOT_FOUND")

    def test_scenario_s7_loop_fallback(self):
        res = run_agent("[LOOP] Check refund policy for ORD-100048.", "S7_test", max_iterations=6)
        self.assertEqual(res["status"], "FALLBACK_MAX_ITERATIONS")
        self.assertEqual(res["llm_calls"], 6)

    def test_token_limit_guardrail(self):
        res = run_agent("My order ORD-100048 was late. Please refund.", "S1_tok_guard", token_limit=200)
        self.assertEqual(res["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertIn("Token limit exceeded", res["final_answer"])

    def test_timeout_guardrail(self):
        res = run_agent("My order ORD-100048 was late. Please refund.", "S1_time_guard", time_limit_seconds=0)
        self.assertEqual(res["status"], "FALLBACK_TIMEOUT")
        self.assertIn("Execution timed out", res["final_answer"])

    def test_crash_and_resume(self):
        crash_id = "test_s1_crash_resume"
        msg = "My order ORD-100048 reached me very late. I would like a refund."

        # Simulate crash right after step 2 (search_policy)
        with self.assertRaises(RuntimeError):
            run_agent(msg, "S1_crash", simulate_crash_step=2, run_id=crash_id)

        # Resume agent
        resumed = resume_agent(crash_id)
        self.assertEqual(resumed["status"], "REFUND")
        self.assertIn("128.25", resumed["final_answer"])
        # Exactly 4 tools executed total (2 before crash, 2 after resume)
        self.assertEqual(len(resumed["tools_executed"]), 4)
        self.assertEqual(resumed["tools_executed"], ["get_order", "search_policy", "calculate_refund", "issue_refund"])


class TestSupportTicketTriage(unittest.TestCase):
    """Unit tests for Support Ticket Triage PII redaction, retries, and validation."""

    def test_pii_redaction(self):
        text = "Order ORD-123456 delayed. Email user12@test.co.in or call 9876543210."
        redacted, n_e, n_p, n_o = redact_pii(text)
        self.assertEqual(n_e, 1)
        self.assertEqual(n_p, 1)
        self.assertEqual(n_o, 1)
        self.assertNotIn("user12@test.co.in", redacted)
        self.assertNotIn("9876543210", redacted)
        self.assertNotIn("ORD-123456", redacted)
        self.assertIn("[EMAIL]", redacted)
        self.assertIn("[PHONE]", redacted)
        self.assertIn("[ORDER_ID]", redacted)

    def test_response_validation_json(self):
        valid_json = '{"category": "fraud", "reason": "unauthorized charge"}'
        status, cat = validate_response(valid_json)
        self.assertEqual(status, "classified_json")
        self.assertEqual(cat, "fraud")

    def test_response_validation_keyword(self):
        chatty_text = "The customer appears to have a billing-related problem."
        status, cat = validate_response(chatty_text)
        self.assertEqual(status, "classified_keyword")
        self.assertEqual(cat, "billing")

    def test_response_validation_invalid_category(self):
        invalid_cat = '{"category": "shipping", "reason": "parcel movement"}'
        status, cat = validate_response(invalid_cat)
        self.assertEqual(status, "flagged_invalid_output")
        self.assertIsNone(cat)

    def test_mock_llm_retry_recovery(self):
        mock_llm.reset()
        # Call 8 times ok, 9th call triggers TransientLLMError which retry recovers
        for _ in range(8):
            mock_llm._classify([
                {"role": "system", "content": "CLASSIFY"},
                {"role": "user", "content": "Help with bill"}
            ])
        # 9th call should succeed via call_mock_llm_with_retry
        resp = call_mock_llm_with_retry([
            {"role": "system", "content": "CLASSIFY"},
            {"role": "user", "content": "Help with bill"}
        ], "T_RETRY_TEST")
        self.assertIsNotNone(resp)


if __name__ == "__main__":
    unittest.main()
