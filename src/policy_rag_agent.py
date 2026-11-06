"""policy_rag_agent.py -- Part D: Policy RAG Assistant and Refund Agent.
Implements:
1. Document chunking (overlap=10 and overlap=0)
2. TF-IDF vectorization and cosine similarity retrieval
3. Grounded Q&A assistant with mock_llm
4. Grounding and retrieval evaluation against policy_eval.csv
5. 5 Refund Agent tools with schema and execution map
6. ReAct agent loop with tool calls, iteration/token/time bounds
7. Stop-safely demonstrations (token limit, timeout)
8. Crash-and-resume demonstration with persistent checkpointing
"""
import os
import re
import glob
import json
import time
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mock_llm

os.makedirs("state", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# Task 1: Chunk Policy Documents
# ==========================================
def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def load_and_chunk_policies(overlap=10, policies_dir="data/policies"):
    files = sorted(glob.glob(os.path.join(policies_dir, "*.txt")))
    all_chunks = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        chnks = chunk_text(txt, 40, overlap)
        for c in chnks:
            all_chunks.append({"source": fname, "text": c})
    return all_chunks


# ==========================================
# Task 2: Vectorize and Retrieve
# ==========================================
class RAGRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": round(float(sims[idx]), 4),
                "text": self.chunks[idx]["text"]
            })
        return results


# ==========================================
# Task 3: Augment and Answer
# ==========================================
def answer(question, retriever_obj):
    retrieved = retriever_obj.retrieve(question, k=3)
    sources = [r["source"] for r in retrieved]
    context_blocks = "\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)
    messages = [
        {
            "role": "system",
            "content": "You are a POLICY ASSISTANT. Answer the question strictly using the supplied context. If the information is not present in the context, say 'I could not find that information in the policy documents.'"
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context_blocks}\n\nQUESTION:\n{question}"
        }
    ]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, sources


# ==========================================
# Task 4: Evaluate Grounding & Overlap
# ==========================================
def evaluate_retrieval_and_grounding(retriever_obj, eval_path="data/raw/policy_eval.csv", label=""):
    eval_df = pd.read_csv(eval_path)
    top_doc_matches = 0
    fact_matches = 0
    fallback_matches = 0
    failed_details = []

    for idx, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]

        ans_text, sources = answer(q, retriever_obj)

        if exp_doc != "NONE":
            doc_in_top3 = exp_doc in sources
            fact_in_ans = exp_fact in ans_text
            if doc_in_top3:
                top_doc_matches += 1
            if fact_in_ans:
                fact_matches += 1
            if not doc_in_top3 or not fact_in_ans:
                failed_details.append({
                    "question": q,
                    "expected_doc": exp_doc,
                    "doc_found": doc_in_top3,
                    "expected_fact": exp_fact,
                    "fact_found": fact_in_ans,
                    "answer": ans_text
                })
        else:
            is_fallback = ans_text.startswith("I could not find")
            if is_fallback:
                fallback_matches += 1
            else:
                failed_details.append({
                    "question": q,
                    "expected_doc": "NONE",
                    "doc_found": True,
                    "expected_fact": "Fallback",
                    "fact_found": False,
                    "answer": ans_text
                })

    return {
        "label": label,
        "total_answerable": 12,
        "top_doc_matches": top_doc_matches,
        "fact_matches": fact_matches,
        "total_unanswerable": 2,
        "fallback_matches": fallback_matches,
        "failed_details": failed_details
    }


# ==========================================
# Task 5: Define Tools and Tool Map
# ==========================================
def get_order(order_id, db_path="data/instamart.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "order_id": row[0],
            "status": row[1],
            "order_amount": float(row[2]),
            "delivery_minutes": int(row[3]) if row[3] is not None and str(row[3]).isdigit() else row[3]
        }
    return {"error": "order not found"}


# Global retriever for the agent
_agent_retriever = None

def get_agent_retriever():
    global _agent_retriever
    if _agent_retriever is None:
        chunks = load_and_chunk_policies(overlap=10)
        _agent_retriever = RAGRetriever(chunks)
    return _agent_retriever


def search_policy(query):
    r = get_agent_retriever()
    res = r.retrieve(query, k=3)
    return {"results": res}


def calculate_refund(order_amount, delivery_minutes):
    amt = float(order_amount)
    deliv = float(delivery_minutes) if delivery_minutes is not None else 0.0
    if deliv > 45.0:
        return {"refund_inr": round(amt, 2)}
    return {"refund_inr": 0.0}


def issue_refund(order_id, refund_inr):
    amt = float(refund_inr)
    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": amt
    }
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": amt}


def escalate_to_human(order_id, refund_inr):
    amt = float(refund_inr)
    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": amt
    }
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": amt}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Fetch order status, order amount, and delivery minutes from the database.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search store policy documents using TF-IDF retrieval.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate eligible refund amount in INR based on order amount and delivery minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Approve and issue an automated refund up to INR 1000 to the original payment method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate high-value refund requests above INR 1000 to a human support lead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


# ==========================================
# Task 6, 7 & 8: ReAct Loop, Safe Stops, Resumability
# ==========================================
def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, tools_executed):
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_num": step_num,
        "tools_executed": tools_executed
    }
    ckpt_path = os.path.join("state", f"{run_id}.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, simulate_crash_step=None, run_id=None):
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time() * 1000)}"

    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "You are a customer service AGENT that handles refund requests."},
        {"role": "user", "content": user_message}
    ]
    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []

    for it in range(max_iterations):
        # Time limit check at top of iteration
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            summary = f"Execution timed out after {elapsed:.2f}s across {len(steps)} steps. Handing off case to human lead."
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": summary,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        # Model call
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # Token limit check after model call
        if tokens_used > token_limit:
            summary = f"Token limit exceeded ({tokens_used} > {token_limit}). Stopped immediately without tool execution. Handing off to human lead."
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": summary,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            # Final text response
            final_content = msg.content
            status = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        # Assistant tool call message
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
        })

        for tc in tool_calls:
            func = TOOL_MAP[tc.name]
            args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            obs = func(**args)
            tools_executed.append(tc.name)
            steps.append({
                "step": len(steps) + 1,
                "tool": tc.name,
                "arguments": args,
                "observation": obs
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(obs)
            })

            save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed)

            if simulate_crash_step is not None and len(tools_executed) == simulate_crash_step:
                raise RuntimeError(f"Simulated crash after tool execution #{simulate_crash_step} ({tc.name})")

    summary = f"Max iterations ({max_iterations}) reached without a final answer. Handing off case with {len(steps)} steps completed to human lead."
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": summary,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    ckpt_path = os.path.join("state", f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    start_time = time.monotonic()
    scenario_id = state["scenario_id"]
    messages = state["messages"]
    tokens_used = state["tokens_used"]
    tools_executed = list(state["tools_executed"])
    steps = []
    step_num = 1
    for m in messages:
        if m["role"] == "tool":
            steps.append({
                "step": step_num,
                "tool": m["name"],
                "arguments": {},
                "observation": json.loads(m["content"])
            })
            step_num += 1

    llm_calls = 0

    for it in range(max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            summary = f"Execution timed out after {elapsed:.2f}s during resume. Handing off to human lead."
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": summary,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            summary = f"Token limit exceeded during resume ({tokens_used} > {token_limit}). Stopped. Handing off to human lead."
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": summary,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final_content = msg.content
            status = final_content.split(":")[0].strip() if ":" in final_content else "UNKNOWN"
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed
            }

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
        })

        for tc in tool_calls:
            func = TOOL_MAP[tc.name]
            args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            obs = func(**args)
            tools_executed.append(tc.name)
            steps.append({
                "step": len(steps) + 1,
                "tool": tc.name,
                "arguments": args,
                "observation": obs
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(obs)
            })

            save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), tools_executed)

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "Max iterations reached in resumed run.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }


def run_all_part_d():
    print("=== PART D: POLICY RAG ASSISTANT AND REFUND AGENT ===")
    
    # 1. Chunking
    chunks_10 = load_and_chunk_policies(overlap=10)
    print(f"Total chunks with overlap 10: {len(chunks_10)}")
    by_doc_10 = {}
    for c in chunks_10:
        by_doc_10[c["source"]] = by_doc_10.get(c["source"], 0) + 1
    for doc, cnt in sorted(by_doc_10.items()):
        print(f"  {doc}: {cnt} chunks")

    # 2. Retrieval check
    retriever_10 = RAGRetriever(chunks_10)
    top3 = retriever_10.retrieve("refund policy late delivery", k=3)
    print("\nTop 3 chunks for 'refund policy late delivery':")
    for r in top3:
        print(f"  Source: {r['source']}, Score: {r['score']:.4f}")

    # 3 & 4. Evaluation with overlap 10
    eval_10 = evaluate_retrieval_and_grounding(retriever_10, label="Overlap = 10")
    print(f"\nEvaluation (Overlap 10): Top-3 Doc: {eval_10['top_doc_matches']}/12, Fact in answer: {eval_10['fact_matches']}/12, Fallback: {eval_10['fallback_matches']}/2")

    # Overlap 0 evaluation
    chunks_0 = load_and_chunk_policies(overlap=0)
    print(f"\nTotal chunks with overlap 0: {len(chunks_0)}")
    retriever_0 = RAGRetriever(chunks_0)
    eval_0 = evaluate_retrieval_and_grounding(retriever_0, label="Overlap = 0")
    print(f"Evaluation (Overlap 0): Top-3 Doc: {eval_0['top_doc_matches']}/12, Fact in answer: {eval_0['fact_matches']}/12, Fallback: {eval_0['fallback_matches']}/2")
    if eval_0["failed_details"]:
        print("Failed questions under overlap 0:")
        for fd in eval_0["failed_details"]:
            print(f"  Question: '{fd['question']}' -> Expected fact: '{fd['expected_fact']}' not found in answer.")

    # 5 & 6. Run 7 Scenarios
    print("\n=== RUNNING 7 REFUND AGENT SCENARIOS ===")
    if os.path.exists("outputs/refund_ledger.jsonl"):
        os.remove("outputs/refund_ledger.jsonl")
    if os.path.exists("outputs/agent_runs.jsonl"):
        os.remove("outputs/agent_runs.jsonl")

    scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv")
    scenario_results = []

    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_oc = row["expected_outcome"]
        exp_amt = float(row["expected_refund_inr"])

        res = run_agent(msg, sid)

        # Extract refund amount
        actual_amt = 0.0
        if res["status"] == "REFUND":
            # issued refund
            for step in reversed(res["steps"]):
                if step["tool"] == "issue_refund":
                    actual_amt = float(step["observation"].get("refund_inr", 0.0))
                    break
        elif res["status"] == "ESCALATE":
            # escalated refund
            for step in reversed(res["steps"]):
                if step["tool"] == "escalate_to_human":
                    actual_amt = float(step["observation"].get("refund_inr", 0.0))
                    break

        scenario_results.append({
            "scenario_id": sid,
            "status": res["status"],
            "refund_amount": actual_amt,
            "expected_outcome": exp_oc,
            "expected_refund": exp_amt,
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"]
        })

        # Save to outputs/agent_runs.jsonl
        run_record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

        print(f"Scenario {sid}: status={res['status']} (exp={exp_oc}), refund=INR {actual_amt:.2f} (exp={exp_amt:.2f}), calls={res['llm_calls']}, tokens={res['tokens_used']}")

    # 7. Stop-Safely Demos
    print("\n=== STOP-SAFELY DEMONSTRATIONS ===")
    demo_token = run_agent("My order ORD-100001 reached me very late. I would like a refund.", "S1_demo_token", token_limit=200)
    print(f"Token limit demo (limit=200): status={demo_token['status']}, answer={demo_token['final_answer']}")

    demo_timeout = run_agent("My order ORD-100001 reached me very late. I would like a refund.", "S1_demo_timeout", time_limit_seconds=0)
    print(f"Timeout demo (limit=0s): status={demo_timeout['status']}, answer={demo_timeout['final_answer']}")

    # 8. Crash and Resume Demo on S1
    print("\n=== CRASH AND RESUME DEMO ON S1 ===")
    crash_run_id = "s1_crash_resume_demo"
    s1_msg = scenarios_df.loc[scenarios_df["scenario_id"] == "S1", "message"].values[0]
    try:
        run_agent(s1_msg, "S1", simulate_crash_step=2, run_id=crash_run_id)
    except RuntimeError as e:
        print(f"Simulated crash occurred as expected: {e}")

    resumed_result = resume_agent(crash_run_id)
    print(f"Resumed Run Status: {resumed_result['status']}")
    print(f"Resumed Final Answer: {resumed_result['final_answer']}")
    print(f"Tools executed across crash & resume: {resumed_result['tools_executed']}")

    return {
        "chunks_10": chunks_10,
        "eval_10": eval_10,
        "chunks_0": chunks_0,
        "eval_0": eval_0,
        "scenario_results": pd.DataFrame(scenario_results),
        "demo_token": demo_token,
        "demo_timeout": demo_timeout,
        "resumed_result": resumed_result
    }


if __name__ == "__main__":
    run_all_part_d()
