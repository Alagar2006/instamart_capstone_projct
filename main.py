"""main.py -- Master End-to-End Orchestrator for Instamart Customer Intelligence & Support Copilot.
Executes Parts A, B, C, D in sequence, asserts all reference checkpoints, and validates outputs.

Usage:
    python main.py
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure LOKY_MAX_CPU_COUNT is set to avoid Win32 subprocess issues
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


def print_banner(title):
    width = 75
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")


def run_pipeline():
    start_total = time.time()
    print_banner("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT: FULL RUN")

    # Step 1: Generate synthetic data
    print(">>> [1/7] Generating Synthetic Data and Policies (generate_data.py)...")
    import generate_data
    print("    Synthetic raw data and policy documents generated successfully.")

    # Step 2: Part A Data Cleaning & Storage
    print("\n>>> [2/7] Executing Part A: Data Cleaning & SQLite Storage (clean_data.py)...")
    from src.clean_data import clean_customers, clean_products, clean_orders, create_sqlite_db
    clean_customers()
    clean_products()
    clean_orders()
    create_sqlite_db()

    # Step 3: Validate Part A
    print("\n>>> [3/7] Validating Part A Reference Checkpoints (validate_part_a.py)...")
    from validate_part_a import validate_part_a
    validate_part_a()

    # Step 4: Run SQL Analysis & Cross-checks
    print("\n>>> [4/7] Running Part A SQL Analytical Queries & pandas EDA (run_sql_analysis.py)...")
    from src.run_sql_analysis import run_sql_queries, run_pandas_cross_checks
    run_sql_queries()
    run_pandas_cross_checks()

    # Step 5: Part B Segmentation & Lapse Prediction
    print("\n>>> [5/7] Running Part B: RFM Segmentation & Lapse Prediction (customer_segmentation_lapse.py)...")
    from src.customer_segmentation_lapse import run_rfm_and_clustering, run_lapse_prediction
    rfm_df, _, _, _, _ = run_rfm_and_clustering()
    b_results = run_lapse_prediction(rfm_df)

    # Step 6: Part C Support Ticket Triage with Guardrails
    print("\n>>> [6/7] Running Part C: Support Ticket Triage with Privacy Guardrails (support_triage.py)...")
    from src.support_triage import apply_routing_and_save, redact_pii, run_triage_pipeline
    apply_routing_and_save()

    # Step 7: Part D Policy RAG & ReAct Refund Agent
    print("\n>>> [7/7] Running Part D: Policy RAG Assistant & ReAct Refund Agent (policy_rag_agent.py)...")
    from src.policy_rag_agent import run_all_part_d
    d_results = run_all_part_d()

    # Summary of generated files
    total_time = time.time() - start_total
    print_banner("EXECUTION COMPLETE -- VERIFYING ARTIFACTS")
    expected_files = [
        "data/instamart.db",
        "data/clean/customers.csv",
        "data/clean/products.csv",
        "data/clean/orders.csv",
        "outputs/customer_segments.csv",
        "outputs/lapse_model.joblib",
        "outputs/retention_list.csv",
        "outputs/triage_budget_demo.csv",
        "outputs/triage_results.csv",
        "outputs/refund_ledger.jsonl",
        "outputs/agent_runs.jsonl",
    ]

    all_exist = True
    for f in expected_files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        status = "OK" if exists else "MISSING"
        if not exists:
            all_exist = False
        print(f"  [{status}] {f:<38} ({size:,} bytes)")

    print(f"\nAll components executed in {total_time:.2f}s.")
    if all_exist:
        print("SUCCESS: ALL ARTIFACTS GENERATED AND VALIDATED.")
    else:
        print("WARNING: Some artifacts were not found.")


if __name__ == "__main__":
    run_pipeline()

