"""run_sql_analysis.py -- Executes SQL queries and pandas cross-checks/EDA for Part A.
"""
import re
import sqlite3
import pandas as pd
import numpy as np


def parse_sql_file(sql_path="sql/queries.sql"):
    with open(sql_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by semicolon followed by newline or end of string
    raw_statements = [stmt.strip() for stmt in content.split(";") if stmt.strip()]
    queries = []
    for stmt in raw_statements:
        # Strip comments
        lines = [line for line in stmt.splitlines() if not line.strip().startswith("--")]
        clean_stmt = "\n".join(lines).strip()
        if clean_stmt:
            queries.append(stmt)
    return queries


def run_sql_queries(db_path="data/instamart.db", sql_path="sql/queries.sql"):
    conn = sqlite3.connect(db_path)
    queries = parse_sql_file(sql_path)
    results = {}

    print(f"=== SQL QUERY RESULTS ({len(queries)} queries found) ===")
    for i, q in enumerate(queries, 1):
        df = pd.read_sql_query(q, conn)
        results[f"Q{i}"] = df
        print(f"\n--- Query {i} ---")
        if i == 1:
            print(f"Row count: {len(df)} customers")
        elif i == 2:
            print(f"Row count: {len(df)}, Total revenue: INR {df['revenue'].sum():,.2f}")
            max_row = df.loc[df["revenue"].idxmax()]
            print(f"Largest cell: {max_row['city']} / {max_row['category']} at INR {max_row['revenue']:,.2f}")
        elif i == 3:
            print(f"Above average price products count: {len(df)}")
        elif i == 4:
            print(f"Customers with no completed orders: {len(df)}")
        elif i == 5:
            avg_val = pd.read_sql_query("WITH cs AS (SELECT customer_id, SUM(order_amount) AS total_spend FROM orders WHERE status = 'completed' GROUP BY customer_id) SELECT ROUND(AVG(total_spend), 2) AS avg_s FROM cs", conn).iloc[0, 0]
            print(f"Above average spend customers count: {len(df)}, Average spend: INR {avg_val:,.2f}")
        elif i == 6:
            top_per_city = df[df["rnk"] == 1]
            print("Top customer per city:")
            for _, r in top_per_city.iterrows():
                print(f"  {r['city']}: {r['customer_id']} ({r['total_spend']:,.2f})")
            diff_count = (df["rnk"] != df["row_num"]).sum()
            print(f"Rows where RANK() != ROW_NUMBER(): {diff_count}")
        elif i == 7:
            print(f"Daily rows count: {len(df)}, Final cumulative revenue: INR {df.iloc[-1]['cumulative_revenue']:,.2f}")

    conn.close()
    return results


def run_pandas_cross_checks():
    print("\n=== PANDAS CROSS-CHECK & EXPLORATORY DATA ANALYSIS ===")
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")

    completed = orders[orders["status"] == "completed"].copy()
    merged = completed.merge(customers, on="customer_id").merge(products, on="product_id")

    # (a) Rebuild Q2 with pivot_table
    pivot = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    ).round(2)
    grand_total = pivot.loc["Total", "Total"]
    print(f"Pivot table grand total: INR {grand_total:,.2f}")

    # Named aggregation per city
    city_summary = merged.groupby("city").agg(
        sum_amount=("order_amount", "sum"),
        mean_amount=("order_amount", "mean"),
        count_orders=("order_amount", "count")
    ).round(2)
    print("\nCity summary (completed orders):")
    print(city_summary)

    # (b) Completed order_amount distribution & statistics
    amt = completed["order_amount"]
    mean_amt = amt.mean()
    median_amt = amt.median()
    skew_amt = amt.skew()
    q1 = amt.quantile(0.25)
    q3 = amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_below = (amt < lower_fence).sum()
    outliers_above = (amt > upper_fence).sum()

    print("\nOrder Amount Statistics (Completed Orders):")
    print(f"  Mean: {mean_amt:.2f}")
    print(f"  Median: {median_amt:.2f}")
    print(f"  Skewness: {skew_amt:.2f}")
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"  Lower fence (1.5xIQR): {lower_fence:.2f} (Outliers below: {outliers_below})")
    print(f"  Upper fence (1.5xIQR): {upper_fence:.2f} (Outliers above: {outliers_above})")

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed[corr_cols].corr()
    print("\nCorrelation matrix over completed orders:")
    print(corr_matrix.round(4))

    return {
        "grand_total": grand_total,
        "city_summary": city_summary,
        "stats": {
            "mean": mean_amt,
            "median": median_amt,
            "skewness": skew_amt,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outliers_below": outliers_below,
            "outliers_above": outliers_above
        },
        "corr_matrix": corr_matrix
    }


if __name__ == "__main__":
    run_sql_queries()
    run_pandas_cross_checks()
