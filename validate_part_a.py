"""validate_part_a.py -- Validation script for Part A checkpoints.
Asserts reference checkpoints from tasks 3 and 4 and prints ALL CHECKPOINTS PASSED.
"""
import sqlite3
import pandas as pd
import numpy as np


def validate_part_a():
    # Load clean data
    cust = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    orders = pd.read_csv("data/clean/orders.csv")

    # Connect to SQLite database
    conn = sqlite3.connect("data/instamart.db")
    db_cust = pd.read_sql_query("SELECT * FROM customers", conn)
    db_orders = pd.read_sql_query("SELECT * FROM orders", conn)
    db_products = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()

    # Checkpoint 1: Clean customer city counts
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184
    }
    actual_cities = cust["city"].value_counts().to_dict()
    assert actual_cities == expected_cities, f"City counts mismatch: {actual_cities} vs {expected_cities}"

    # Checkpoint 2: Clean customer gender counts
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    actual_genders = cust["gender"].value_counts().to_dict()
    assert actual_genders == expected_genders, f"Gender counts mismatch: {actual_genders} vs {expected_genders}"

    # Checkpoint 3: Customer age constraints (0 missing, ages between 18 and 60)
    assert cust["age"].isna().sum() == 0, "Missing ages found in customers"
    assert cust["age"].min() >= 18 and cust["age"].max() <= 60, f"Age out of bounds: {cust['age'].min()} to {cust['age'].max()}"

    # Checkpoint 4: Total clean orders count and distinct order_ids
    assert len(orders) == 12309, f"Order rows mismatch: {len(orders)} != 12309"
    assert orders["order_id"].nunique() == 12309, f"Order IDs not unique: {orders['order_id'].nunique()}"

    # Checkpoint 5: Order status counts (11,110 completed, 1,199 cancelled, 0 pending)
    assert (orders["status"] == "completed").sum() == 11110, f"Completed orders mismatch: {(orders['status'] == 'completed').sum()}"
    assert (orders["status"] == "cancelled").sum() == 1199, f"Cancelled orders mismatch: {(orders['status'] == 'cancelled').sum()}"
    assert (orders["status"] == "pending").sum() == 0, "Pending orders still present in clean orders"

    # Checkpoint 6: Missing order_amount is 0
    assert orders["order_amount"].isna().sum() == 0, f"Missing order_amount count: {orders['order_amount'].isna().sum()}"

    # Checkpoint 7: Missing delivery_minutes among completed is 0 and among cancelled is 1199
    completed_orders = orders[orders["status"] == "completed"]
    cancelled_orders = orders[orders["status"] == "cancelled"]
    assert completed_orders["delivery_minutes"].isna().sum() == 0, f"Missing delivery_minutes for completed: {completed_orders['delivery_minutes'].isna().sum()}"
    assert cancelled_orders["delivery_minutes"].isna().sum() == 1199, f"Missing delivery_minutes for cancelled: {cancelled_orders['delivery_minutes'].isna().sum()}"

    # Checkpoint 8: Sum of order_amount over completed orders is INR 2,998,668.80
    completed_revenue = completed_orders["order_amount"].sum()
    assert np.isclose(completed_revenue, 2998668.80, atol=0.01), f"Completed revenue mismatch: {completed_revenue} != 2998668.80"

    # Checkpoint 9: Verify DB tables match clean CSVs
    assert len(db_cust) == len(cust), "DB customers row count mismatch"
    assert len(db_orders) == len(orders), "DB orders row count mismatch"
    assert len(db_products) == len(products), "DB products row count mismatch"

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate_part_a()
