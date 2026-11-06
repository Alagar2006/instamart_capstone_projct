"""clean_data.py -- Data cleaning pipeline for Instamart Copilot Part A.
Cleans customers, products, orders raw data, writes data/clean/*.csv and data/instamart.db.
"""
import os
import sqlite3
import pandas as pd
import numpy as np


def clean_customers(raw_path="data/raw/customers.csv", clean_path="data/clean/customers.csv"):
    df = pd.read_csv(raw_path, keep_default_na=False)
    
    # Audit tracking
    initial_shape = df.shape
    blank_age = (df["age"] == "").sum()
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    out_of_bounds_age = ((age_numeric < 10) | (age_numeric > 100)).sum()
    distinct_raw_cities = df["city"].nunique()
    non_blank_gender = df[df["gender"] != ""]["gender"]
    distinct_raw_genders = non_blank_gender.nunique()
    blank_gender = (df["gender"] == "").sum()
    
    # 1. Standardise city: strip spaces and apply title case
    df["city"] = df["city"].str.strip().str.title()
    
    # 2. Standardise gender
    def map_gender(val):
        g = val.strip().lower()
        if g in ["male", "m"]:
            return "male"
        elif g in ["female", "f"]:
            return "female"
        return "unknown"
        
    df["gender"] = df["gender"].apply(map_gender)
    
    # 3. Clean age: treat outside 10-100 as missing, fill missing with city median
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df.groupby("city")["age"].transform(lambda s: s.fillna(s.median())).astype(int)
    
    # 4. Standardise signup_date
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.strftime("%Y-%m-%d")
    
    clean_df = df[["customer_id", "city", "signup_date", "age", "gender"]]
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    clean_df.to_csv(clean_path, index=False)
    
    audit_stats = {
        "initial_shape": initial_shape,
        "blank_age": blank_age,
        "out_of_bounds_age": out_of_bounds_age,
        "distinct_raw_cities": distinct_raw_cities,
        "distinct_raw_genders": distinct_raw_genders,
        "blank_gender": blank_gender,
        "final_shape": clean_df.shape,
        "city_counts": clean_df["city"].value_counts().to_dict(),
        "gender_counts": clean_df["gender"].value_counts().to_dict(),
        "age_min": clean_df["age"].min(),
        "age_max": clean_df["age"].max(),
        "missing_age": clean_df["age"].isna().sum(),
    }
    return clean_df, audit_stats


def clean_products(raw_path="data/raw/products.csv", clean_path="data/clean/products.csv"):
    df = pd.read_csv(raw_path, keep_default_na=False)
    df["price_inr"] = pd.to_numeric(df["price_inr"])
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    return df, {"shape": df.shape}


def clean_orders(raw_orders_path="data/raw/orders.csv", products_path="data/clean/products.csv", clean_path="data/clean/orders.csv"):
    orders = pd.read_csv(raw_orders_path, keep_default_na=False)
    products = pd.read_csv(products_path, keep_default_na=False)
    
    initial_rows = len(orders)
    exact_duplicates = orders.duplicated().sum()
    pending_rows = (orders["status"] == "pending").sum()
    distinct_order_ids = orders["order_id"].nunique()
    non_numeric_amount = pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()
    deliv_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative_delivery = (deliv_numeric < 0).sum()
    blank_delivery = (orders["delivery_minutes"] == "").sum()
    
    # 1. Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    
    # 2. Remove exact duplicate rows
    orders = orders.drop_duplicates()
    
    # 3. Keep latest order_ts per order_id
    orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
    
    # 4. Convert order_amount to numeric, fill missing from product price calculation
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
    calc_amount = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100.0)).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(calc_amount)
    
    # 5. Delivery minutes: negative to missing. For completed orders, fill missing with category median among completed.
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    
    completed_mask = orders["status"] == "completed"
    medians = orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(medians)
    
    # Format order_ts as text YYYY-MM-DD HH:MM:SS
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean_orders_df = orders[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]]
    
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    clean_orders_df.to_csv(clean_path, index=False)
    
    audit_stats = {
        "initial_rows": initial_rows,
        "exact_duplicates": exact_duplicates,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amount": non_numeric_amount,
        "negative_delivery": negative_delivery,
        "blank_delivery": blank_delivery,
        "final_rows": len(clean_orders_df),
        "completed_count": (clean_orders_df["status"] == "completed").sum(),
        "cancelled_count": (clean_orders_df["status"] == "cancelled").sum(),
        "missing_amount": clean_orders_df["order_amount"].isna().sum(),
        "missing_delivery_completed": clean_orders_df.loc[clean_orders_df["status"] == "completed", "delivery_minutes"].isna().sum(),
        "missing_delivery_cancelled": clean_orders_df.loc[clean_orders_df["status"] == "cancelled", "delivery_minutes"].isna().sum(),
        "completed_order_amount_sum": clean_orders_df.loc[clean_orders_df["status"] == "completed", "order_amount"].sum()
    }
    return clean_orders_df, audit_stats


def create_sqlite_db(db_path="data/instamart.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    cust = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    
    cust.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)
    
    conn.close()
    print(f"SQLite database created at {db_path} with tables customers, products, orders.")


if __name__ == "__main__":
    print("Cleaning customers...")
    _, cust_stats = clean_customers()
    print("Cleaning products...")
    _, prod_stats = clean_products()
    print("Cleaning orders...")
    _, order_stats = clean_orders()
    print("Storing tables in SQLite...")
    create_sqlite_db()
    print("Cleaning and storage completed successfully.")
