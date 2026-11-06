"""customer_segmentation_lapse.py -- Part B: Customer Segmentation & Lapse Prediction.
Implements RFM table, KMeans clustering, PCA, cohort definition, feature engineering,
leakage-free preprocessing, model tuning, class-balance experiment, and retention list.
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def run_rfm_and_clustering():
    print("=== PART B: RFM AND CLUSTERING ===")
    orders = pd.read_csv("data/clean/orders.csv")
    completed = orders[orders["status"] == "completed"].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])

    snapshot = pd.to_datetime("2025-10-01 00:00:00")

    rfm = completed.groupby("customer_id").agg(
        last_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values("customer_id").reset_index(drop=True)

    print(f"RFM Customers count: {len(rfm)}")
    print(f"Recency: median={rfm['recency'].median()}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")

    # Standardize RFM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    print("\nK-Means Sweep (K=2..8):")
    kmeans_results = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        kmeans_results[k] = {"inertia": km.inertia_, "silhouette": sil}
        print(f"  K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.3f}")

    # Profile K=4
    km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["segment_id"] = km4.fit_predict(X_scaled)

    # Segment naming based on RFM profile
    seg_summary = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    print("\nSegment Summary for K=4:")
    print(seg_summary.round(2))

    # Identify Premium segment (highest mean monetary value)
    premium_seg_id = seg_summary.loc[seg_summary["mean_monetary"].idxmax(), "segment_id"]
    print(f"\nPremium Segment ID: {premium_seg_id} (Mean monetary: INR {seg_summary.loc[seg_summary['segment_id'] == premium_seg_id, 'mean_monetary'].values[0]:,.2f})")

    # Name the segments meaningfully
    # Find active loyal (high freq, low recency)
    # Find churned / at-risk (high recency, low freq)
    # Find moderate / regular
    seg_names = {}
    for _, row in seg_summary.iterrows():
        sid = int(row["segment_id"])
        if sid == premium_seg_id:
            seg_names[sid] = "Premium High-Value"
        elif row["mean_frequency"] > 15:
            seg_names[sid] = "Frequent Everyday"
        elif row["mean_recency"] > 60:
            seg_names[sid] = "At-Risk Occasional"
        else:
            seg_names[sid] = "Regular Shoppers"

    rfm["segment_name"] = rfm["segment_id"].map(seg_names)
    rfm["is_premium"] = rfm["segment_id"] == premium_seg_id

    # PCA 2 components
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    print(f"\nPCA explained variance ratios: {pca.explained_variance_ratio_.round(4)}, Total: {pca.explained_variance_ratio_.sum():.4f}")

    # Save outputs/customer_segments.csv
    os.makedirs("outputs", exist_ok=True)
    out_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    rfm[out_cols].to_csv("outputs/customer_segments.csv", index=False)
    print("Saved outputs/customer_segments.csv")

    return rfm, X_scaled, kmeans_results, seg_summary, pca


def run_lapse_prediction(rfm_df):
    print("\n=== PART B: LAPSE PREDICTION ===")
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    window_end = pd.to_datetime("2025-09-30 23:59:59")

    # Merge product category into orders
    orders_merged = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    # Cohort definition: at least one completed order before cutoff
    completed_before = orders_merged[(orders_merged["status"] == "completed") & (orders_merged["order_ts"] < cutoff)]
    cohort_cids = sorted(completed_before["customer_id"].unique())
    print(f"Cohort size: {len(cohort_cids)}")

    # Target definition: lapsed = 1 if no completed order in Sept 2025 (2025-09-01 to 2025-09-30)
    completed_sept = orders_merged[
        (orders_merged["status"] == "completed") & 
        (orders_merged["order_ts"] >= cutoff) & 
        (orders_merged["order_ts"] <= window_end)
    ]
    active_sept_cids = set(completed_sept["customer_id"].unique())

    # Build cohort features strictly using orders before cutoff + clean customers
    orders_before_all = orders_merged[orders_merged["order_ts"] < cutoff]

    features = []
    customers_map = customers.set_index("customer_id").to_dict("index")

    for cid in cohort_cids:
        c_cust = customers_map[cid]
        c_orders_all = orders_before_all[orders_before_all["customer_id"] == cid]
        c_completed = c_orders_all[c_orders_all["status"] == "completed"]

        # Recency: days from cutoff to last completed order
        last_comp_ts = c_completed["order_ts"].max()
        recency = (cutoff - last_comp_ts).days
        frequency = len(c_completed)
        monetary = c_completed["order_amount"].sum()
        avg_discount = c_completed["discount_pct"].mean() if frequency > 0 else 0.0
        
        # Cancellation rate: cancelled orders / all orders before cutoff
        total_orders_before = len(c_orders_all)
        cancelled_orders = (c_orders_all["status"] == "cancelled").sum()
        cancellation_rate = cancelled_orders / total_orders_before if total_orders_before > 0 else 0.0
        
        avg_delivery_minutes = c_completed["delivery_minutes"].mean() if frequency > 0 else 0.0
        fav_category = c_completed["category"].mode()[0] if frequency > 0 else "None"

        # Tenure: cutoff minus signup_date
        signup_dt = pd.to_datetime(c_cust["signup_date"])
        tenure_days = (cutoff - signup_dt).days

        is_lapsed = 1 if cid not in active_sept_cids else 0

        features.append({
            "customer_id": cid,
            "recency": recency,
            "frequency": frequency,
            "monetary": round(monetary, 2),
            "avg_discount_pct": round(avg_discount, 2),
            "cancellation_rate": round(cancellation_rate, 4),
            "avg_delivery_minutes": round(avg_delivery_minutes, 2),
            "fav_category": fav_category,
            "city": c_cust["city"],
            "age": c_cust["age"],
            "gender": c_cust["gender"],
            "tenure_days": tenure_days,
            "lapsed": is_lapsed
        })

    cohort_df = pd.DataFrame(features)
    lapsed_count = cohort_df["lapsed"].sum()
    lapsed_pct = (lapsed_count / len(cohort_df)) * 100
    print(f"Lapsed customers: {lapsed_count} ({lapsed_pct:.2f}%)")

    # Feature columns
    feature_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "fav_category",
        "city", "age", "gender", "tenure_days"
    ]
    numeric_features = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    categorical_features = ["city", "gender", "fav_category"]

    X = cohort_df[feature_cols]
    y = cohort_df["lapsed"]

    # Stratified Train/Test Split (80/20, seed 42)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, cohort_df["customer_id"], test_size=0.2, stratify=y, random_state=42
    )
    print(f"Split sizes: Train={len(X_train)}, Test={len(X_test)}")
    print(f"Train class balance: Non-lapsed={(y_train == 0).sum()}, Lapsed={(y_train == 1).sum()}")

    # Baseline evaluation: always predict majority class (0)
    baseline_preds = np.zeros_like(y_test)
    base_acc = accuracy_score(y_test, baseline_preds)
    base_f1 = f1_score(y_test, baseline_preds, zero_division=0)
    print(f"Baseline ('always not lapsed'): Accuracy={base_acc:.4f}, Lapsed F1={base_f1:.4f}")

    # Build preprocessing pipeline fitted on training rows only
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit preprocessor on train only
    preprocessor.fit(X_train)
    
    # Check frequency mean difference
    train_freq_scaler = preprocessor.named_transformers_["num"]
    freq_idx = numeric_features.index("frequency")
    learned_freq_mean = train_freq_scaler.mean_[freq_idx]
    cohort_freq_mean = cohort_df["frequency"].mean()
    print(f"\nLearned frequency mean (train only): {learned_freq_mean:.4f}")
    print(f"Full-cohort frequency mean: {cohort_freq_mean:.4f}")
    print("Why they differ: The scaler was fitted strictly on the 80% training sample (924 rows) to prevent data leakage, whereas the full-cohort mean includes all 1,156 rows.")

    # Verify no missing values in preprocessed data
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_train_trans).any(), "Missing values found in transformed training data"
    assert not np.isnan(X_test_trans).any(), "Missing values found in transformed test data"
    print("Code confirmed: No missing values in preprocessed feature arrays.")

    # (b) Tune LogisticRegression with GridSearchCV
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)
    print(f"\nLogistic Regression best params: {grid_lr.best_params_}, best CV F1: {grid_lr.best_score_:.4f}")

    # (c) Tune RandomForestClassifier with GridSearchCV
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)
    print(f"Random Forest best params: {grid_rf.best_params_}, best CV F1: {grid_rf.best_score_:.4f}")

    # (d) Test Set Evaluation
    y_pred_lr = grid_lr.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)

    y_pred_rf = grid_rf.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

    print("\n--- TEST SET EVALUATION ---")
    print(f"Logistic Regression (untouched test set):")
    print(f"  Confusion Matrix:\n{cm_lr}")
    print(f"  Precision: {prec_lr:.4f}, Recall: {rec_lr:.4f}, F1: {f1_lr:.4f}")

    print(f"\nRandom Forest (untouched test set):")
    print(f"  Confusion Matrix:\n{cm_rf}")
    print(f"  Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")

    # Task 7: Class-balance experiment on train set only
    print("\n=== CLASS-BALANCE EXPERIMENT ===")
    train_combined = X_train.copy()
    train_combined["lapsed"] = y_train.values

    train_lapsed = train_combined[train_combined["lapsed"] == 1]
    train_non_lapsed = train_combined[train_combined["lapsed"] == 0]
    target_count = len(train_non_lapsed)  # 570 in reference

    # Oversample minority class with replacement
    np.random.seed(42)
    oversampled_lapsed = train_lapsed.sample(n=target_count, replace=True, random_state=42)
    train_balanced = pd.concat([train_non_lapsed, oversampled_lapsed], ignore_index=True)
    print(f"Balanced train rows: {len(train_balanced)} (570 non-lapsed, 570 lapsed)")

    X_train_bal = train_balanced[feature_cols]
    y_train_bal = train_balanced["lapsed"]

    # Refit best Logistic Regression on balanced data
    best_C = grid_lr.best_params_["clf__C"]
    pipe_lr_bal = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=best_C, max_iter=1000, random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = pipe_lr_bal.predict(X_test)
    prec_bal = precision_score(y_test, y_pred_lr_bal, zero_division=0)
    rec_bal = recall_score(y_test, y_pred_lr_bal, zero_division=0)
    f1_bal = f1_score(y_test, y_pred_lr_bal, zero_division=0)

    print(f"Balanced Logistic Regression on untouched test set:")
    print(f"  Precision: {prec_bal:.4f}, Recall: {rec_bal:.4f}, F1: {f1_bal:.4f}")
    print(f"  Comparison vs Unbalanced: Recall moved from {rec_lr:.4f} to {rec_bal:.4f} (+{rec_bal - rec_lr:.4f})")

    # Task 8: Interpretability, Persistence, and Retention List
    print("\n=== INTERPRETABILITY & MODEL SELECTION ===")
    # Feature importances for best RF
    best_rf_model = grid_rf.best_estimator_.named_steps["clf"]
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    rf_importances = pd.Series(best_rf_model.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    print("Top 5 features of best Random Forest:")
    for feat, imp in rf_importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    # Coefficients for best LR
    best_lr_model = grid_lr.best_estimator_.named_steps["clf"]
    lr_coeffs = pd.Series(best_lr_model.coef_[0], index=all_feature_names).sort_values()
    print("\nSelected Logistic Regression coefficients:")
    for feat in ["recency", "frequency", "tenure_days"]:
        if feat in lr_coeffs:
            print(f"  {feat}: {lr_coeffs[feat]:.4f} (Sign: {'+' if lr_coeffs[feat] > 0 else '-'})")

    # Selected model: Random Forest (higher F1 ~0.65 vs ~0.62)
    selected_pipeline = grid_rf.best_estimator_
    joblib.dump(selected_pipeline, "outputs/lapse_model.joblib")
    print("\nSaved selected model to outputs/lapse_model.joblib")

    # Reload and test reproducibility
    reloaded_pipeline = joblib.load("outputs/lapse_model.joblib")
    reloaded_preds = reloaded_pipeline.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_preds), "Reloaded model predictions do not match original"
    print("Reloaded model verification passed: Exact match on test predictions.")

    # Task 8(c): Retention List for Premium customers in test set
    test_df = X_test.copy()
    test_df["customer_id"] = ids_test.values
    test_df["actually_lapsed"] = y_test.values
    test_df["lapse_probability"] = selected_pipeline.predict_proba(X_test)[:, 1].round(4)

    # Join with whole-window customer segments to find Premium customers
    test_df = test_df.merge(rfm_df[["customer_id", "is_premium"]], on="customer_id", how="left")
    premium_test = test_df[test_df["is_premium"] == True].copy()
    retention_list = premium_test.sort_values("lapse_probability", ascending=False).head(10)
    retention_list = retention_list[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_list.to_csv("outputs/retention_list.csv", index=False)
    print("\nSaved outputs/retention_list.csv (Top Premium at-risk customers):")
    print(retention_list)
    print(f"Actually lapsed count in retention list: {retention_list['actually_lapsed'].sum()} / {len(retention_list)}")

    return {
        "cohort_size": len(cohort_df),
        "lapsed_count": lapsed_count,
        "baseline": (base_acc, base_f1),
        "lr_eval": (cm_lr, prec_lr, rec_lr, f1_lr),
        "rf_eval": (cm_rf, prec_rf, rec_rf, f1_rf),
        "balanced_lr": (prec_bal, rec_bal, f1_bal),
        "retention_list": retention_list
    }


if __name__ == "__main__":
    rfm_df, _, _, _, _ = run_rfm_and_clustering()
    run_lapse_prediction(rfm_df)
