# Instamart Customer Intelligence & Support Copilot

An enterprise-grade Customer Intelligence and Support Copilot built for Swiggy Instamart across four integrated modules: an auditable data foundation, behavioral RFM segmentation and lapse prediction, an automated support ticket triage engine with privacy guardrails, and a grounded policy RAG assistant paired with an autonomous ReAct refund agent.

---

## Overview

Swiggy Instamart operates ultra-fast grocery and essentials delivery across major Indian metropolitan hubs (Chennai, Mumbai, Bengaluru, Delhi, Hyderabad, and Pune). Operating at scale introduces three critical operational challenges:
1. **Unreliable Raw Data**: Upstream order exports contain duplicates, corrupted text, unparseable numeric values, and negative delivery times.
2. **Customer Churn Visibility**: Marketing and retention teams struggle to proactively identify which customers are on the verge of lapsing before churn occurs.
3. **Overwhelmed Customer Support**: Support desks handle thousands of unstructured tickets containing sensitive PII (emails, phone numbers, order IDs) across mixed categories (fraud, billing, technical, delivery).

This repository implements the **Customer Intelligence & Support Copilot** as a production-ready, modular system solving these issues:
- **Part A (Data Foundation)**: Cleans, audits, validates, and models raw customer and order data into a SQLite database with seven analytical SQL queries and pandas cross-checks.
- **Part B (Customer Segmentation & Lapse Prediction)**: Builds an RFM framework, evaluates K-Means clustering ($K=2..8$), projects segments via PCA, trains an anti-leakage lapse prediction model achieving test $F_1 \ge 0.55$, conducts class-balancing experiments, and generates an actionable retention list.
- **Part C (Support Ticket Triage with Guardrails)**: Implements deterministic input guardrails, regex-based PII redaction, resilient retry handling with exponential backoff and jitter, robust JSON/keyword output validation, token budgeting, and priority routing.
- **Part D (Policy RAG Assistant & ReAct Refund Agent)**: Chunks store policies, indexes text via TF-IDF vectorization with cosine similarity retrieval, benchmarks retrieval grounding, defines tool schemas, executes a ReAct agent loop with execution guards (timeout, token budget, max iterations), implements persistent checkpointing for crash-and-resume workflows, and contrasts ReAct with plan-and-execute architectures.

---

## Reproduce

You can run the entire pipeline end-to-end with a single command:

```bash
python main.py
```

To run the automated test suite covering all tools, RAG retrieval, ReAct agent scenarios, and triage guardrails:

```bash
python test_agent.py
```

Alternatively, follow these individual numbered shell commands to run each module step-by-step:

```bash
# 1. Clone the repository and enter the workspace
git clone https://github.com/<user>/<repo>.git
cd <repo>

# 2. Create an isolated virtual environment and activate it
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
# source venv/bin/activate

# 3. Upgrade pip and install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate deterministic synthetic data and store policies
python generate_data.py

# 5. Run Part A: Data cleaning, CSV exports, and SQLite database creation
python src/clean_data.py

# 6. Validate Part A reference checkpoints
python validate_part_a.py

# 7. Run Part A: Analytical SQL queries and pandas exploratory cross-checks
python src/run_sql_analysis.py

# 8. Run Part B: RFM segmentation, K-Means sweep, lapse model training, and retention list
python src/customer_segmentation_lapse.py

# 9. Run Part C: Ticket PII redaction, zero/few-shot evaluations, budget demo, and routing
python src/support_triage.py

# 10. Run Part D: Policy chunking, TF-IDF RAG evaluation, ReAct refund agent, safe-stops, and crash-resume demo
python src/policy_rag_agent.py

# 11. Run Unit and Integration Test Suite
python test_agent.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit
The raw tables (`customers.csv`, `products.csv`, `orders.csv`) were audited for structural shape, datatypes, and missing/corrupted values:

| Table | Shape | Columns & Data Types | Missing Values Count (%) |
| :--- | :--- | :--- | :--- |
| **customers.csv** | 1,200 rows, 5 cols | `customer_id` (str), `city` (str), `signup_date` (str), `age` (str/int), `gender` (str) | `age`: 61 (5.08%), `gender`: 30 (2.50%), `city`: 0 (0.0%), `signup_date`: 0 (0.0%) |
| **products.csv** | 40 rows, 4 cols | `product_id` (str), `product_name` (str), `category` (str), `price_inr` (int) | All columns: 0 (0.0%) |
| **orders.csv** | 13,084 rows, 9 cols | `order_id` (str), `customer_id` (str), `product_id` (str), `quantity` (int), `order_ts` (str), `status` (str), `discount_pct` (int), `order_amount` (str/float), `delivery_minutes` (str/int) | `delivery_minutes`: 2,541 (19.42%), `order_amount`: 120 unparseable (0.92%) |

#### Specific Audit Findings
| Metric / Anomaly | Audit Count | Reference Checkpoint | Status |
| :--- | :--- | :--- | :--- |
| Raw `orders.csv` rows | 13,084 | 13,084 | Matched |
| Exact duplicate order rows | 119 | 119 | Matched |
| Rows with status `pending` | 656 | 656 | Matched |
| Distinct `order_id` values | 12,309 | 12,309 | Matched |
| Non-numeric `order_amount` entries | 120 | 120 | Matched |
| Negative `delivery_minutes` (impossible) | 106 | 106 | Matched |
| Blank `delivery_minutes` entries | 2,541 | 2,541 | Matched |
| Customers with blank `age` | 61 | 61 | Matched |
| Customers with `age` outside 10–100 | 6 | 6 | Matched |
| Distinct raw city spellings (casing/spaces) | 24 | 24 | Matched |
| Distinct non-blank raw gender spellings | 10 | 10 | Matched |
| Blank `gender` entries | 30 | 30 | Matched |

### 2. Cleaning Rules & Transformations
Cleaning was executed deterministically via `src/clean_data.py`:
1. **Customer Standardization**:
   - Stripped leading/trailing whitespaces and applied `.title()` to `city`, collapsing 24 variations into exactly 6 cities: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
   - Mapped 10 gender spellings to `male` (560), `female` (610), or `unknown` (30).
   - Treated ages outside 10–100 (6 values) as missing, then imputed all 67 missing ages using the median age of the customer's city via `groupby("city")["age"].transform("median")`. Post-cleaning, 0 missing ages remain, ranging from 18 to 60.
   - Converted `signup_date` to standard ISO format (`YYYY-MM-DD`).
2. **Order Cleaning & Imputation**:
   - Converted `order_ts` to datetime (`YYYY-MM-DD HH:MM:SS`).
   - Deduplicated 119 exact duplicate records.
   - Resolved multi-record orders by keeping only the record with the latest `order_ts` per `order_id`, eliminating all 656 `pending` status rows and leaving 12,309 unique orders.
   - Coerced 120 non-numeric `order_amount` entries to NaN and recalculated from product catalog prices: $\text{round}(\text{price\_inr} \times \text{quantity} \times (1 - \text{discount\_pct}/100), 2)$.
   - Replaced 106 negative delivery times with NaN. For completed orders, imputed missing delivery times with the median delivery duration of that product category among completed orders (692 values imputed). For cancelled orders, left all 1,199 delivery durations as missing because cancelled orders were never delivered.
   - Completed orders total 11,110 (revenue: INR 2,998,668.80) and cancelled orders total 1,199.

### 3. Database Storage & Checkpoint Verification
Cleaned tables were saved to `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`, and ingested into SQLite database `data/instamart.db` under tables `customers`, `products`, `orders`.
Running `python validate_part_a.py` asserts all 9 checkpoints and prints:
```
ALL CHECKPOINTS PASSED
```

### 4. SQL Analysis Results (`sql/queries.sql`)
All seven queries run directly against `data/instamart.db`:

| Query | Description | SQL Concepts | Output Summary |
| :--- | :--- | :--- | :--- |
| **Q1** | High-value spenders (`order_amount > 500`, total spend $> 8000$) | `WHERE`, `GROUP BY`, `HAVING` | **30 customers** identified |
| **Q2** | Completed revenue by city and product category | 3-table `JOIN` (`orders`, `customers`, `products`) | **30 rows**, grand total **INR 2,998,668.80**; largest cell is **Bengaluru / Electronics** at **INR 233,909.45** |
| **Q3** | Products priced above average catalog price | Scalar subquery `WHERE price_inr > (SELECT AVG(...))` | **11 products** |
| **Q4** | Customers who have never placed a completed order | `LEFT JOIN` on `status = 'completed'` `WHERE order_id IS NULL` | **25 customers** |
| **Q5** | Customers with spend above average customer spend | Common Table Expression (CTE) | **550 customers**; average completed spend is **INR 2,552.06** |
| **Q6** | City-wise completed spend rankings | Window functions `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()` | Top per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). **Exactly 8 rows have RANK() != ROW_NUMBER()**. |
| **Q7** | Daily completed revenue and cumulative running total | Window function `SUM(...) OVER (ORDER BY day)` | **183 daily rows** (2025-04-01 to 2025-09-30); final cumulative revenue is **INR 2,998,668.80** |

*Tie-breaking explanation for Q6*: `RANK()` assigns identical ranks to tied values and skips subsequent positions, whereas `ROW_NUMBER()` assigns strictly distinct sequential integers arbitrarily among ties, causing 8 rows to diverge where customer spend totals are identical.

### 5. pandas Cross-Check and Exploratory Data Analysis
- **Cross-Check**: A pivot table generated via `pd.pivot_table(merged, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)` confirmed a grand total of exactly **INR 2,998,668.80**, matching SQL Q2.
- **City Aggregation**:
  - Bengaluru: Sum INR 523,643.40, Mean INR 276.47, Count 1,894
  - Mumbai: Sum INR 479,058.50, Mean INR 297.00, Count 1,613
  - Chennai: Sum INR 463,629.15, Mean INR 272.72, Count 1,700
  - Delhi: Sum INR 502,829.80, Mean INR 243.62, Count 2,064
  - Hyderabad: Sum INR 528,589.90, Mean INR 284.95, Count 1,855
  - Pune: Sum INR 500,918.05, Mean INR 252.48, Count 1,984
- **Order Amount Distribution & Outlier Analysis**:
  - Mean: INR 269.91, Median: INR 135.00, Skewness: 4.61 (strong positive skew).
  - Quartiles: $Q_1 = \text{INR } 64.80$, $Q_3 = \text{INR } 297.00$, $\text{IQR} = \text{INR } 232.20$.
  - Fences ($1.5 \times \text{IQR}$): Lower fence = INR -283.50 (0 outliers below), Upper fence = INR 645.30 (897 high-value outliers above).
- **Correlation Matrix**:
  - `order_amount` and `discount_pct`: **-0.0951** (largest absolute correlation)
  - `order_amount` and `quantity`: **+0.0873**
  - `quantity` and `delivery_minutes`: -0.0126
  - `discount_pct` and `delivery_minutes`: -0.0098
  - `order_amount` and `delivery_minutes`: -0.0031
  - `discount_pct` and `quantity`: -0.0004

#### Business & Data Synthesis (under 200 words)
This dataset captures 11,110 completed grocery and electronics orders across six Indian metros over six months. The monetary distribution is heavily right-skewed (mean INR 269.91 vs median INR 135.00, skewness 4.61). The lower 50% comprises high-frequency grocery and dairy essentials, while the upper tail consists of 897 high-value electronics purchases exceeding the upper fence of INR 645.30. Delivery operations remain stable with standard deliveries centered around 24 minutes and late deliveries capped at 75 minutes. The strongest correlation exists between order amount and discount percentage (-0.0951), followed by quantity (+0.0873). Correlations of this magnitude are very weak and do not establish causation; higher basket totals reflect higher base unit pricing in electronics rather than heavy discounting or bulk volumes.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Table Construction
RFM metrics were calculated as of snapshot `2025-10-01 00:00:00` for all 1,175 customers with completed orders:
- **Recency**: Median = 19.0 days, Min = 0 days, Max = 180 days
- **Frequency**: Mean = 9.46 orders, Max = 33 orders
- **Monetary**: Mean = INR 2,552.06, Max = INR 14,766.55

### 2. K-Means Clustering Sweep ($K=2..8$)
RFM features were standardized using `StandardScaler`. Clustering was evaluated across $K \in [2, 8]$ with `random_state=42` and `n_init=10`:

| Clusters ($K$) | Inertia | Silhouette Score |
| :---: | :---: | :---: |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| **4** | **743.1** | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

*Why Inertia alone cannot choose $K$*: As $K$ increases, cluster centers move closer to individual data points, causing inertia (within-cluster sum of squares) to monotonically decrease toward zero where each point is its own centroid. Relying on inertia alone would favor trivial over-clustering; the elbow method and silhouette scores identify where additional clusters yield diminishing returns in compactness and separation.

### 3. Cluster Profiles, Business Personas & PCA
$K=4$ was selected based on the silhouette maximum (0.526) and business utility:

| Segment ID | Segment Name | Size | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | Recommended Business Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | Regular Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Cross-sell dairy and snacks to increase order frequency. |
| **1** | **Premium High-Value** | **138** | **24.12** | **6.77** | **7,129.70** | VIP support routing, early access to electronics deals, concierge desk. |
| **2** | Frequent Everyday | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty cashback, subscription grocery plans, churn-prevention alerts. |
| **3** | At-Risk Occasional | 182 | 109.04 | 2.20 | 685.99 | Win-back reactivations, personalized deep-discount coupons. |

- Segment 1 is designated **Premium** (`is_premium = True`) because it has the highest mean monetary value (INR 7,129.70).
- **PCA Projection**: Two components explain **63.43%** and **21.80%** of variance respectively (Total: **85.23%**).
- Exported to `outputs/customer_segments.csv`.

### 4. Cohort & Leakage-Free Feature Engineering
- **Cutoff Date**: `2025-09-01`. Cohort comprises 1,156 customers with at least one completed order prior to cutoff.
- **Target (`lapsed`)**: 1 if customer has 0 completed orders in September 2025 (2025-09-01 to 2025-09-30), else 0.
  - Non-lapsed: 713 (61.68%), Lapsed: 443 (38.32%).
- **Features Engineered (strictly prior to cutoff)**: `recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `fav_category`, `city`, `age`, `gender`, `tenure_days`.
- **Leakage Prevention Rules**: No encoder, scaler or resampler saw test rows, no data from 2025-09-01 onwards entered the features, and the segment labels from task 3 were not used as features.
- **Stratified Split (80/20, `random_state=42`)**: Train = 924 rows (570 non-lapsed, 354 lapsed), Test = 232 rows (143 non-lapsed, 89 lapsed).
- **Baseline**: Always predicting majority class (not lapsed) achieves accuracy **61.64%** and positive-class $F_1 = \mathbf{0.0000}$.

### 5. Preprocessing & Leakage Prevention
- Scalers and one-hot encoders (`OneHotEncoder(drop="first")`) were enclosed in a `ColumnTransformer` fitted exclusively on training rows.
- **Learned Frequency Mean**: Training scaler mean = **8.1147** vs Full-cohort mean = **8.0285**.
- *Why they differ*: The training mean reflects only the 924 training partition observations; fitting across the full cohort would introduce statistical data leakage from the test split.
- Verified in code: 0 missing values across all transformed feature matrices.

### 6. Model Tuning and Test Set Evaluation
Grid search cross-validation ($cv=5$, scoring=$F_1$):
- **Logistic Regression**: Best $C=100$ (CV $F_1 = 0.6506$).
- **Random Forest**: Best `max_depth=8`, `min_samples_leaf=10` (CV $F_1 = 0.6801$).

#### Untouched Test Set Evaluation (Positive Class: Lapsed)
| Model | Test Accuracy | Precision | Recall | $F_1$ Score | Confusion Matrix [TN, FP / FN, TP] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Majority Class) | 61.64% | 0.0000 | 0.0000 | 0.0000 | `[[143, 0], [89, 0]]` |
| Tuned Logistic Regression | 72.84% | 0.6444 | 0.6517 | 0.6480 | `[[111, 32], [31, 58]]` |
| **Tuned Random Forest** | **73.71%** | **0.6591** | **0.6517** | **0.6554** | `[[113, 30], [31, 58]]` |

*Model Choice Justification*: Random Forest was selected as the final production model because it achieves the highest test $F_1$ score (0.6554, comfortably exceeding the 0.55 benchmark) and superior precision (0.6591 vs 0.6444). In customer retention, misclassifying an active customer as lapsing incurs modest marketing discount costs, but failing to detect a true lapsing customer incurs the total loss of future customer lifetime value. Random Forest balances recall (65.17%) while minimizing false alarms (30 vs 32 false positives).

### 7. Class-Balance Experiment
Oversampling the minority lapsed class in the training partition (balanced to 570 lapsed and 570 non-lapsed) and refitting Logistic Regression:
- **Balanced LR Performance on Untouched Test Set**: Precision = **0.6000**, Recall = **0.8090**, $F_1 = \mathbf{0.6890}$.
- **Metric Impact**: Recall surged by **+15.73 percentage points** (0.6517 to 0.8090) while precision dropped slightly from 0.6444 to 0.6000. For Instamart's retention team, this trade-off is highly valuable when retention campaigns are inexpensive (e.g. push notifications or app banners), ensuring 81% of lapsing customers are captured.

### 8. Interpretability, Persistence & Retention List
- **Top 5 Random Forest Features**:
  1. `frequency` (0.2797): Higher historical order counts indicate entrenched ordering habits.
  2. `monetary` (0.2721): Spenders with larger cumulative basket totals exhibit distinct churn thresholds.
  3. `recency` (0.1149): Days since last purchase is the strongest immediate indicator of lapsed behavior.
  4. `avg_delivery_minutes` (0.0613): Chronic delivery friction correlates with customer disengagement.
  5. `cancellation_rate` (0.0527): High order cancellation frequency reflects service dissatisfaction.
- **Logistic Regression Key Coefficients**:
  - `recency` (+0.0391): Positive sign indicates longer inactivity directly increases lapse odds.
  - `frequency` (-1.6973): Negative sign confirms frequent ordering strongly protects against lapsing.
  - `tenure_days` (-0.0503): Older accounts demonstrate greater baseline loyalty and retention.
- **Model Persistence**: Saved to `outputs/lapse_model.joblib`. Reloading and re-predicting test data confirmed a 100% exact match.
- **Top At-Risk Premium Retention List**: Exported to `outputs/retention_list.csv` showing top 10 test-set Premium customers ranked by predicted lapse probability (1 customer actually lapsed in September).

---

## Part C — Support Ticket Triage with Guardrails

### 1. Guardrails & PII Redaction Audit
Processing all 164 tickets in `data/raw/tickets.csv`:
- **Input Guardrails**:
  - 4 empty tickets flagged and skipped without LLM calls (`skipped_empty`).
  - 4 tickets exceeded $MAX\_INPUT\_TOKENS = 400$ (each estimating to 734 tokens) and were routed to human review (`flagged_long`).
- **PII Redaction Substitutions**:
  - Emails redacted (`[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]`): **55**
  - Phone numbers redacted (`\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]`): **91**
  - Order IDs redacted (`ORD-\d{6}` $\rightarrow$ `[ORDER_ID]`): **132**
  - **Total substitutions**: **278** across 164 tickets.
  - Post-redaction assertion verified **0** residual PII patterns sent to the model.

### 2. Resilient Execution & Output Validation
- **Retry Logic**: Wrapped calls in exponential backoff ($BASE\_DELAY = 0.05\text{s} \times 2^n \times \text{uniform}(0.9, 1.15)$). In every full run of 156 tickets, exactly **19 TransientLLMErrors** occurred (every 9th call) and all 19 were successfully recovered on retry 0 with wait times between 0.045s and 0.0575s.
- **Validation Pipeline**:
  - `classified_json`: **135 tickets** returned clean JSON with valid categories.
  - `classified_keyword`: **12 tickets** returned chatty phrases (e.g. `The customer appears to have a billing-related problem.`), successfully parsed via regex word extraction (`re.findall(r"[a-z]+", text.lower())`).
  - `flagged_invalid_output`: **9 tickets** returned invalid categories (e.g. `shipping`), safely diverted to human review.

### 3. Prompt Comparison (Zero-Shot vs Few-Shot)
Both runs evaluated across the 147 classified tickets:

| Prompt Strategy | Classified Tickets | Accuracy | Total Tokens | Tokens / Ticket | Key Error Modes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot** | 147 | **85.71%** | 14,236 | 96.8 | 11 fraud tickets misclassified as billing; 10 technical tickets misclassified as billing. |
| **Few-Shot** | 147 | **100.00%** | 22,348 | 152.0 | None (0 errors across all 147 classified tickets). |

#### Zero-Shot Confusion Matrix
| True \ Predicted | billing | delivery | fraud | technical |
| :--- | :---: | :---: | :---: | :---: |
| **billing** | 38 | 0 | 0 | 0 |
| **delivery** | 0 | 36 | 0 | 0 |
| **fraud** | 11 | 0 | 23 | 0 |
| **technical** | 10 | 0 | 0 | 29 |

*Trade-off analysis*: The few-shot prompt requires 57% more tokens per ticket (152.0 vs 96.8), but eliminates all misclassifications. In production, routing an unauthorized transaction to billing causes severe security delays; paying ~55 additional tokens per ticket is well worth eliminating fraud routing failures.

### 4. Budget Demo & Segment-Aware Routing
- **Token Budget Demonstration**: Running zero-shot with $TOKEN\_BUDGET = 6,000$ stopped after consuming 6,056 tokens; **90 tickets** were safely halted with status `budget_exhausted` and saved to `outputs/triage_budget_demo.csv`.
- **Final Triage Routing**: Executed with few-shot prompt under budget 100,000 and saved to `outputs/triage_results.csv` (164 rows):
  - `security_team` (Priority HIGH): 31 tickets (all fraud tickets)
  - `premium_desk` (Priority HIGH): 23 tickets (non-fraud tickets from verified Premium customers)
  - `support_queue` (Priority NORMAL): 93 tickets (standard delivery, billing, and technical tickets)
  - `human_review` (Priority NORMAL): 17 tickets (4 empty, 4 long, 9 invalid output categories)
- **Rule-Validated vs Human Handoff Breakdown**: Exactly **147 outputs were rule-validated** (135 via valid JSON category matching and 12 via single-category regex word extraction) and automated; exactly **17 tickets went to humans** (4 empty tickets skipped by input guardrails, 4 long tickets flagged by input guardrails, and 9 invalid category responses caught by output validation).

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Document Chunking & TF-IDF Retrieval
The 5 policy documents in `data/policies/` were split into 40-word chunks with 10-word overlap (30-word step):
- `refund_policy.txt`: 4 chunks
- `delivery_policy.txt`: 3 chunks
- `escalation_policy.txt`: 3 chunks
- `payment_policy.txt`: 3 chunks
- `privacy_policy.txt`: 2 chunks
- **Total Chunks**: **15 chunks**

*TF-IDF Definition & Limitation*: Term Frequency (TF) measures how frequently a term occurs inside a single chunk, while Inverse Document Frequency (IDF) discounts ubiquitous words across all chunks to emphasize distinctive, informative keywords. A primary limitation is the lack of semantic awareness: lexical TF-IDF cannot recognize that synonyms (e.g. "reimbursement" and "refund") share identical intent if the exact token is absent.

Query benchmark: Retrieving $k=3$ for `"refund policy late delivery"` returned top 3 chunks all originating from `refund_policy.txt` (similarity scores: 0.2719, 0.2514, 0.2454).

### 2. Retrieval & Grounding Evaluation
Evaluated over all 14 questions in `data/raw/policy_eval.csv`:

| Evaluation Benchmark | Chunks (Total) | Top-3 Doc Matches | Fact in Answer Matches | Fallback Answers |
| :--- | :---: | :---: | :---: | :---: |
| **Overlap = 10 words** | 15 | **12 / 12** (100%) | **12 / 12** (100%) | **2 / 2** (100%) |
| **Overlap = 0 words** | 13 | 12 / 12 (100%) | 10 / 12 (83.3%) | 2 / 2 (100%) |

*Breakage with Overlap = 0*:
1. Question 1 (*"What delivery time makes a completed order eligible for a full refund?"*): With 0 overlap, the phrase `"delivered in more than 45 minutes"` is severed across chunk boundaries, causing the model to output only `"receives a full refund of the order amount."` without the eligibility condition.
2. Question 8 (*"How soon is money returned when a payment fails but is deducted?"*): The critical fact `"48 hours"` is split from the failure statement, returning an incomplete answer.

### 3. Agent Tool Definitions
Implemented in `src/policy_rag_agent.py` under `TOOLS_SPEC` and `TOOL_MAP`:
1. `get_order(order_id)`: Queries `data/instamart.db` for status, amount, and delivery minutes.
2. `search_policy(query)`: Executes TF-IDF retrieval against chunked policy documents.
3. `calculate_refund(order_amount, delivery_minutes)`: Returns 100% order amount if delivery minutes $> 45$, else INR 0.00.
4. `issue_refund(order_id, refund_inr)`: Records automated refunds $\le \text{INR } 1000$ to `outputs/refund_ledger.jsonl`.
5. `escalate_to_human(order_id, refund_inr)`: Records high-value escalations $> \text{INR } 1000$ to `outputs/refund_ledger.jsonl`.

### 4. ReAct Refund Agent Scenarios
All seven scenarios in `data/raw/agent_scenarios.csv` executed through the ReAct loop and appended to `outputs/agent_runs.jsonl`:

| Scenario ID | Outcome Status | Refund Amount (INR) | Model Calls | Tokens Used | Match Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | `REFUND` | INR 128.25 | 5 | 897 | Matched (Expected: REFUND, INR 128.25) |
| **S2** | `REFUND` | INR 135.00 | 5 | 897 | Matched (Expected: REFUND, INR 135.00) |
| **S3** | `ESCALATE` | INR 1999.00 | 5 | 884 | Matched (Expected: ESCALATE, INR 1999.00) |
| **S4** | `DENIED` | INR 0.00 | 4 | 605 | Matched (Expected: DENIED, on-time delivery) |
| **S5** | `DENIED` | INR 0.00 | 3 | 337 | Matched (Expected: DENIED, cancelled order) |
| **S6** | `NOT_FOUND` | INR 0.00 | 2 | 72 | Matched (Expected: NOT_FOUND, invalid ID) |
| **S7** | `FALLBACK_MAX_ITERATIONS` | INR 0.00 | 6 | 3,000 | Matched (Expected: FALLBACK, loop query) |

*Notes on S7*: Under default `token_limit=6000`, S7 successfully completed all 6 model iterations without tripping the token budget, consuming 3,000 tokens (within the reference range of 2,700 to 5,500 tokens).

### 5. Stop-Safely Demonstrations
1. **Token Limit Guardrail Demo** (`token_limit = 200` on S1):
   - Status: `FALLBACK_TOKEN_LIMIT`
   - Final Answer: `Token limit exceeded (354 > 200). Stopped immediately without tool execution. Handing off to human lead.`
2. **Timeout Guardrail Demo** (`time_limit_seconds = 0` on S1):
   - Status: `FALLBACK_TIMEOUT`
   - Final Answer: `Execution timed out after 0.00s across 0 steps. Handing off case to human lead.`

### 6. Crash-and-Resume Demonstration
- Executed S1 with a simulated crash right after tool call #2 (`search_policy`).
- Checkpoint `state/s1_crash_resume_demo.json` was saved.
- `resume_agent("s1_crash_resume_demo")` reloaded state and continued without repeating tools.
- **Tools executed in order**:
  1. `get_order` (pre-crash)
  2. `search_policy` (pre-crash)
  3. `calculate_refund` (post-resume)
  4. `issue_refund` (post-resume)
- **Total Tool Executions**: Exactly 4, none repeated. Final Status: `REFUND: INR 128.25 issued.`

### 7. ReAct Versus Plan-and-Execute (244 words)
In autonomous agent workflows, architectural choice dictates cost, latency, and resilience. Our measured ReAct implementation averaged 4.29 model calls per scenario (ranging from 2 calls for NOT_FOUND up to 6 for S7), consuming between 72 and 3,000 tokens per request. A plan-and-execute agent, by contrast, queries the model once up front to generate an entire sequence of steps (e.g. `[get_order, search_policy, calculate_refund, issue_refund]`), executing them sequentially and replanning only upon exceptions.

While plan-and-execute reduces LLM round-trips for standard happy-path refund requests, it performs poorly under unexpected observations. When an order is cancelled (S5) or non-existent (S6), a pre-generated plan becomes obsolete after the first step, requiring a full replanning cycle that negates initial token savings. More critically, an agent generating speculative multi-step plans risks executing side-effecting financial actions—such as queuing an unverified refund transaction—before policy rules are validated.

For Instamart’s customer refund operations, the ReAct pattern is unequivocally superior. Each decision is strictly conditioned on the immediate observation of the preceding tool call. When `get_order` returns `cancelled` or `order not found`, the agent halts immediately without issuing redundant policy searches or misallocating company capital. Coupled with runtime execution guardrails (token ceilings, monotonic timeouts, iteration caps) and persistent JSON checkpointing, ReAct ensures deterministic compliance and fault-tolerant recoverability in mission-critical financial workflows.

---

## Limitations and Next Steps

1. **Post-Hoc Whole-Window Segment Labeling in Retention List**:
   The `is_premium` flag assigned in `outputs/retention_list.csv` originates from the whole-window RFM segmentation (Task B3, covering April to September 2025). Because the September ordering window defines the lapse target, the retention list is illustrative of high-value at-risk accounts and must not be treated as a leakage-free predictive evaluation. In production, segmentation should be recomputed using rolling 90-day snapshot windows prior to the cutoff.
2. **Lexical TF-IDF Limitations in Policy RAG**:
   While TF-IDF achieves 100% retrieval accuracy on exact-keyword queries, it struggles with paraphrased queries, typos, and semantic abstractions. Upgrading to dense bi-encoder embeddings (e.g. `all-MiniLM-L6-v2`) with hybrid BM25 re-ranking will improve domain generalization across messy user vernacular.
3. **Synthetic Data Distributions vs Metropolitan Scale**:
   The synthetic dataset models 1,200 customers and ~13,000 orders. Real-world Instamart logistics handle millions of deliveries with hyper-local micro-fulfillment dynamics (dark stores, rain surges, traffic delays, multi-item basket splittings). Expanding the feature space to include live dark-store inventory levels, rider transit delays, and payment gateway health will enhance predictive accuracy.
4. **Human-in-the-Loop Orchestration**:
   The current triage and refund system routes edge cases to `human_review` and `escalate_to_human`. Building a bidirectional review dashboard where support leads can approve escalated refunds with single-click ledger reconciliation will close the operational feedback loop.
