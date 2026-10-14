Customer Churn Prediction for Retail Banking
=============================================

**Tools:** SQL (SQLite) · Python (pandas, scikit-learn, XGBoost, SHAP) · Tableau Public

I built this project to answer a question every retail bank cares about: which customers are about to leave, and which of them are worth fighting for? I engineered RFM (Recency, Frequency, Monetary) features in SQL, trained an XGBoost model in Python to score churn risk for 28,382 customers, and built a Tableau dashboard that turns those scores into a ranked list of customers for the retention team to call first.


Results at a Glance
-------------------

| Metric (test set, 20% holdout) | Logistic Regression | XGBoost |
|---|---|---|
| Decision threshold | 0.5 | 0.4 |
| Accuracy | 69.5% | 78.6% |
| Precision (churn) | 34.1% | 45.4% |
| Recall (churn) | 69.1% | 75.2% |
| ROC-AUC | 0.754 | 0.839 |

XGBoost beat the baseline on every metric, raising AUC from 0.754 to 0.839 and catching 75% of churners in the holdout set.

Business results:

* On the holdout set, the **High risk tier (score above 0.6) flagged about 20% of customers and caught 62% of all churners.** 58% of flagged customers actually churned vs an 18.5% baseline, so targeting this group is about 3.1x more effective than picking customers at random.
* Across the full customer base, 5,600 customers fall in the High risk tier, holding about **20.7 million in combined balances**, which is the money the bank stands to lose.
* Customers whose balance dropped the most churned at **42%**, compared to about 11% for customers whose balance held steady or grew.


The Business Problem
--------------------

Losing a banking customer is expensive. The bank loses the deposits it lends out, the fees, and the chance to sell that customer a loan or card later. Winning a new customer usually costs several times more than keeping an existing one.

Retention teams can't call everyone, though. They need a short, ranked list: who is likely to leave, and how much money leaves with them. That is what this project produces.


Domain Knowledge
----------------

A few banking and analytics concepts that shaped the decisions in this project.

**What "churn" means here.** In this dataset, churn is not an account closure. A customer is flagged as churned when their average balance falls below the minimum balance in the next quarter. In practice, the customer has moved their money elsewhere even if the account is technically still open. This is sometimes called "silent churn," and it is harder to spot than a closure because nothing formally happens.

**RFM analysis.** RFM is a classic customer segmentation method from retail and direct marketing.

* **Recency:** how recently the customer did something. Recent activity usually means engagement.
* **Frequency:** how often they do things.
* **Monetary:** how much money is involved.

Each customer gets a score from 1 to 5 on each dimension, and the combination places them in a segment like "Champions" or "At Risk."

**Why accuracy alone is misleading.** Only 18.5% of customers churned. A model that predicts "nobody churns" would be 81.5% accurate and completely useless. That is why I report recall (what share of real churners the model catches), precision (how many flagged customers actually churn), and ROC-AUC (how well the model ranks risky customers above safe ones).

**Recall vs precision trade-off.** Missing a churner means losing a customer. A false alarm means one unnecessary retention call. For most banks, missing a churner costs more, so I leaned toward higher recall when picking the decision threshold.

**Lift.** Lift compares the model's hit rate to random selection. If 58% of flagged customers churn and the base rate is 18.5%, the model is about 3.1x better than calling customers at random. This is the number a business stakeholder actually cares about.


Dataset
-------

**Source:** [Bank Customer Churn Data on Kaggle](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data)

* 28,382 customers, 21 columns
* Transactions dated December 31, 2018 to December 31, 2019
* Churn rate: 18.5%

Main column groups:

| Group | Columns |
|---|---|
| Demographics | age, gender, dependents, occupation, city |
| Relationship | vintage (days since account opened), customer_nw_category, branch_code |
| Balances | current_balance, previous_month_end_balance, average_monthly_balance_prevQ, average_monthly_balance_prevQ2, current_month_balance, previous_month_balance |
| Activity | current_month_credit, previous_month_credit, current_month_debit, previous_month_debit, last_transaction |
| Target | churn (1 = churned, 0 = stayed) |


Tech Stack
----------

| Tool | What I used it for |
|---|---|
| SQLite | Storing raw data, data quality checks, RFM feature engineering, pulling the retention target list |
| Python (pandas, NumPy) | Loading data, EDA, preparing model inputs |
| scikit-learn | Train/test split, logistic regression baseline, evaluation metrics |
| XGBoost | Main churn model |
| SHAP | Explaining which features drive churn predictions |
| Matplotlib, Seaborn | EDA charts |
| Tableau Public | Interactive retention dashboard |
| Cursor, Git, GitHub | Development and version control |


Project Structure
-----------------

```
churn-prediction/
  data/
    raw/                    original Kaggle CSV (not committed)
    processed/              scored_customers.csv, drivers.csv, retention_targets.csv
  sql/
    01_checks.sql           data quality checks
    02_features.sql         RFM and behavioral feature engineering
    03_segments.sql         RFM segment labels
  notebooks/
    01_eda.ipynb            exploratory analysis
    02_model.ipynb          modeling, evaluation, SHAP, scoring
  outputs/                  saved charts
  load_data.py              loads the CSV into SQLite
  build_features.py         runs the SQL feature pipeline
  README.md
```


How to Run It
-------------

1. Clone the repo and create a virtual environment:

```bash
git clone [add repo link]
cd churn-prediction
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

2. Download the dataset from Kaggle and place it at `data/raw/churn_prediction.csv`.

3. Load the data and build features:

```bash
python load_data.py
python build_features.py
```

4. Open `notebooks/01_eda.ipynb` and `notebooks/02_model.ipynb` and run all cells in order.


Step by Step: What I Did and Why
--------------------------------

**Step 1: Load the data into SQLite**

`load_data.py` reads the CSV with pandas and writes it to a `customers` table in `churn.db`. I chose SQLite because it needs no server setup and still supports window functions like `NTILE`, which the RFM scoring depends on.

**Step 2: Data quality checks**

Before building anything, I checked row counts, the churn rate, the date range, and nulls. Gender, dependents, occupation, city, and last_transaction all had missing values. I kept a `has_last_txn` flag instead of dropping those rows, because a missing transaction can itself be a signal.

**Step 3: RFM feature engineering in SQL**

`02_features.sql` builds a `features` table:

* **recency_days:** days between the last transaction and a reference date (one day after the latest transaction in the data). Customers with no transaction get 999 so they fall into the lowest recency group.
* **freq_score (0 to 4):** the dataset has no raw transaction counts, so I built a proxy. It counts how many of the four monthly credit and debit amounts show meaningful activity.
* **monetary_flow:** total credits plus debits across the last two months.
* **balance_change:** current balance minus last month's ending balance.
* **debit_trend:** this month's debits divided by last month's.
* **r_score and m_score:** 1 to 5 scores using `NTILE(5)`.

`03_segments.sql` then labels each customer as Champions, Active Low Value, At Risk High Value, Dormant, or Middle based on their R and M scores.

**Step 4: Exploratory analysis**

I compared churn across segments, occupations, net worth categories, and balance trends. The clearest pattern was balance movement: customers with the biggest balance drops churned at nearly four times the rate of everyone else.

**Step 5: Modeling**

* Dropped ID columns, dates, and two high-cardinality codes (city with 1,604 values and branch_code with 3,185) that would mostly add noise.
* Filled missing gender and occupation with "Unknown" and one-hot encoded them.
* Split 80/20 with stratification so both sets kept the same churn ratio.
* Trained a logistic regression baseline, then XGBoost with `scale_pos_weight` to give the minority churn class more weight.
* Tested decision thresholds from 0.3 to 0.6 and picked 0.4. It caught 75% of churners while keeping false alarms well below the 0.3 cutoff, which caught 82% of churners but cut precision to 38%.

**Step 6: Explaining the model**

I used SHAP to measure how much each feature pushes predictions up or down. This turns the model from a black box into something a retention manager can act on.

**Step 7: Scoring and risk tiers**

I scored every customer and grouped them by churn probability: Low (below 0.3), Medium (0.3 to 0.6), and High (above 0.6). The 0.4 threshold answers "will this customer churn?" for model evaluation, while the tiers answer "who do we call first?" The High tier is deliberately stricter so the retention team starts with the most certain cases. The scores were written back to SQLite, and a SQL query pulls the High risk customers sorted by balance into `retention_targets.csv`.

**Step 8: Tableau dashboard**

The dashboard includes KPI cards (customers, churn rate, high risk count, balance at risk), customer counts by risk tier, an R score vs M score churn heatmap, the top 10 churn drivers, and a filterable table of high risk customers sorted by balance.


Data Issues I Found and Fixed
-----------------------------

Two bugs showed up after my first full run. Both were silent, meaning nothing errored out, and the outputs just looked wrong.

**1. Missing dates stored as the text "NaT."** 3,223 rows had the string "NaT" instead of a real null in last_transaction. In SQL, the text "NaT" sorts after every date, so `MAX(last_transaction)` returned "NaT," the reference date became null, and every customer got a recency of 999. The giveaway was that r_score came out identical to m_score for every row, and "Champions" had higher churn than "Dormant." I fixed it with `na_values=["NaT"]` when reading the CSV.

**2. Placeholder values instead of zeros.** Every credit and debit column had a minimum of 0.01, so checking for activity with `> 0` marked every customer as fully active and freq_score was 4 for everyone. About half the customers had values at or below 1, which in practice means no real activity. Changing the threshold to `> 1` gave a proper spread across all five frequency levels.

The lesson: always check the distribution of every engineered feature. A feature with one unique value or two features that are perfectly correlated is a bug until proven otherwise.


Key Takeaways
-------------

**Top churn drivers (SHAP)**

| Rank | Feature | What it means |
|---|---|---|
| 1 | Current balance | Current balance is the strongest single signal |
| 2 | Avg monthly balance, previous quarter | Where the balance used to sit matters as much as where it is now |
| 3 | Balance change | Money leaving the account is an early warning |
| 4 | Avg monthly balance, two quarters ago | Longer-term balance trend |
| 5 | Monetary flow | Total money moving through the account |

**Churn by RFM segment**

| Segment | Customers | Churn rate |
|---|---|---|
| At Risk High Value | 1,317 | 48.7% |
| Champions | 7,359 | 25.4% |
| Middle | 9,859 | 20.3% |
| Dormant | 8,451 | 7.7% |
| Active Low Value | 1,396 | 7.4% |

**What surprised me**

* **Inactive customers churn less, not more.** Dormant customers churned at 7.7%, and customers with no recorded transaction at about 9%. Because churn here means a balance falling below the minimum, customers with little money moving have less to lose. Churn in this dataset is about money leaving, not customers going quiet.
* **Champions churn more than average (25%).** High value, active customers have the most money that can move, and competitors target them too. Being "good" customers doesn't make them safe.
* **Demographics barely matter.** Churn rates by net worth category ranged only from 17.9% to 19.2%, and gender and dependents ranked near the bottom of the drivers. Behavior predicts churn far better than who the customer is.


Recommendations for the Bank
----------------------------

1. **Start with At Risk High Value customers.** They churn at nearly half and still hold meaningful balances. A relationship manager call or a rate offer here has the highest expected return.
2. **Set up a balance drop alert.** A large drop in balance from one month to the next is the clearest early signal. Flagging it weekly would give the retention team time to act before the quarter ends.
3. **Don't take Champions for granted.** A loyalty program, priority service, or better rates on deposits could protect the bank's most valuable relationships.
4. **Focus on self-employed customers.** They make up 68% of the High risk tier and churn at the highest rate of any large occupation group (19.8%). Business banking products may be a better fit for them.


Limitations
-----------

* **Frequency is a proxy.** The dataset has monthly totals, not individual transactions, so freq_score counts active months rather than actual transaction counts.
* **One snapshot in time.** The data covers a single year, so I couldn't check whether the model holds up across seasons or economic changes.
* **Currency is not stated** in the dataset, so balance figures are shown without units.
* **No cost data.** A real deployment would weigh the cost of a retention offer against the value of each customer to choose the threshold.


What I Would Do Next
--------------------

* Tune XGBoost hyperparameters with cross-validation.
* Pick the threshold using expected profit instead of a metric balance.
* Rebuild the pipeline in Snowflake or PostgreSQL with scheduled scoring.
* Add model monitoring to catch drift as customer behavior changes.
* Test retention offers on a holdout group to measure real impact.


What I Learned
--------------

* Doing feature engineering in SQL keeps the logic transparent and reusable, and it mirrors how analytics teams work with warehouse data.
* Accuracy is the wrong headline metric for imbalanced problems. Recall, AUC, and lift tell the real story.
* Silent data issues are more dangerous than errors. Checking feature distributions caught two bugs that would have made the RFM analysis meaningless.
* A model is only useful if someone can act on it. The ranked target list and the SHAP drivers matter more to a business user than the AUC score.
