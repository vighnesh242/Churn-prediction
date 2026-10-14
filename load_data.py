import pandas as pd
import sqlite3

df = pd.read_csv("data/churn_prediction.csv", na_values=["NaT"])
conn = sqlite3.connect("churn.db")
df.to_sql("customers", conn, if_exists="replace", index=False)
print(f"Loaded {len(df)} rows")
conn.close()