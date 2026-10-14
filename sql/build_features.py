import sqlite3
import pandas as pd

conn = sqlite3.connect("churn.db")

for f in ["sql/02_features.sql", "sql/03_segments.sql"]:
    conn.executescript(open(f).read())
    print(f"Ran {f}")

conn.commit()

conn.close()