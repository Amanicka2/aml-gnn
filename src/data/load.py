"""Load the IBM AML CSV into DuckDB and assign a chronological 60/20/20 split.

Usage:
    python -m src.data.load --csv data/HI-Small_Trans.csv --db data/aml.duckdb
"""
import argparse
import time
from pathlib import Path

import duckdb

# The raw CSV has two columns both called "Account" (sender and receiver).
# We give every column an explicit clean name so nothing gets silently renamed.
COLUMNS = {
    "ts_raw": "VARCHAR",
    "from_bank": "INTEGER",
    "from_account": "VARCHAR",
    "to_bank": "INTEGER",
    "to_account": "VARCHAR",
    "amount_received": "DOUBLE",
    "receiving_currency": "VARCHAR",
    "amount_paid": "DOUBLE",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "INTEGER",
}


def load(csv_path: str, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    cols_sql = "{" + ", ".join(f"'{k}': '{v}'" for k, v in COLUMNS.items()) + "}"

    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE transactions AS
        WITH raw AS (
            SELECT * FROM read_csv('{csv_path}', header = true, columns = {cols_sql})
        ),
        clean AS (
            SELECT
                strptime(ts_raw, '%Y/%m/%d %H:%M') AS ts,
                from_bank, from_account, to_bank, to_account,
                -- A node is a (bank, account) pair, so the same account string
                -- at two different banks never gets merged by accident.
                CAST(from_bank AS VARCHAR) || '_' || from_account AS src_id,
                CAST(to_bank AS VARCHAR)   || '_' || to_account   AS dst_id,
                amount_received, receiving_currency,
                amount_paid, payment_currency,
                payment_format, is_laundering
            FROM raw
        ),
        ordered AS (
            SELECT
                -- Sort by time, then tie break deterministically so the split is reproducible.
                row_number() OVER (ORDER BY ts, src_id, dst_id, amount_paid) - 1 AS tx_id,
                count(*) OVER () AS n,
                *
            FROM clean
        )
        SELECT
            * EXCLUDE (n),
            CASE
                WHEN tx_id < 0.6 * n THEN 'train'
                WHEN tx_id < 0.8 * n THEN 'val'
                ELSE 'test'
            END AS split
        FROM ordered
        ORDER BY tx_id
    """)
    n = con.execute("SELECT count(*) FROM transactions").fetchone()[0]
    print(f"Loaded {n:,} transactions into {db_path} in {time.time() - t0:.1f}s")
    print(con.execute("""
        SELECT split, count(*) AS n_tx, min(ts) AS first_ts, max(ts) AS last_ts,
               sum(is_laundering) AS n_laundering
        FROM transactions GROUP BY split ORDER BY min(tx_id)
    """).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/HI-Small_Trans.csv")
    p.add_argument("--db", default="data/aml.duckdb")
    a = p.parse_args()
    load(a.csv, a.db)
