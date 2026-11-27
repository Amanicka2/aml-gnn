"""SQL exploratory analysis on the DuckDB table.

Usage:
    python -m src.eda --db data/aml.duckdb
"""
import argparse

import duckdb
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

QUERIES = {
    "1. Overall size and laundering rate": """
        SELECT
            count(*) AS n_tx,
            (SELECT count(*) FROM (SELECT src_id AS id FROM transactions
                                   UNION SELECT dst_id FROM transactions)) AS n_accounts,
            sum(is_laundering) AS n_laundering,
            round(100.0 * avg(is_laundering), 4) AS laundering_pct,
            round(1.0 / avg(is_laundering)) AS one_in_every,
            min(ts) AS first_ts, max(ts) AS last_ts
        FROM transactions
    """,
    "2. Laundering rate by split (is the rate stable over time?)": """
        SELECT split, count(*) AS n_tx, sum(is_laundering) AS n_laundering,
               round(100.0 * avg(is_laundering), 4) AS laundering_pct
        FROM transactions GROUP BY split ORDER BY min(tx_id)
    """,
    "3. Amounts and laundering rate by payment format": """
        SELECT payment_format,
               count(*) AS n_tx,
               sum(is_laundering) AS n_laundering,
               round(100.0 * avg(is_laundering), 4) AS laundering_pct,
               round(median(amount_paid), 2) AS median_paid,
               round(median(amount_paid) FILTER (WHERE is_laundering = 1), 2) AS median_paid_laundering
        FROM transactions GROUP BY payment_format ORDER BY n_laundering DESC
    """,
    "4. Top 10 sending banks by laundering count": """
        SELECT from_bank, count(*) AS n_tx, sum(is_laundering) AS n_laundering,
               round(100.0 * avg(is_laundering), 4) AS laundering_pct
        FROM transactions GROUP BY from_bank
        ORDER BY n_laundering DESC LIMIT 10
    """,
    "5. Cross bank and cross currency transfers": """
        SELECT
            from_bank <> to_bank AS cross_bank,
            payment_currency <> receiving_currency AS cross_currency,
            count(*) AS n_tx, sum(is_laundering) AS n_laundering,
            round(100.0 * avg(is_laundering), 4) AS laundering_pct
        FROM transactions GROUP BY ALL ORDER BY ALL
    """,
    "6. Self transfers (same account sends to itself)": """
        SELECT src_id = dst_id AS self_transfer, count(*) AS n_tx,
               sum(is_laundering) AS n_laundering,
               round(100.0 * avg(is_laundering), 4) AS laundering_pct
        FROM transactions GROUP BY ALL ORDER BY ALL
    """,
    "7. Daily volume and laundering (watch for odd tail days)": """
        SELECT CAST(ts AS DATE) AS day, count(*) AS n_tx, sum(is_laundering) AS n_laundering,
               round(100.0 * avg(is_laundering), 4) AS laundering_pct
        FROM transactions GROUP BY ALL ORDER BY day
    """,
    "8. How many accounts touch at least one laundering tx": """
        WITH acc AS (
            SELECT src_id AS id, is_laundering FROM transactions
            UNION ALL SELECT dst_id, is_laundering FROM transactions
        )
        SELECT count(DISTINCT id) AS n_accounts,
               count(DISTINCT id) FILTER (WHERE is_laundering = 1) AS n_accounts_in_laundering
        FROM acc
    """,
}


def main(db_path: str) -> None:
    con = duckdb.connect(db_path, read_only=True)
    for title, sql in QUERIES.items():
        print(f"\n=== {title} ===")
        print(con.execute(sql).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/aml.duckdb")
    main(p.parse_args().db)
