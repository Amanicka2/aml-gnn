"""Build leak-free transaction features for the tabular baseline.

Every feature for a transaction at time T only uses transactions at or before T.
That is exactly what a live scoring service would know, so there is no peeking
into the future.

Usage:
    python -m src.features.build_features --db data/aml.duckdb
"""
import argparse
import time

import duckdb

# Rolling windows (in hours) for the per-account graph features.
WINDOWS = {"24h": 24, "7d": 24 * 7}


def build(db_path: str, memory_limit: str = "2GB") -> None:
    con = duckdb.connect(db_path)
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET temp_directory = 'data/duckdb_tmp'")
    con.execute("SET preserve_insertion_order = false")
    t0 = time.time()

    # 1) Currency-aware amount. Amounts come in different currencies (Bitcoin is
    #    tiny, Yen is huge), so we compare each amount with the typical amount in
    #    its own currency. The "typical" value is the median log amount on the
    #    TRAIN split only, so val/test never influence it.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cur_stats AS
        SELECT payment_currency AS currency,
               median(ln(1 + amount_paid)) AS med_log_amt
        FROM transactions WHERE split = 'train'
        GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE tx AS
        SELECT t.tx_id, t.ts, t.src_id, t.dst_id, t.split, t.is_laundering,
               ln(1 + t.amount_paid) - coalesce(c.med_log_amt, 0) AS amt_rel
        FROM transactions t LEFT JOIN cur_stats c ON t.payment_currency = c.currency
    """)
    print(f"[{time.time() - t0:5.1f}s] relative amounts done")

    # 2) Turn every transaction into two "account events": the sender sees an
    #    outgoing payment, the receiver sees an incoming one. Then a window over
    #    each account's own events gives its recent behaviour.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT tx_id, ts, src_id AS account, dst_id AS counterparty, 'out' AS dir, amt_rel FROM tx
        UNION ALL
        SELECT tx_id, ts, dst_id, src_id, 'in', amt_rel FROM tx
    """)

    win_cols = []
    win_defs = []
    for name, hours in WINDOWS.items():
        w = f"w_{name}"
        win_defs.append(
            f"{w} AS (PARTITION BY account ORDER BY ts "
            f"RANGE BETWEEN INTERVAL {hours} HOURS PRECEDING AND CURRENT ROW)"
        )
        win_cols += [
            f"count(*) FILTER (WHERE dir = 'out') OVER {w} AS out_cnt_{name}",
            f"count(*) FILTER (WHERE dir = 'in')  OVER {w} AS in_cnt_{name}",
            f"count(DISTINCT counterparty) FILTER (WHERE dir = 'out') OVER {w} AS out_uniq_{name}",
            f"count(DISTINCT counterparty) FILTER (WHERE dir = 'in')  OVER {w} AS in_uniq_{name}",
            f"avg(amt_rel) FILTER (WHERE dir = 'out') OVER {w} AS out_amt_mean_{name}",
            f"avg(amt_rel) FILTER (WHERE dir = 'in')  OVER {w} AS in_amt_mean_{name}",
        ]
    win_defs.append("w_all AS (PARTITION BY account ORDER BY ts "
                    "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)")
    win_cols += [
        "count(*) FILTER (WHERE dir = 'out') OVER w_all AS out_cnt_all",
        "count(*) FILTER (WHERE dir = 'in')  OVER w_all AS in_cnt_all",
        "date_diff('minute', min(ts) OVER w_all, ts) AS acct_age_min",
    ]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE acct_feats AS
        SELECT tx_id, dir, {', '.join(win_cols)}
        FROM events
        WINDOW {', '.join(win_defs)}
    """)
    print(f"[{time.time() - t0:5.1f}s] rolling account features done")

    # 3) Join the sender's view (dir = 'out') and the receiver's view (dir = 'in')
    #    back onto each transaction, next to the plain transaction columns.
    feat_names = [c.split(" AS ")[-1] for c in win_cols]
    src_cols = ", ".join(f"s.{c} AS src_{c}" for c in feat_names)
    dst_cols = ", ".join(f"d.{c} AS dst_{c}" for c in feat_names)
    con.execute(f"""
        CREATE OR REPLACE TABLE features AS
        SELECT
            t.tx_id, t.split, t.is_laundering, t.ts,
            ln(1 + t.amount_paid)     AS log_amt_paid,
            x.amt_rel,
            ln(1 + t.amount_received) - ln(1 + t.amount_paid) AS log_recv_minus_paid,
            t.payment_format, t.payment_currency, t.receiving_currency,
            CAST(t.from_bank <> t.to_bank AS INTEGER)                    AS cross_bank,
            CAST(t.payment_currency <> t.receiving_currency AS INTEGER) AS cross_currency,
            CAST(t.src_id = t.dst_id AS INTEGER)                         AS self_transfer,
            hour(t.ts) AS hour, dayofweek(t.ts) AS dow,
            {src_cols},
            {dst_cols}
        FROM transactions t
        JOIN tx x USING (tx_id)
        JOIN acct_feats s ON s.tx_id = t.tx_id AND s.dir = 'out'
        JOIN acct_feats d ON d.tx_id = t.tx_id AND d.dir = 'in'
        ORDER BY t.tx_id
    """)
    n, ncol = con.execute(
        "SELECT count(*), (SELECT count(*) FROM information_schema.columns WHERE table_name='features') FROM features"
    ).fetchone()
    print(f"[{time.time() - t0:5.1f}s] features table: {n:,} rows x {ncol} columns")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/aml.duckdb")
    p.add_argument("--memory-limit", default="2GB")
    a = p.parse_args()
    build(a.db, a.memory_limit)
