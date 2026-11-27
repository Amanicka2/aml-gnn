"""Turn the transactions table into graph arrays for the GNN.

Nodes  = accounts (bank + account id).
Edges  = transactions, directed sender -> receiver, each with a feature vector.

Edge features are only the RAW transaction fields (amount, format, currency,
flags, hour). The engineered rolling graph features from the XGBoost baseline
are deliberately left out: the whole point of the GNN is to learn the graph
patterns itself, so the comparison with XGBoost is fair.

Usage:
    python -m src.gnn.prepare_graph --db data/aml.duckdb --out data/graph
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def prepare(db_path: str, out_dir: str) -> None:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)
    con.execute("SET memory_limit = '1GB'")

    # Category lists come from TRAIN only (anything unseen later is just all zeros).
    formats = [r[0] for r in con.execute(
        "SELECT DISTINCT payment_format FROM transactions WHERE split='train' ORDER BY 1").fetchall()]
    currencies = [r[0] for r in con.execute(
        "SELECT DISTINCT payment_currency FROM transactions WHERE split='train' ORDER BY 1").fetchall()]

    # Standardize the continuous columns with TRAIN mean / std.
    cont = ["amt_rel", "log_amt_paid", "log_recv_minus_paid"]
    stats = con.execute("SELECT " + ", ".join(
        f"avg({c}), stddev_pop({c})" for c in cont) + " FROM features WHERE split='train'").fetchall()[0]
    mean_std = {c: (stats[2 * i], stats[2 * i + 1] or 1.0) for i, c in enumerate(cont)}

    def q(s):
        return "'" + s.replace("'", "''") + "'"

    feat_sql, feat_names = [], []
    for c in cont:
        m, s = mean_std[c]
        feat_sql.append(f"CAST((f.{c} - {m}) / {s} AS FLOAT)")
        feat_names.append(c)
    for v in formats:
        feat_sql.append(f"CAST(t.payment_format = {q(v)} AS FLOAT)")
        feat_names.append(f"fmt={v}")
    for v in currencies:
        feat_sql.append(f"CAST(t.payment_currency = {q(v)} AS FLOAT)")
        feat_names.append(f"cur={v}")
    for c in ["cross_bank", "cross_currency", "self_transfer"]:
        feat_sql.append(f"CAST(f.{c} AS FLOAT)")
        feat_names.append(c)
    feat_sql += ["CAST(sin(2 * pi() * f.hour / 24) AS FLOAT)", "CAST(cos(2 * pi() * f.hour / 24) AS FLOAT)"]
    feat_names += ["hour_sin", "hour_cos"]

    con.execute("""
        CREATE TEMP TABLE node_ids AS
        SELECT id, CAST(row_number() OVER (ORDER BY id) - 1 AS INTEGER) AS nid
        FROM (SELECT src_id AS id FROM transactions UNION SELECT dst_id FROM transactions)
    """)
    num_nodes = con.execute("SELECT count(*) FROM node_ids").fetchall()[0][0]
    n = con.execute("SELECT count(*) FROM transactions").fetchall()[0][0]

    src = np.empty(n, np.int32)
    dst = np.empty(n, np.int32)
    split = np.empty(n, np.int8)
    y = np.empty(n, np.int8)
    ts_min = np.empty(n, np.int64)
    X = np.empty((n, len(feat_names)), np.float16)

    cols = ", ".join(f"{e} AS f{i}" for i, e in enumerate(feat_sql))
    reader = con.sql(f"""
        SELECT s.nid AS src, d.nid AS dst,
               CASE t.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END AS split,
               t.is_laundering AS y, epoch(t.ts) // 60 AS ts_min, {cols}
        FROM transactions t
        JOIN features f USING (tx_id)
        JOIN node_ids s ON s.id = t.src_id
        JOIN node_ids d ON d.id = t.dst_id
        ORDER BY t.tx_id
    """).fetch_arrow_reader(250_000)
    i = 0
    for b in reader:
        k = b.num_rows
        src[i:i + k] = b.column("src").to_numpy()
        dst[i:i + k] = b.column("dst").to_numpy()
        split[i:i + k] = b.column("split").to_numpy()
        y[i:i + k] = b.column("y").to_numpy()
        ts_min[i:i + k] = b.column("ts_min").to_numpy()
        for j in range(len(feat_names)):
            X[i:i + k, j] = b.column(f"f{j}").to_numpy(zero_copy_only=False)
        i += k
    con.close()

    for name, arr in dict(src=src, dst=dst, split=split, y=y, ts_min=ts_min, X=X).items():
        np.save(out / f"{name}.npy", arr)
    meta = {"num_nodes": int(num_nodes), "num_edges": int(n), "feature_names": feat_names,
            "split_code": SPLIT_CODE}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.time() - t0:5.1f}s] graph: {num_nodes:,} nodes, {n:,} edges, "
          f"{len(feat_names)} edge features -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/aml.duckdb")
    p.add_argument("--out", default="data/graph")
    a = p.parse_args()
    prepare(a.db, a.out)
