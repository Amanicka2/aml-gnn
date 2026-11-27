"""Baseline 1: XGBoost on tabular + engineered graph features.

Usage:
    python -m src.models.train_xgb --db data/aml.duckdb
"""
import argparse
import gc
import json
import time
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.metrics import best_f1_threshold, scores_at

CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency"]
DROP = ["tx_id", "split", "is_laundering", "ts"]
TAIL_START = pd.Timestamp("2022-09-11")  # sparse end-of-simulation days found in EDA

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "nthread": 4,
    "seed": 42,
}


def feature_columns(con):
    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'features' ORDER BY ordinal_position").fetchall()]
    return [c for c in cols if c not in DROP]


def category_maps(con):
    """Category -> integer code, learned from TRAIN only. Unseen categories become NULL (missing)."""
    return {c: [r[0] for r in con.execute(
                f"SELECT DISTINCT {c} FROM features WHERE split = 'train' ORDER BY 1").fetchall()]
            for c in CATEGORICAL}


def load_split(con, split, cols, cat_maps):
    """Load one split straight into a compact float32 matrix (no pandas copy), to fit in little RAM."""
    exprs = []
    for c in cols:
        if c in CATEGORICAL:
            cats = ", ".join("'" + v.replace("'", "''") + "'" for v in cat_maps[c])
            exprs.append(f"CAST(nullif(list_position([{cats}], {c}), 0) - 1 AS FLOAT) AS {c}")
        else:
            exprs.append(f"CAST({c} AS FLOAT) AS {c}")
    where = f"WHERE split = '{split}'"
    n = con.execute(f"SELECT count(*) FROM features {where}").fetchone()[0]
    X = np.empty((n, len(cols)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    ts = np.empty(n, dtype="datetime64[us]")
    reader = con.sql(
        f"SELECT {', '.join(exprs)}, is_laundering, ts FROM features {where} ORDER BY tx_id"
    ).fetch_arrow_reader(200_000)
    i = 0
    for batch in reader:  # fill the preallocated matrix 200k rows at a time
        k = batch.num_rows
        for j, c in enumerate(cols):
            X[i:i + k, j] = batch.column(c).to_numpy(zero_copy_only=False)
        y[i:i + k] = batch.column("is_laundering").to_numpy(zero_copy_only=False)
        ts[i:i + k] = batch.column("ts").to_numpy(zero_copy_only=False)
        i += k
    return X, y, ts


def main(db_path, max_rounds, scale_pos_weight):
    con = duckdb.connect(db_path, read_only=True)
    con.execute("SET memory_limit = '1GB'")
    t0 = time.time()
    cols = feature_columns(con)
    cat_maps = category_maps(con)
    ftypes = ["c" if c in CATEGORICAL else "q" for c in cols]

    def dmatrix(split, ref=None):
        X, y, ts = load_split(con, split, cols, cat_maps)
        d = xgb.QuantileDMatrix(X, y, feature_names=cols, feature_types=ftypes,
                                enable_categorical=True, ref=ref, max_bin=256)
        n = len(y)
        del X
        gc.collect()
        print(f"[{time.time() - t0:5.1f}s] {split}: {n:,} rows, {int(y.sum()):,} laundering")
        return d, y, ts

    dtr, ytr, _ = dmatrix("train")
    dva, yva, _ = dmatrix("val", ref=dtr)
    dte, yte, ts_te = dmatrix("test", ref=dtr)
    con.close()
    n_features = len(cols)

    params = {**PARAMS, "scale_pos_weight": scale_pos_weight}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("aml-baselines")
    with mlflow.start_run(run_name="xgboost_tabular_graphfeats"):
        mlflow.log_params({**params, "max_rounds": max_rounds, "n_features": n_features})
        booster = xgb.train(
            params, dtr, num_boost_round=max_rounds,
            evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=50,
        )
        best_it = (0, booster.best_iteration + 1)
        print(f"[{time.time() - t0:5.1f}s] trained, best iteration {booster.best_iteration}")

        p_va = booster.predict(dva, iteration_range=best_it)
        p_te = booster.predict(dte, iteration_range=best_it)

        # Threshold is chosen on validation ONLY, then frozen and applied to test.
        thr, val_f1 = best_f1_threshold(yva, p_va)
        results = {
            "val": scores_at(yva, p_va, thr),
            "test": scores_at(yte, p_te, thr),
        }
        no_tail = np.asarray(ts_te < TAIL_START)
        results["test_no_tail"] = scores_at(yte[no_tail], p_te[no_tail], thr)
        results["threshold"] = thr
        results["best_iteration"] = booster.best_iteration

        for split in ["val", "test", "test_no_tail"]:
            for k in ["f1", "precision", "recall", "pr_auc"]:
                mlflow.log_metric(f"{split}_{k}", results[split][k])
        mlflow.log_metric("threshold", thr)

        imp = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False)
        out = Path("artifacts/xgb")
        out.mkdir(parents=True, exist_ok=True)
        booster.save_model(out / "model.json")
        imp.to_csv(out / "feature_importance_gain.csv", header=["gain"])
        (out / "results.json").write_text(json.dumps(results, indent=2))
        np.save(out / "test_probs.npy", p_te)
        mlflow.log_artifacts(str(out))

    print(json.dumps(results, indent=2))
    print("\nTop 15 features by gain:")
    print(imp.head(15).round(1).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/aml.duckdb")
    p.add_argument("--max-rounds", type=int, default=1000)
    p.add_argument("--scale-pos-weight", type=float, default=1.0)
    a = p.parse_args()
    main(a.db, a.max_rounds, a.scale_pos_weight)
