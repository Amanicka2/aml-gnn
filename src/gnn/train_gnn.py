"""Baseline 2: GNN edge classifier (GINE + edge updates) with neighbor sampling.

Temporal protocol (same idea as the IBM Multi-GNN benchmark):
  * training    : message passing on the TRAIN graph, predict train edges
  * validation  : message passing on TRAIN + VAL edges, predict val edges
  * test        : message passing on ALL edges, predict test edges

Usage:
    python -u -m src.gnn.train_gnn --graph data/graph
"""
import argparse
import gc
import json
import resource
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.gnn.model import EdgeGINE
from src.gnn.sampler import IncidenceCSR, sample_subgraph
from src.metrics import best_f1_threshold, scores_at

TAIL_START = pd.Timestamp("2022-09-11")


def load_graph(d):
    d = Path(d)
    g = {k: torch.from_numpy(np.load(d / f"{k}.npy")) for k in ["src", "dst", "split", "y", "ts_min"]}
    g["src"], g["dst"] = g["src"].long(), g["dst"].long()
    g["X"] = torch.from_numpy(np.load(d / "X.npy"))  # float16, cast per batch to save RAM
    g["meta"] = json.loads((d / "meta.json").read_text())
    return g


def batch_inputs(g, csr, target, fanouts, gen):
    n, ei, eids, revs, ts, td = sample_subgraph(target, g["src"], g["dst"], csr, fanouts, gen)
    msg = torch.cat([g["X"][eids].float(), revs.float()[:, None]], 1)
    tgt = torch.cat([g["X"][target].float(), torch.zeros(len(target), 1)], 1)
    return n, ei, msg, ts, td, tgt


@torch.no_grad()
def predict(model, g, csr, eids, fanouts, batch_size, seed):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(eids), batch_size):
        out.append(torch.sigmoid(model(*batch_inputs(g, csr, eids[i:i + batch_size], fanouts, gen))))
    return torch.cat(out).numpy()


def main(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    t0 = time.time()

    def log(msg):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        print(f"[{time.time() - t0:6.1f}s] [peak {peak} MB] {msg}", flush=True)

    g = load_graph(a.graph)
    N = g["meta"]["num_nodes"]
    split, y = g["split"], g["y"]
    idx = torch.arange(len(y))
    tr_pos, tr_neg = idx[(split == 0) & (y == 1)], idx[(split == 0) & (y == 0)]
    va_all, te_all = idx[split == 1], idx[split == 2]
    log(f"graph loaded: {N:,} nodes, {len(y):,} edges, {g['X'].shape[1]} edge features")

    csr_train = IncidenceCSR(g["src"], g["dst"], idx[split == 0], N)
    csr_val = IncidenceCSR(g["src"], g["dst"], idx[split <= 1], N)
    log("train and val adjacency built")

    # Fixed validation subset for picking the best epoch: all val positives + random negatives.
    rng = torch.Generator().manual_seed(a.seed)
    va_pos, va_neg = va_all[y[va_all] == 1], va_all[y[va_all] == 0]
    va_sel = torch.cat([va_pos, va_neg[torch.randperm(len(va_neg), generator=rng)[:a.val_negatives]]])

    model = EdgeGINE(g["X"].shape[1], a.hidden, a.layers, a.dropout, edge_updates=not a.no_edge_updates)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    fanouts = tuple(a.fanouts)
    out = Path("artifacts/gnn")
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"
    gen = torch.Generator().manual_seed(a.seed)

    # Resumable state: everything needed to continue exactly where we stopped.
    st = {"phase": "train", "epoch": 0, "best": -1.0, "best_state": None, "bad": 0, "run_id": None}
    if ckpt_path.exists() and not a.fresh:
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); gen.set_state(ck["gen"])
        st = ck["st"]
        log(f"resumed from checkpoint: phase={st['phase']} epoch={st['epoch']}")

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": gen.get_state(), "st": st}, ckpt_path)

    def out_of_time():
        return time.time() - t0 > a.time_budget

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("aml-baselines")
    with mlflow.start_run(run_id=st["run_id"], run_name=None if st["run_id"] else "gnn_gine_edge_updates") as run:
        if st["run_id"] is None:
            st["run_id"] = run.info.run_id
            mlflow.log_params({k: str(v) for k, v in vars(a).items()})

        while st["phase"] == "train":
            if st["epoch"] >= a.epochs or st["bad"] >= a.patience:
                st["phase"] = "score"
                break
            if out_of_time():
                save(); log("time budget reached, checkpoint saved. Run again to resume."); return
            epoch = st["epoch"] + 1
            model.train()
            # Every epoch: ALL laundering edges + a fresh random sample of normal edges.
            neg = tr_neg[torch.randperm(len(tr_neg), generator=gen)[:a.neg_ratio * len(tr_pos)]]
            tgt = torch.cat([tr_pos, neg])
            tgt = tgt[torch.randperm(len(tgt), generator=gen)]
            total = 0.0
            for i in range(0, len(tgt), a.batch_size):
                b = tgt[i:i + a.batch_size]
                logits = model(*batch_inputs(g, csr_train, b, fanouts, gen))
                loss = loss_fn(logits, y[b].float())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
                del logits, loss
            gc.collect()
            p = predict(model, g, csr_val, va_sel, fanouts, a.eval_batch_size, a.seed)
            ap = average_precision_score(y[va_sel].numpy(), p)
            mlflow.log_metric("train_loss", total / len(tgt), step=epoch)
            mlflow.log_metric("val_subset_pr_auc", ap, step=epoch)
            log(f"epoch {epoch:2d}  loss {total / len(tgt):.4f}  val-subset PR-AUC {ap:.4f}")
            st["epoch"] = epoch
            if ap > st["best"]:
                st["best"], st["bad"] = ap, 0
                st["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                st["bad"] += 1
            save()

        del csr_train
        model.load_state_dict(st["best_state"])
        for name, eids, csr_fn in [("val", va_all, lambda: csr_val),
                                   ("test", te_all, lambda: IncidenceCSR(g["src"], g["dst"], idx, N))]:
            done_file = out / f"{name}_probs.npy"
            if done_file.exists() and st.get(f"{name}_done"):
                continue
            part_file = out / f"{name}_probs_partial.npy"
            probs = np.load(part_file) if part_file.exists() else np.empty(0, np.float32)
            csr = csr_fn()
            log(f"scoring {name}: {len(probs):,}/{len(eids):,} done so far")
            step = a.eval_batch_size * 10
            while len(probs) < len(eids):
                if out_of_time():
                    np.save(part_file, probs); save()
                    log(f"time budget reached while scoring {name}. Run again to resume."); return
                chunk = eids[len(probs):len(probs) + step]
                probs = np.concatenate([probs, predict(model, g, csr, chunk, fanouts,
                                                       a.eval_batch_size, a.seed + len(probs))])
            np.save(done_file, probs)
            part_file.unlink(missing_ok=True)
            st[f"{name}_done"] = True
            save()
            if name == "val":
                del csr_val
            del csr
            gc.collect()

        p_va, p_te = np.load(out / "val_probs.npy"), np.load(out / "test_probs.npy")
        yva, yte = y[va_all].numpy(), y[te_all].numpy()
        thr, _ = best_f1_threshold(yva, p_va)
        ts_te = pd.to_datetime(g["ts_min"][te_all].numpy() * 60, unit="s")
        no_tail = np.asarray(ts_te < TAIL_START)
        results = {
            "val": scores_at(yva, p_va, thr),
            "test": scores_at(yte, p_te, thr),
            "test_no_tail": scores_at(yte[no_tail], p_te[no_tail], thr),
            "threshold": thr,
            "best_epoch_val_subset_pr_auc": st["best"],
            "epochs_trained": st["epoch"],
        }
        for s_ in ["val", "test", "test_no_tail"]:
            for k in ["f1", "precision", "recall", "pr_auc"]:
                mlflow.log_metric(f"{s_}_{k}", results[s_][k])
        mlflow.log_metric("threshold", thr)
        torch.save(st["best_state"], out / "model.pt")
        (out / "results.json").write_text(json.dumps(results, indent=2))
        st["phase"] = "done"
        save()
        mlflow.log_artifact(str(out / "results.json"))
        mlflow.log_artifact(str(out / "model.pt"))
    log("done")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="data/graph")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-edge-updates", action="store_true")
    p.add_argument("--fanouts", type=int, nargs="+", default=[15, 10])
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--neg-ratio", type=int, default=30)
    p.add_argument("--val-negatives", type=int, default=60_000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget", type=float, default=150, help="seconds before checkpointing and exiting")
    p.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    main(p.parse_args())
