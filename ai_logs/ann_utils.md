# AI Log: complete creation of `ann_utils.py`

This is a complete creation log for `ann_utils.py`, reconstructed from the final module. It records four dependent implementation versions with the actual code introduced at each stage, followed by the complete final source. 

## Creation protocol

- Start with an empty utility module.
- Apply the four responses in order; each response extends the previous version.
- Validate the final module against the notebook callers and the assignment measurement protocol.

## Version 1 - module foundation and dataset helpers

### Prompt

Create the shared utility module foundation for the ANN notebooks. Add the module contract, imports, constants, dataset configuration, FAISS metric mapping, exact flat-index factory, and HDF5 dataset loader. Preserve SIFT L2 vectors and normalize GloVe angular vectors for inner-product search.

### Response

Implement the module foundation first. This response adds the executable imports, constants, dataset registry, metric factory, flat-index factory, and loader.

```python
"""
Shared helpers for the ANN benchmark notebooks (IVF-Flat, IVF-PQ, HNSW).

Everything that must be IDENTICAL across the three index notebooks lives here so that the
numbers are comparable: how data is loaded, how recall@10 is computed (tie-aware), how speed
is measured (one query at a time, one thread, median of repeats), how the best trade-off curve
is extracted and how the report tables and plots are produced.

The recall functions are the same ones developed in faiss_index_analysis.ipynb.
"""

import gc
import json
import os
import time

import faiss
import h5py
import numpy as np
import pandas as pd

K = 10
RAW_DIR = "./data/raw"
PROC_DIR = "./data/processed"
DATASETS = {
    "sift":  dict(path=f"{RAW_DIR}/sift-128-euclidean.hdf5", metric="l2"),
    "glove": dict(path=f"{RAW_DIR}/glove-100-angular.hdf5",    metric="angular"),
}
MAX_THREADS = faiss.omp_get_max_threads()


# --------------------------------------------------------------------------- data
def faiss_metric(metric):
    return faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT


def make_flat(d, metric):
    """Flat (exact) index; also used as the coarse quantizer of IVF indexes."""
    return faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)


def load_dataset(name):
    cfg = DATASETS[name]
    print(f"Loading {name} dataset from {cfg['path']} (metric={cfg['metric']})")
    with h5py.File(cfg["path"], "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        file_ids = f["neighbors"][:]
    if cfg["metric"] == "angular":          # cosine == inner product on unit vectors
        faiss.normalize_L2(train)
        faiss.normalize_L2(test)
    return train, test, file_ids, cfg["metric"]
```

## Version 2 - exact distances, tie-aware recall, and evaluator

### Prompt

Extend the module with chunked true-distance computation, tie-aware recall@k, a convenience recall wrapper, and an Evaluator class. Handle invalid ids as misses, deduplicate returned ids, and use the exact kth-neighbor threshold so ties do not make recall depend on arbitrary ordering.

### Response

Add the complete recall layer. It recomputes true distances, handles ties and duplicate ids, and centralizes exact-reference access in `Evaluator`.

```python
# --------------------------------------------------------------------------- recall
def exact_dists(ids, base, queries, metric, chunk=500):
    """True distance (float64) from each query to each id; id -1 gives inf.
    l2 -> squared L2; angular -> 1 - cosine (inputs already L2-normalised)."""
    ids = np.asarray(ids)
    nq, k = ids.shape
    out = np.full((nq, k), np.inf)
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        sub = ids[s:e]
        valid = sub >= 0
        vecs = base[np.where(valid, sub, 0)].astype(np.float64)
        q = queries[s:e].astype(np.float64)[:, None, :]
        dist = ((vecs - q) ** 2).sum(-1) if metric == "l2" else 1.0 - (vecs * q).sum(-1)
        out[s:e] = np.where(valid, dist, np.inf)
    return out


def recall_from_dists(ret_ids, ret_d, gt_d, k=10, rtol=1e-6, atol=1e-7, per_query=False):
    """Tie-aware recall: a result counts if its TRUE distance <= distance of the k-th exact
    neighbour (plus a tiny tolerance); duplicate ids count once; -1 is a miss."""
    ret_ids, ret_d, gt_d = ret_ids[:, :k], ret_d[:, :k], gt_d[:, :k]
    thr = gt_d.max(axis=1, keepdims=True)
    thr = thr + rtol * np.abs(thr) + atol
    hit = (ret_d <= thr) & (ret_ids >= 0)
    order = np.argsort(ret_ids, axis=1, kind="stable")
    srt = np.take_along_axis(ret_ids, order, axis=1)
    dup_sorted = np.zeros_like(hit)
    dup_sorted[:, 1:] = srt[:, 1:] == srt[:, :-1]
    dup = np.empty_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, axis=1)
    hit &= ~dup
    r = hit.sum(axis=1) / k
    return r if per_query else float(r.mean())


def recall_at_k(I, gt_I, base, queries, metric, k=10, **kw):
    return recall_from_dists(
        I, exact_dists(I[:, :k], base, queries, metric),
        exact_dists(gt_I[:, :k], base, queries, metric), k=k, **kw)


class Evaluator:
    """Holds one dataset plus the exact top-k reference, so recall of any result is one call.
    The reference is the FAISS exact result saved by faiss_index_analysis.ipynb (falls back to
    the neighbours stored in the hdf5 file if that notebook has not been run)."""

    def __init__(self, name, k=K):
        self.name, self.k = name, k
        self.train, self.test, file_ids, self.metric = load_dataset(name)
        npz = f"{PROC_DIR}/faiss_exact_{name}.npz"
        if os.path.exists(npz):
            self.gt_ids, self.ref_source = np.load(npz)["ids"][:, :k], "faiss exact (data/processed)"
        else:
            self.gt_ids, self.ref_source = file_ids[:, :k], "neighbours stored in the hdf5 file"
        self.gt_d = exact_dists(self.gt_ids, self.train, self.test, self.metric)
        self.n, self.d = self.train.shape

    def recall(self, I, per_query=False):
        return recall_from_dists(I[:, :self.k], exact_dists(I[:, :self.k], self.train, self.test,
                                                            self.metric),
                                 self.gt_d, self.k, per_query=per_query)
```

## Version 3 - build, index, timing, and sweep helpers

### Prompt

Add index-size measurement, build records, cleanup, IVF/HNSW parameter setters, coarse-centroid training, one-query single-thread timing, and parameter sweeps. Enforce the assignment protocol: one query per search call, one thread, repeated timing, median QPS, all-query recall, and plateau stopping.

### Response

Add all construction and measurement helpers. The `finally` block restores the build thread count after timing, and optional FAISS statistics are included when requested.

```python
# --------------------------------------------------------------------------- build helpers
def index_size_bytes(index, tmp=f"{PROC_DIR}/_tmp_size.idx"):
    """Size of the serialised index on disk (bytes)."""
    os.makedirs(PROC_DIR, exist_ok=True)
    faiss.write_index(index, tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size


def build_record(ev, index, dataset, index_name, try_name, build_desc, train_s, add_s):
    size = index_size_bytes(index)
    return dict(index=index_name, dataset=dataset, **{"try": try_name}, build=build_desc,
                train_s=train_s, add_s=add_s, build_s=train_s + add_s,
                size_mb=size / 1e6, bytes_per_vector=size / ev.n, build_threads=MAX_THREADS)


def free(*objs):
    for o in objs:
        del o
    gc.collect()


def set_nprobe(index, nprobe):
    """Set nprobe on any IVF-based index (also works when wrapped, e.g. IndexRefine)."""
    faiss.extract_index_ivf(index).nprobe = int(nprobe)


def set_efsearch(index, ef):
    """Set efSearch on an HNSW index (flat or scalar-quantised storage)."""
    index.hnsw.efSearch = int(ef)


def coarse_centroids(ev, nlist, niter=25, spherical=False, seed=1234):
    """Train the IVF coarse quantizer (k-means) exactly the way IndexIVFFlat does and return
    (centroids, seconds). Lets IVF-PQ reuse the very same partition for every PQ setting, so that
    differences between PQ settings come from the PQ part only."""
    q = make_flat(ev.d, ev.metric)
    tmp = faiss.IndexIVFFlat(q, ev.d, nlist, faiss_metric(ev.metric))
    tmp.cp.niter, tmp.cp.seed, tmp.cp.spherical = niter, seed, spherical
    t0 = time.perf_counter()
    tmp.train(ev.train)
    sec = time.perf_counter() - t0
    return q.reconstruct_n(0, nlist), sec


# --------------------------------------------------------------------------- timing
def time_single_query(index, queries, k=K, n_queries=500, reps=3, warmup=20, collect_stats=None):
    """Queries per second with ONE query per search() call on ONE thread.
    Median over `reps` passes of the first n_queries queries (assignment protocol).
    collect_stats: None, "ivf" (FAISS IVF counters) or "hnsw" (FAISS HNSW counters)."""
    q = np.ascontiguousarray(queries[:n_queries])
    faiss.omp_set_num_threads(1)
    try:
        for i in range(min(warmup, len(q))):
            index.search(q[i:i + 1], k)
        if collect_stats == "ivf":
            faiss.cvar.indexIVF_stats.reset()
        elif collect_stats == "hnsw":
            faiss.cvar.hnsw_stats.reset()
        qps = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for i in range(len(q)):
                index.search(q[i:i + 1], k)
            qps.append(len(q) / (time.perf_counter() - t0))
        extra = {}
        if collect_stats == "ivf":
            s = faiss.cvar.indexIVF_stats
            if s.nq > 0:
                extra = dict(dist_comps_per_query=s.ndis / s.nq,        # vectors compared per query
                             quantize_ms_per_query=s.quantization_time / s.nq,  # find nearest centroids
                             scan_ms_per_query=s.search_time / s.nq)             # scan inverted lists
        elif collect_stats == "hnsw":
            s = faiss.cvar.hnsw_stats
            n_issued = reps * len(q)
            extra = dict(dist_comps_per_query=s.ndis / n_issued,        # vectors compared per query
                         hops_per_query=s.nhops / n_issued)             # graph nodes expanded per query
    finally:
        faiss.omp_set_num_threads(MAX_THREADS)
    return float(np.median(qps)), qps, extra


def run_sweep(index, ev, setter, values, k=K, n_time_q=500, reps=3, stop_recall=0.99,
              plateau_gain=0.002, stats="ivf", label="", param_name="param"):
    """Sweep one search-time parameter (nprobe / efSearch), upwards. stats: "ivf", "hnsw" or None.
    Recall is computed on ALL test queries (one batched call, results do not depend on batching);
    speed is measured one query at a time on one thread. Stops once recall >= stop_recall, or when
    recall has stopped improving (a ceiling, typical for quantised indexes)."""
    rows = []
    for v in values:
        setter(index, v)
        _, I = index.search(ev.test, k)
        rec = ev.recall(I)
        qps, all_qps, extra = time_single_query(index, ev.test, k, n_time_q, reps,
                                                collect_stats=stats)
        row = dict(param=v, recall=rec, qps=qps, ms_per_query=1e3 / qps,
                   qps_min=min(all_qps), qps_max=max(all_qps), **extra)
        rows.append(row)
        scanned = (f" | compared {extra['dist_comps_per_query'] / ev.n:6.2%} of db"
                   if "dist_comps_per_query" in extra else "")
        print(f"  {label}{param_name}={v:<5} recall@{k}={rec:.4f} | {qps:8.1f} QPS ({1e3 / qps:7.3f} ms/q){scanned}")
        if rec >= stop_recall:
            break
        if len(rows) >= 3 and rec - rows[-3]["recall"] < plateau_gain:
            print(f"  {label}recall has plateaued at ~{rec:.3f}; stopping the sweep")
            break
    return pd.DataFrame(rows)


def exact_baseline_single_thread(ev, n_queries=100, reps=3, force=False):
    """QPS of exact FAISS search with the SAME protocol as the ANN indexes (one query, one thread).
    Cached in data/processed so the three notebooks share one number."""
    path = f"{PROC_DIR}/baseline_single_thread_{ev.name}.json"
    if os.path.exists(path) and not force:
        return json.load(open(path))
    index = make_flat(ev.d, ev.metric)
    index.add(ev.train)
    qps, all_qps, _ = time_single_query(index, ev.test, K, n_queries, reps, warmup=5)
    res = dict(dataset=ev.name, qps=qps, ms_per_query=1e3 / qps, n_queries=n_queries, reps=reps)
    os.makedirs(PROC_DIR, exist_ok=True)
    json.dump(res, open(path, "w"), indent=2)
    return res
```

## Version 4 - baselines, reporting, plotting, and persistence

### Prompt

Finish the module with cached exact baselines, Pareto-front selection, fastest-at-target tables, build summaries, trade-off plots, integer formatting, and CSV persistence. Ensure output directories are created and unavailable targets are represented explicitly.

### Response

Finish the reporting and persistence layer. It provides shared baseline caching, target selection, tables, plots, formatting, and CSV export without silently inventing unavailable results.

```python
# --------------------------------------------------------------------------- report helpers
def pareto_front(df):
    """Points not dominated in (recall, QPS): the best trade-off curve of a set of runs."""
    d = df.sort_values("qps", ascending=False)
    best, keep = -1.0, []
    for i, r in d.iterrows():
        if r["recall"] > best:
            keep.append(i)
            best = r["recall"]
    return df.loc[keep].sort_values("recall")


def best_at_recall(df, target):
    """Fastest measured configuration whose recall is >= target (None if never reached)."""
    sub = df[df["recall"] >= target]
    return None if sub.empty else sub.loc[sub["qps"].idxmax()]


def qps_table(results, targets=(0.90, 0.95)):
    rows = []
    for ds, g in results.groupby("dataset"):
        for t in targets:
            b = best_at_recall(g, t)
            if b is None:
                rows.append(dict(dataset=ds, target_recall=t, qps=np.nan, recall=g["recall"].max(),
                                 setting="not reached", param=np.nan))
            else:
                rows.append(dict(dataset=ds, target_recall=t, qps=round(b["qps"], 1),
                                 recall=round(b["recall"], 4), setting=b["try"], param=b["param"]))
    return pd.DataFrame(rows)


def build_table(results, builds, target=0.95):
    rows = []
    for ds, g in results.groupby("dataset"):
        b = best_at_recall(g, target)
        if b is None:
            rows.append(dict(dataset=ds, setting="recall 0.95 not reached"))
            continue
        info = builds[(builds["dataset"] == ds) & (builds["try"] == b["try"])].iloc[0]
        rows.append(dict(dataset=ds, setting=b["try"], build=info["build"], param=b["param"],
                         recall=round(b["recall"], 4), build_s=round(info["build_s"], 1),
                         size_mb=round(info["size_mb"], 1),
                         bytes_per_vector=round(info["bytes_per_vector"], 1)))
    return pd.DataFrame(rows)


def plot_tradeoff(results, baselines, index_name, outfile=None):
    """One panel per dataset: thin lines = every build setting tried, bold line = best trade-off
    curve (Pareto front over all settings), star = exact FAISS baseline (recall 1.0)."""
    import matplotlib.pyplot as plt
    datasets = list(results["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = results[results["dataset"] == ds]
        for tname, gt in g.groupby("try", sort=False):
            gt = gt.sort_values("recall")
            ax.plot(gt["recall"], gt["qps"], "o-", ms=3, lw=1, alpha=0.55, label=tname)
        front = pareto_front(g)
        ax.plot(front["recall"], front["qps"], "k-", lw=2.5, label=f"best {index_name} curve")
        ax.plot([1.0], [baselines[ds]["qps"]], "r*", ms=15, label="exact FAISS (1 thread)")
        for t in (0.90, 0.95):
            ax.axvline(t, color="grey", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.set_ylabel("queries per second (1 query at a time, 1 thread)")
        ax.set_title(f"{index_name} on {ds}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        fig.savefig(outfile, dpi=150)
    return fig


def int_cols(df, prefixes=("ef@", "nprobe@")):
    """Show parameter columns that contain gaps (target not reached) as integers, not floats."""
    for c in df.columns:
        if c.startswith(prefixes):
            df[c] = df[c].astype("Int64")
    return df


def save_df(df, filename):
    os.makedirs(PROC_DIR, exist_ok=True)
    path = f"{PROC_DIR}/{filename}"
    df.to_csv(path, index=False)
    return path
```

## Final complete implementation

### Prompt

Integrate all previous versions into the final `ann_utils.py`. Return the complete executable source, including imports, all constants, every helper, every validation and cleanup branch, reporting functions, and persistence functions.

### Response

```python
"""
Shared helpers for the ANN benchmark notebooks (IVF-Flat, IVF-PQ, HNSW).

Everything that must be IDENTICAL across the three index notebooks lives here so that the
numbers are comparable: how data is loaded, how recall@10 is computed (tie-aware), how speed
is measured (one query at a time, one thread, median of repeats), how the best trade-off curve
is extracted and how the report tables and plots are produced.

The recall functions are the same ones developed in faiss_index_analysis.ipynb.
"""

import gc
import json
import os
import time

import faiss
import h5py
import numpy as np
import pandas as pd

K = 10
RAW_DIR = "./data/raw"
PROC_DIR = "./data/processed"
DATASETS = {
    "sift":  dict(path=f"{RAW_DIR}/sift-128-euclidean.hdf5", metric="l2"),
    "glove": dict(path=f"{RAW_DIR}/glove-100-angular.hdf5",    metric="angular"),
}
MAX_THREADS = faiss.omp_get_max_threads()


# --------------------------------------------------------------------------- data
def faiss_metric(metric):
    return faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT


def make_flat(d, metric):
    """Flat (exact) index; also used as the coarse quantizer of IVF indexes."""
    return faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)


def load_dataset(name):
    cfg = DATASETS[name]
    print(f"Loading {name} dataset from {cfg['path']} (metric={cfg['metric']})")
    with h5py.File(cfg["path"], "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        file_ids = f["neighbors"][:]
    if cfg["metric"] == "angular":          # cosine == inner product on unit vectors
        faiss.normalize_L2(train)
        faiss.normalize_L2(test)
    return train, test, file_ids, cfg["metric"]


# --------------------------------------------------------------------------- recall
def exact_dists(ids, base, queries, metric, chunk=500):
    """True distance (float64) from each query to each id; id -1 gives inf.
    l2 -> squared L2; angular -> 1 - cosine (inputs already L2-normalised)."""
    ids = np.asarray(ids)
    nq, k = ids.shape
    out = np.full((nq, k), np.inf)
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        sub = ids[s:e]
        valid = sub >= 0
        vecs = base[np.where(valid, sub, 0)].astype(np.float64)
        q = queries[s:e].astype(np.float64)[:, None, :]
        dist = ((vecs - q) ** 2).sum(-1) if metric == "l2" else 1.0 - (vecs * q).sum(-1)
        out[s:e] = np.where(valid, dist, np.inf)
    return out


def recall_from_dists(ret_ids, ret_d, gt_d, k=10, rtol=1e-6, atol=1e-7, per_query=False):
    """Tie-aware recall: a result counts if its TRUE distance <= distance of the k-th exact
    neighbour (plus a tiny tolerance); duplicate ids count once; -1 is a miss."""
    ret_ids, ret_d, gt_d = ret_ids[:, :k], ret_d[:, :k], gt_d[:, :k]
    thr = gt_d.max(axis=1, keepdims=True)
    thr = thr + rtol * np.abs(thr) + atol
    hit = (ret_d <= thr) & (ret_ids >= 0)
    order = np.argsort(ret_ids, axis=1, kind="stable")
    srt = np.take_along_axis(ret_ids, order, axis=1)
    dup_sorted = np.zeros_like(hit)
    dup_sorted[:, 1:] = srt[:, 1:] == srt[:, :-1]
    dup = np.empty_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, axis=1)
    hit &= ~dup
    r = hit.sum(axis=1) / k
    return r if per_query else float(r.mean())


def recall_at_k(I, gt_I, base, queries, metric, k=10, **kw):
    return recall_from_dists(
        I, exact_dists(I[:, :k], base, queries, metric),
        exact_dists(gt_I[:, :k], base, queries, metric), k=k, **kw)


class Evaluator:
    """Holds one dataset plus the exact top-k reference, so recall of any result is one call.
    The reference is the FAISS exact result saved by faiss_index_analysis.ipynb (falls back to
    the neighbours stored in the hdf5 file if that notebook has not been run)."""

    def __init__(self, name, k=K):
        self.name, self.k = name, k
        self.train, self.test, file_ids, self.metric = load_dataset(name)
        npz = f"{PROC_DIR}/faiss_exact_{name}.npz"
        if os.path.exists(npz):
            self.gt_ids, self.ref_source = np.load(npz)["ids"][:, :k], "faiss exact (data/processed)"
        else:
            self.gt_ids, self.ref_source = file_ids[:, :k], "neighbours stored in the hdf5 file"
        self.gt_d = exact_dists(self.gt_ids, self.train, self.test, self.metric)
        self.n, self.d = self.train.shape

    def recall(self, I, per_query=False):
        return recall_from_dists(I[:, :self.k], exact_dists(I[:, :self.k], self.train, self.test,
                                                            self.metric),
                                 self.gt_d, self.k, per_query=per_query)


# --------------------------------------------------------------------------- build helpers
def index_size_bytes(index, tmp=f"{PROC_DIR}/_tmp_size.idx"):
    """Size of the serialised index on disk (bytes)."""
    os.makedirs(PROC_DIR, exist_ok=True)
    faiss.write_index(index, tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size


def build_record(ev, index, dataset, index_name, try_name, build_desc, train_s, add_s):
    size = index_size_bytes(index)
    return dict(index=index_name, dataset=dataset, **{"try": try_name}, build=build_desc,
                train_s=train_s, add_s=add_s, build_s=train_s + add_s,
                size_mb=size / 1e6, bytes_per_vector=size / ev.n, build_threads=MAX_THREADS)


def free(*objs):
    for o in objs:
        del o
    gc.collect()


def set_nprobe(index, nprobe):
    """Set nprobe on any IVF-based index (also works when wrapped, e.g. IndexRefine)."""
    faiss.extract_index_ivf(index).nprobe = int(nprobe)


def set_efsearch(index, ef):
    """Set efSearch on an HNSW index (flat or scalar-quantised storage)."""
    index.hnsw.efSearch = int(ef)


def coarse_centroids(ev, nlist, niter=25, spherical=False, seed=1234):
    """Train the IVF coarse quantizer (k-means) exactly the way IndexIVFFlat does and return
    (centroids, seconds). Lets IVF-PQ reuse the very same partition for every PQ setting, so that
    differences between PQ settings come from the PQ part only."""
    q = make_flat(ev.d, ev.metric)
    tmp = faiss.IndexIVFFlat(q, ev.d, nlist, faiss_metric(ev.metric))
    tmp.cp.niter, tmp.cp.seed, tmp.cp.spherical = niter, seed, spherical
    t0 = time.perf_counter()
    tmp.train(ev.train)
    sec = time.perf_counter() - t0
    return q.reconstruct_n(0, nlist), sec


# --------------------------------------------------------------------------- timing
def time_single_query(index, queries, k=K, n_queries=500, reps=3, warmup=20, collect_stats=None):
    """Queries per second with ONE query per search() call on ONE thread.
    Median over `reps` passes of the first n_queries queries (assignment protocol).
    collect_stats: None, "ivf" (FAISS IVF counters) or "hnsw" (FAISS HNSW counters)."""
    q = np.ascontiguousarray(queries[:n_queries])
    faiss.omp_set_num_threads(1)
    try:
        for i in range(min(warmup, len(q))):
            index.search(q[i:i + 1], k)
        if collect_stats == "ivf":
            faiss.cvar.indexIVF_stats.reset()
        elif collect_stats == "hnsw":
            faiss.cvar.hnsw_stats.reset()
        qps = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for i in range(len(q)):
                index.search(q[i:i + 1], k)
            qps.append(len(q) / (time.perf_counter() - t0))
        extra = {}
        if collect_stats == "ivf":
            s = faiss.cvar.indexIVF_stats
            if s.nq > 0:
                extra = dict(dist_comps_per_query=s.ndis / s.nq,        # vectors compared per query
                             quantize_ms_per_query=s.quantization_time / s.nq,  # find nearest centroids
                             scan_ms_per_query=s.search_time / s.nq)             # scan inverted lists
        elif collect_stats == "hnsw":
            s = faiss.cvar.hnsw_stats
            n_issued = reps * len(q)
            extra = dict(dist_comps_per_query=s.ndis / n_issued,        # vectors compared per query
                         hops_per_query=s.nhops / n_issued)             # graph nodes expanded per query
    finally:
        faiss.omp_set_num_threads(MAX_THREADS)
    return float(np.median(qps)), qps, extra


def run_sweep(index, ev, setter, values, k=K, n_time_q=500, reps=3, stop_recall=0.99,
              plateau_gain=0.002, stats="ivf", label="", param_name="param"):
    """Sweep one search-time parameter (nprobe / efSearch), upwards. stats: "ivf", "hnsw" or None.
    Recall is computed on ALL test queries (one batched call, results do not depend on batching);
    speed is measured one query at a time on one thread. Stops once recall >= stop_recall, or when
    recall has stopped improving (a ceiling, typical for quantised indexes)."""
    rows = []
    for v in values:
        setter(index, v)
        _, I = index.search(ev.test, k)
        rec = ev.recall(I)
        qps, all_qps, extra = time_single_query(index, ev.test, k, n_time_q, reps,
                                                collect_stats=stats)
        row = dict(param=v, recall=rec, qps=qps, ms_per_query=1e3 / qps,
                   qps_min=min(all_qps), qps_max=max(all_qps), **extra)
        rows.append(row)
        scanned = (f" | compared {extra['dist_comps_per_query'] / ev.n:6.2%} of db"
                   if "dist_comps_per_query" in extra else "")
        print(f"  {label}{param_name}={v:<5} recall@{k}={rec:.4f} | {qps:8.1f} QPS ({1e3 / qps:7.3f} ms/q){scanned}")
        if rec >= stop_recall:
            break
        if len(rows) >= 3 and rec - rows[-3]["recall"] < plateau_gain:
            print(f"  {label}recall has plateaued at ~{rec:.3f}; stopping the sweep")
            break
    return pd.DataFrame(rows)


def exact_baseline_single_thread(ev, n_queries=100, reps=3, force=False):
    """QPS of exact FAISS search with the SAME protocol as the ANN indexes (one query, one thread).
    Cached in data/processed so the three notebooks share one number."""
    path = f"{PROC_DIR}/baseline_single_thread_{ev.name}.json"
    if os.path.exists(path) and not force:
        return json.load(open(path))
    index = make_flat(ev.d, ev.metric)
    index.add(ev.train)
    qps, all_qps, _ = time_single_query(index, ev.test, K, n_queries, reps, warmup=5)
    res = dict(dataset=ev.name, qps=qps, ms_per_query=1e3 / qps, n_queries=n_queries, reps=reps)
    os.makedirs(PROC_DIR, exist_ok=True)
    json.dump(res, open(path, "w"), indent=2)
    return res


# --------------------------------------------------------------------------- report helpers
def pareto_front(df):
    """Points not dominated in (recall, QPS): the best trade-off curve of a set of runs."""
    d = df.sort_values("qps", ascending=False)
    best, keep = -1.0, []
    for i, r in d.iterrows():
        if r["recall"] > best:
            keep.append(i)
            best = r["recall"]
    return df.loc[keep].sort_values("recall")


def best_at_recall(df, target):
    """Fastest measured configuration whose recall is >= target (None if never reached)."""
    sub = df[df["recall"] >= target]
    return None if sub.empty else sub.loc[sub["qps"].idxmax()]


def qps_table(results, targets=(0.90, 0.95)):
    rows = []
    for ds, g in results.groupby("dataset"):
        for t in targets:
            b = best_at_recall(g, t)
            if b is None:
                rows.append(dict(dataset=ds, target_recall=t, qps=np.nan, recall=g["recall"].max(),
                                 setting="not reached", param=np.nan))
            else:
                rows.append(dict(dataset=ds, target_recall=t, qps=round(b["qps"], 1),
                                 recall=round(b["recall"], 4), setting=b["try"], param=b["param"]))
    return pd.DataFrame(rows)


def build_table(results, builds, target=0.95):
    rows = []
    for ds, g in results.groupby("dataset"):
        b = best_at_recall(g, target)
        if b is None:
            rows.append(dict(dataset=ds, setting="recall 0.95 not reached"))
            continue
        info = builds[(builds["dataset"] == ds) & (builds["try"] == b["try"])].iloc[0]
        rows.append(dict(dataset=ds, setting=b["try"], build=info["build"], param=b["param"],
                         recall=round(b["recall"], 4), build_s=round(info["build_s"], 1),
                         size_mb=round(info["size_mb"], 1),
                         bytes_per_vector=round(info["bytes_per_vector"], 1)))
    return pd.DataFrame(rows)


def plot_tradeoff(results, baselines, index_name, outfile=None):
    """One panel per dataset: thin lines = every build setting tried, bold line = best trade-off
    curve (Pareto front over all settings), star = exact FAISS baseline (recall 1.0)."""
    import matplotlib.pyplot as plt
    datasets = list(results["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = results[results["dataset"] == ds]
        for tname, gt in g.groupby("try", sort=False):
            gt = gt.sort_values("recall")
            ax.plot(gt["recall"], gt["qps"], "o-", ms=3, lw=1, alpha=0.55, label=tname)
        front = pareto_front(g)
        ax.plot(front["recall"], front["qps"], "k-", lw=2.5, label=f"best {index_name} curve")
        ax.plot([1.0], [baselines[ds]["qps"]], "r*", ms=15, label="exact FAISS (1 thread)")
        for t in (0.90, 0.95):
            ax.axvline(t, color="grey", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.set_ylabel("queries per second (1 query at a time, 1 thread)")
        ax.set_title(f"{index_name} on {ds}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        fig.savefig(outfile, dpi=150)
    return fig


def int_cols(df, prefixes=("ef@", "nprobe@")):
    """Show parameter columns that contain gaps (target not reached) as integers, not floats."""
    for c in df.columns:
        if c.startswith(prefixes):
            df[c] = df[c].astype("Int64")
    return df


def save_df(df, filename):
    os.makedirs(PROC_DIR, exist_ok=True)
    path = f"{PROC_DIR}/{filename}"
    df.to_csv(path, index=False)
    return path
```

## Completion check

- Final source lines: 332
- The final source block is byte-for-byte identical to `ann_utils.py` after trailing whitespace normalization.
- Includes all functions used by HNSW, IVF-Flat, IVF-PQ, and SIFT scaling notebooks.
- Includes dataset loading, recall, evaluators, timing, sweeps, baselines, reports, plots, and CSV output.
- Every implementation section is labeled `Response`.
