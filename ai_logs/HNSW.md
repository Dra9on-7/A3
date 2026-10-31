# AI Log: complete creation of `HNSW.ipynb`

This log reconstructs the notebook from an empty file to its final executed state. Each cell has a implementation prompt followed by a `Response` containing the complete source for that cell. Outputs are included when the notebook captured them.

## Versioned implementation milestones

1. **Version 1 > notebook setup and shared evaluator**
2. **Version 2 > HNSW builders and graph measurements**
3. **Version 3 > efSearch sweeps, timing, and recall curves**
4. **Version 4 > result tables, plots, exports, and validation**

## Creation sequence

- Notebook cells: 31
- Markdown cells: 17
- Code cells: 14
- Execution order: top to bottom; each response depends on prior cells.

## Cell 1 > markdown

### Prompt

Create notebook cell 1 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
# Assignemnt 3: ANN analysis

## Problem statement:
Finding the stored vectors closest to a query is the core step of dense retrieval. Doing it exactly means comparing the query against every vector, which gets too slow at scale. Approximate nearest neighbour (ANN) indexes are much faster, at the price of sometimes missing a true neighbour. In this assignment you will measure that trade-off yourself. You will benchmark three popular index types, HNSW, IVF-Flat and IVF-PQ, on two public datasets, run one experiment of your choice from a short list, carry out one open-ended experiment of your own on data you generate, and write a short report explaining what you found. We care about careful measurement and clear explanations, not about which index turns out fastest.
~~~~

## Cell 2 > markdown

### Prompt

Create notebook cell 2 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Index Type 1: HNSW
HNSW (Hierarchical Navigable Small World) is a graph-based index that organizes vectors in a multi-layered graph structure. It allows for efficient nearest neighbor search by navigating through the graph layers, starting from a high-level overview and progressively refining the search in lower layers. HNSW is known for its high recall and low latency, making it suitable for large-scale retrieval tasks.

The logic behind HNSW is that it builds a small-world graph where each node (vector) is connected to a limited number of neighbors. During the search process, the algorithm starts from an entry point and traverses the graph, exploring neighboring nodes to find the closest vectors to the query. The hierarchical structure allows for efficient pruning of the search space, leading to faster retrieval times compared to exhaustive search methods.

The implementation of HNSW typically involves parameters such as the **number of layers**, **the number of connections per node**, and the **search depth**. These parameters can be tuned to balance between search speed and accuracy, depending on the specific requirements of the application.
~~~~

## Cell 3 > markdown

### Prompt

Create notebook cell 3 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Measurement protocol (identical for all three indexes)

The same rules as in the IVF-Flat and IVF-PQ notebooks are used, implemented once in `ann_utils.py`.

| Item | Rule used |
|---|---|
| Quality | **recall@10**, tie-aware, against the FAISS exact reference |
| Speed | **queries per second, one query per `search()` call, one thread**, 3 repeats, **median** |
| Sweep | the search-time parameter, here **`efSearch`**, is increased until recall reaches about 0.99, **or until recall stops improving** (a plateau means the graph, not the search budget, is the limit) |
| Recall vs timing | recall on all 10,000 queries (one batched call, results do not depend on batching); timing on the first 500 queries |
| Baseline | exact FAISS search measured with the same single-query, single-thread protocol |

Two points specific to HNSW:
- **Building is the expensive part.** Every vector is inserted into the graph by searching the graph itself, so building 1M vectors takes minutes per index even with all CPU threads. Build time is reported together with the thread count.
- **Parallel construction is not deterministic.** With several threads, vectors are inserted in an interleaved order that differs from run to run, so two builds with the same settings give slightly different graphs. Try 1b therefore rebuilds the base configuration once more, to measure that noise before any difference between Tries is interpreted.
~~~~

## Cell 4 > code

### Prompt

Implement notebook code cell 4 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell imports the shared helpers (ann_utils.py), sets the experiment configuration, loads both datasets with their exact reference results and measures the exact-search baseline with the single-query, single-thread protocol. No index is built yet.
"""

import gc
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ann_utils import (K, DATASETS, MAX_THREADS, Evaluator, faiss_metric, build_record,
                       run_sweep, set_efsearch, exact_baseline_single_thread,
                       best_at_recall, qps_table, build_table, plot_tradeoff, save_df, int_cols)

# ---- experiment configuration -------------------------------------------------------------
N_TIME_Q = 500        # queries timed per repeat (the first 500 test queries)
REPS = 3              # repeats per timing; the median QPS is reported
STOP_RECALL = 0.99    # stop a sweep once recall reaches this
SEED = 1234
# efSearch below k = 10 behaves like k, so the sweep starts at 10
EFSEARCH_GRID = [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 192, 256, 384, 512, 768, 1024]

M_BASE, M_SMALL, M_LARGE = 16, 8, 32            # links per node (level 0 holds up to 2M)
EFC_BASE, EFC_LOW, EFC_HIGH = 100, 40, 200      # efConstruction

# ---- data, reference and exact baseline ---------------------------------------------------
EVAL = {name: Evaluator(name) for name in DATASETS}
BASELINE = {name: exact_baseline_single_thread(EVAL[name]) for name in DATASETS}

RESULTS, BUILDS = [], []

rows = []
for name, ev in EVAL.items():
    rows.append({"dataset": name, "metric": ev.metric, "N": f"{ev.n:,}", "dim": ev.d,
                 "reference": ev.ref_source, "exact bytes/vector": 4 * ev.d,
                 "exact QPS (1 query, 1 thread)": round(BASELINE[name]["qps"], 1)})
print(f"FAISS {faiss.__version__} | threads available for building: {MAX_THREADS} | timing: 1 thread")
print(int_cols(pd.DataFrame(rows)).to_string(index=False))
~~~~

### Captured output

~~~~text
Loading sift dataset from ./data/raw/sift-128-euclidean.hdf5 (metric=l2)
Loading glove dataset from ./data/raw/glove-100-angular.hdf5 (metric=angular)
FAISS 1.15.1 | threads available for building: 20 | timing: 1 thread
dataset  metric         N  dim                    reference  exact bytes/vector  exact QPS (1 query, 1 thread)
   sift      l2 1,000,000  128 faiss exact (data/processed)                 512                           29.7
  glove angular 1,183,514  100 faiss exact (data/processed)                 400                           32.6
~~~~

Execution count recorded by the notebook: `1`.

## Cell 5 > markdown

### Prompt

Create notebook cell 5 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## What HNSW is made of, and what each part costs

HNSW does not partition the data. It builds a **graph** in which every vector is a node linked to a few nearby vectors, and a query finds its neighbours by **walking** through that graph.

| Component | Knob | What it controls | What it costs |
|---|---|---|---|
| **Links per node** | `M` | connectivity: each node keeps up to `2M` links on layer 0 and `M` on upper layers | memory (`~ 8M` bytes per vector for the links), build time, and the work per step of a search (more links to check) |
| **Layers** | automatic | each node gets a random top layer; layer `l` holds roughly `N / M^l` nodes, so the upper layers are tiny "express roads" | a few bytes per vector |
| **Construction search** | `efConstruction` | how many candidates are considered when a new vector chooses its links, i.e. **graph quality** | build time only (it does **not** change the index size) |
| **Link selection** | built in | a heuristic keeps links that point in *different* directions rather than the nearest ones only, so the graph stays navigable | nothing extra |
| **Query search** | `efSearch` | size of the candidate list of the best-first search on layer 0, the speed/recall dial | query time grows roughly linearly with it |
| **Storage** | "Flat" = raw float32 vectors | distances along the walk are **exact** | memory `4d` bytes per vector on top of the links |

**How a query works.** The search starts at the single entry point on the top layer and moves greedily to the closest node it can reach, layer by layer, which brings it close to the query in a handful of steps. On layer 0 it switches to a best-first search that keeps `efSearch` candidates and expands the best unexplored one until no candidate can be improved. The top 10 candidates are returned.

**Cost model.** The work per query is about `efSearch x (links per node)` distance computations, which depends on the graph and `efSearch` but hardly on `N` (the walk grows roughly with `log N`). IVF, by contrast, scans a *fixed fraction* of the database, so its work grows linearly with `N`. This is why HNSW compares only a tiny share of the database per query, and also why it needs **more memory** (links) and **a much longer build**.

**Why HNSW can fail.** Distances are exact, so a true neighbour is missed only if the walk never reaches it: the search ends in a region of the graph that does not lead to it (too small `efSearch`), or the graph itself has weak or missing paths to it (too few links or poor construction). Both are tested below.

Each Try changes **one thing** relative to the base, so any change in the curves can be attributed to that component. Every Try has a markdown cell with the decision, the reasoning, the alternatives that were not chosen and a hypothesis to check against the output.
~~~~

## Cell 6 > code

### Prompt

Implement notebook code cell 6 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell defines the helpers used by every Try. build_hnsw builds and times one index, graph_stats reads the layer sizes and the average number of links on layer 0 out of the finished graph, and run_try_hnsw builds the index for each dataset, records build statistics, sweeps efSearch with the single-query single-thread protocol and stores everything in RESULTS / BUILDS.
"""

def build_hnsw(ev, M, efc, sq8=False):
    if sq8:   # same graph, but vectors stored as 8-bit scalar-quantised codes (needs a training step)
        index = faiss.IndexHNSWSQ(ev.d, faiss.ScalarQuantizer.QT_8bit, M, faiss_metric(ev.metric))
        sample = ev.train[np.random.default_rng(SEED).choice(ev.n, min(100_000, ev.n), replace=False)]
        t0 = time.perf_counter(); index.train(sample); train_s = time.perf_counter() - t0
    else:
        index = faiss.IndexHNSWFlat(ev.d, M, faiss_metric(ev.metric))
        train_s = 0.0                                           # a graph over raw vectors needs no training
    index.hnsw.efConstruction = efc
    t0 = time.perf_counter(); index.add(ev.train); add_s = time.perf_counter() - t0       # builds the graph
    return index, train_s, add_s


def graph_stats(index):
    """Nodes per layer and the average number of links actually used on layer 0."""
    h = index.hnsw
    levels = faiss.vector_to_array(h.levels)                    # stores (top layer + 1) for each node
    layer_sizes = [int((levels > l).sum()) for l in range(h.max_level + 1)]
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    offs = faiss.vector_to_array(h.offsets).astype(np.int64)
    nbrs = faiss.vector_to_array(h.neighbors)
    width, n, filled = int(cum[1] - cum[0]), len(offs) - 1, 0
    for s in range(0, n, 200_000):
        idx = offs[s:min(s + 200_000, n), None] + np.arange(width)
        filled += int((nbrs[idx] >= 0).sum())
    return layer_sizes, filled / n


def run_try_hnsw(try_name, M, efc, sq8=False):
    print("=" * 100)
    print(try_name)
    for name in DATASETS:
        ev = EVAL[name]
        print(f"\n[{name}] building HNSW: M={M}, efConstruction={efc}, storage={'8-bit scalar quantised' if sq8 else 'flat float32'}"
              f" (this takes a while)")
        index, train_s, add_s = build_hnsw(ev, M, efc, sq8)
        layers, deg0 = graph_stats(index)
        rec = build_record(ev, index, name, "HNSW", try_name, f"M={M}, efConstruction={efc}, sq8={sq8}", train_s, add_s)
        rec.update(M=M, efC=efc, layers=" > ".join(f"{s:,}" for s in layers), n_layers=len(layers),
                   mean_links_layer0=deg0, is_variant=sq8)
        BUILDS.append(rec)
        print(f"[{name}] build: {train_s + add_s:.1f} s | size {rec['size_mb']:.0f} MB = {rec['bytes_per_vector']:.0f} bytes/vector "
              f"(raw vector {4 * ev.d if not sq8 else ev.d} B, the rest is the graph)")
        print(f"[{name}] nodes per layer (layer 0 first): {rec['layers']}")
        print(f"[{name}] layer 0 uses {deg0:.1f} of {2 * M} link slots per node on average")

        df = run_sweep(index, ev, set_efsearch, EFSEARCH_GRID, n_time_q=N_TIME_Q, reps=REPS,
                       stop_recall=STOP_RECALL, stats="hnsw", param_name="efSearch", label=f"[{name}] ")
        df.insert(0, "try", try_name); df.insert(0, "dataset", name); df.insert(0, "index", "HNSW")
        df["is_variant"] = sq8
        RESULTS.append(df)
        for t in (0.90, 0.95):
            b = best_at_recall(df, t)
            msg = (f"efSearch={int(b['param'])} (recall {b['recall']:.4f}, {b['qps']:.0f} QPS)" if b is not None
                   else f"NOT reached (max recall {df['recall'].max():.4f})")
            print(f"[{name}] recall >= {t}: {msg}")
        del index
        gc.collect()

print("helpers defined")
~~~~

### Captured output

~~~~text
helpers defined
~~~~

Execution count recorded by the notebook: `2`.

## Cell 7 > markdown

### Prompt

Create notebook cell 7 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 1: base configuration, `M = 16`, `efConstruction = 100`

**Decision.** 16 links per node (32 on layer 0), construction candidate list of 100, flat float32 storage, `efSearch` swept from 10 upwards.

**Reasoning.**
- `M = 16` is the most common default and sits in the middle of the sensible range (8 to 64). It is large enough for 100- to 128-dimensional data to be navigable and small enough that the links cost about 130 bytes per vector.
- `efConstruction = 100` is several times `M`, which is the usual rule of thumb for a decent graph; it keeps a 1M-vector build in the range of minutes rather than hours.
- It is the reference every other Try is compared with.

**Alternatives not chosen as the base.**
- *`M = 32` with `efConstruction = 200` (the setting often quoted for best quality):* better recall but roughly 4x the build time and 2x the link memory. It is tested separately (Try 3, Try 5) rather than assumed.
- *`M = 4` or `M = 6`:* graph is too sparse for high-dimensional data; the sweep would mainly show failures to connect.
- *Starting `efSearch` below 10:* FAISS needs at least `k = 10` candidates to return ten results, so values below 10 behave exactly like 10 and would add nothing.

**Hypothesis.** Recall rises steeply with the first few `efSearch` values and then flattens, while QPS falls roughly in proportion to `efSearch`. SIFT should reach 0.95 with a small `efSearch`; GloVe, which is a harder dataset for graph navigation, should need clearly more. The number of distances computed per query should be a tiny fraction of the database.
~~~~

## Cell 8 > code

### Prompt

Implement notebook code cell 8 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 1: M = 16, efConstruction = 100. The base every other Try is compared with.
"""
run_try_hnsw("Try 1: M=16, efC=100 (base)", M_BASE, EFC_BASE)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 1: M=16, efC=100 (base)

[sift] building HNSW: M=16, efConstruction=100, storage=flat float32 (this takes a while)
[sift] build: 32.5 s | size 656 MB = 656 bytes/vector (raw vector 512 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 62,403 > 3,903 > 210 > 16 > 1
[sift] layer 0 uses 19.6 of 32 link slots per node on average
  [sift] efSearch=10    recall@10=0.7003 |  12875.8 QPS (  0.078 ms/q) | compared  0.04% of db
  [sift] efSearch=12    recall@10=0.7373 |  13630.1 QPS (  0.073 ms/q) | compared  0.04% of db
  [sift] efSearch=16    recall@10=0.7910 |  10673.5 QPS (  0.094 ms/q) | compared  0.05% of db
  [sift] efSearch=20    recall@10=0.8289 |   9863.9 QPS (  0.101 ms/q) | compared  0.06% of db
  [sift] efSearch=24    recall@10=0.8566 |  10050.4 QPS (  0.099 ms/q) | compared  0.06% of db
  [sift] efSearch=32    recall@10=0.8946 |   9196.7 QPS (  0.109 ms/q) | compared  0.08% of db
  [sift] efSearch=40    recall@10=0.9197 |   5499.7 QPS (  0.182 ms/q) | compared  0.09% of db
  [sift] efSearch=48    recall@10=0.9354 |   4866.3 QPS (  0.205 ms/q) | compared  0.10% of db
  [sift] efSearch=64    recall@10=0.9568 |   4063.6 QPS (  0.246 ms/q) | compared  0.12% of db
  [sift] efSearch=80    recall@10=0.9692 |   3390.3 QPS (  0.295 ms/q) | compared  0.15% of db
  [sift] efSearch=96    recall@10=0.9766 |   2724.4 QPS (  0.367 ms/q) | compared  0.17% of db
  [sift] efSearch=128   recall@10=0.9854 |   2324.0 QPS (  0.430 ms/q) | compared  0.22% of db
  [sift] efSearch=192   recall@10=0.9930 |   1925.1 QPS (  0.519 ms/q) | compared  0.30% of db
[sift] recall >= 0.9: efSearch=40 (recall 0.9197, 5500 QPS)
[sift] recall >= 0.95: efSearch=64 (recall 0.9568, 4064 QPS)

[glove] building HNSW: M=16, efConstruction=100, storage=flat float32 (this takes a while)
[glove] build: 39.6 s | size 644 MB = 544 bytes/vector (raw vector 400 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 74,078 > 4,644 > 260 > 18 > 1
[glove] layer 0 uses 21.1 of 32 link slots per node on average
  [glove] efSearch=10    recall@10=0.4522 |  13938.5 QPS (  0.072 ms/q) | compared  0.04% of db
  [glove] efSearch=12    recall@10=0.4850 |  12366.9 QPS (  0.081 ms/q) | compared  0.04% of db
  [glove] efSearch=16    recall@10=0.5345 |  10325.9 QPS (  0.097 ms/q) | compared  0.05% of db
  [glove] efSearch=20    recall@10=0.5727 |   8837.8 QPS (  0.113 ms/q) | compared  0.06% of db
  [glove] efSearch=24    recall@10=0.6029 |  10689.6 QPS (  0.094 ms/q) | compared  0.07% of db
  [glove] efSearch=32    recall@10=0.6472 |   6962.3 QPS (  0.144 ms/q) | compared  0.08% of db
  [glove] efSearch=40    recall@10=0.6800 |   6003.2 QPS (  0.167 ms/q) | compared  0.09% of db
  [glove] efSearch=48    recall@10=0.7044 |   5438.8 QPS (  0.184 ms/q) | compared  0.11% of db
  [glove] efSearch=64    recall@10=0.7399 |   4106.0 QPS (  0.244 ms/q) | compared  0.13% of db
  [glove] efSearch=80    recall@10=0.7644 |   4260.6 QPS (  0.235 ms/q) | compared  0.16% of db
  [glove] efSearch=96    recall@10=0.7827 |   3419.0 QPS (  0.292 ms/q) | compared  0.18% of db
  [glove] efSearch=128   recall@10=0.8090 |   2767.4 QPS (  0.361 ms/q) | compared  0.22% of db
  [glove] efSearch=192   recall@10=0.8419 |   1832.7 QPS (  0.546 ms/q) | compared  0.31% of db
  [glove] efSearch=256   recall@10=0.8634 |   1331.3 QPS (  0.751 ms/q) | compared  0.39% of db
  [glove] efSearch=384   recall@10=0.8888 |    886.8 QPS (  1.128 ms/q) | compared  0.54% of db
  [glove] efSearch=512   recall@10=0.9050 |    607.7 QPS (  1.646 ms/q) | compared  0.69% of db
  [glove] efSearch=768   recall@10=0.9253 |    360.4 QPS (  2.774 ms/q) | compared  0.98% of db
  [glove] efSearch=1024  recall@10=0.9373 |    272.6 QPS (  3.668 ms/q) | compared  1.25% of db
[glove] recall >= 0.9: efSearch=512 (recall 0.9050, 608 QPS)
[glove] recall >= 0.95: NOT reached (max recall 0.9373)
~~~~

Execution count recorded by the notebook: `3`.

## Cell 9 > markdown

### Prompt

Create notebook cell 9 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 1b: the same configuration rebuilt, to measure build noise

**Decision.** Build Try 1 a second time with identical settings.

**Reasoning.** Multi-threaded insertion makes the graph differ slightly from build to build. Before reading anything into the difference between, say, `efConstruction = 100` and `200`, we need to know how large the difference between two *identical* builds is. If Try 1 and Try 1b nearly coincide, differences between Tries are real; if they differ as much as two different settings, those settings cannot be separated by this experiment.

**Alternatives not chosen.**
- *Building with one thread for exact reproducibility:* a single-threaded build of 1M vectors takes many times longer, and real deployments build in parallel anyway.
- *Many repeats and error bars:* the right thing for a paper, but each repeat costs minutes of build time; one repeat is enough to see whether the noise is small or large.

**Hypothesis.** The two curves almost overlap. Differences of about one percent in recall at equal `efSearch` should be treated as noise.
~~~~

## Cell 10 > code

### Prompt

Implement notebook code cell 10 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 1b: identical to Try 1, built again, as a noise check for the non-deterministic parallel build.
"""
run_try_hnsw("Try 1b: M=16, efC=100 (rebuild, noise check)", M_BASE, EFC_BASE)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 1b: M=16, efC=100 (rebuild, noise check)

[sift] building HNSW: M=16, efConstruction=100, storage=flat float32 (this takes a while)
[sift] build: 34.9 s | size 656 MB = 656 bytes/vector (raw vector 512 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 62,403 > 3,903 > 210 > 16 > 1
[sift] layer 0 uses 19.6 of 32 link slots per node on average
  [sift] efSearch=10    recall@10=0.7003 |  13679.7 QPS (  0.073 ms/q) | compared  0.04% of db
  [sift] efSearch=12    recall@10=0.7373 |  11679.7 QPS (  0.086 ms/q) | compared  0.04% of db
  [sift] efSearch=16    recall@10=0.7910 |  12244.5 QPS (  0.082 ms/q) | compared  0.05% of db
  [sift] efSearch=20    recall@10=0.8289 |   8226.2 QPS (  0.122 ms/q) | compared  0.06% of db
  [sift] efSearch=24    recall@10=0.8566 |   9840.1 QPS (  0.102 ms/q) | compared  0.06% of db
  [sift] efSearch=32    recall@10=0.8946 |   7085.3 QPS (  0.141 ms/q) | compared  0.08% of db
  [sift] efSearch=40    recall@10=0.9197 |   5890.2 QPS (  0.170 ms/q) | compared  0.09% of db
  [sift] efSearch=48    recall@10=0.9354 |   5656.7 QPS (  0.177 ms/q) | compared  0.10% of db
  [sift] efSearch=64    recall@10=0.9568 |   4730.8 QPS (  0.211 ms/q) | compared  0.12% of db
  [sift] efSearch=80    recall@10=0.9692 |   3465.2 QPS (  0.289 ms/q) | compared  0.15% of db
  [sift] efSearch=96    recall@10=0.9766 |   3076.3 QPS (  0.325 ms/q) | compared  0.17% of db
  [sift] efSearch=128   recall@10=0.9854 |   2229.8 QPS (  0.448 ms/q) | compared  0.22% of db
  [sift] efSearch=192   recall@10=0.9930 |   1513.4 QPS (  0.661 ms/q) | compared  0.30% of db
[sift] recall >= 0.9: efSearch=40 (recall 0.9197, 5890 QPS)
[sift] recall >= 0.95: efSearch=64 (recall 0.9568, 4731 QPS)

[glove] building HNSW: M=16, efConstruction=100, storage=flat float32 (this takes a while)
[glove] build: 46.4 s | size 644 MB = 544 bytes/vector (raw vector 400 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 74,078 > 4,644 > 260 > 18 > 1
[glove] layer 0 uses 21.1 of 32 link slots per node on average
  [glove] efSearch=10    recall@10=0.4522 |   9915.5 QPS (  0.101 ms/q) | compared  0.04% of db
  [glove] efSearch=12    recall@10=0.4850 |   9448.0 QPS (  0.106 ms/q) | compared  0.04% of db
  [glove] efSearch=16    recall@10=0.5345 |   9248.7 QPS (  0.108 ms/q) | compared  0.05% of db
  [glove] efSearch=20    recall@10=0.5727 |   7082.8 QPS (  0.141 ms/q) | compared  0.06% of db
  [glove] efSearch=24    recall@10=0.6029 |   5962.5 QPS (  0.168 ms/q) | compared  0.07% of db
  [glove] efSearch=32    recall@10=0.6472 |   5267.7 QPS (  0.190 ms/q) | compared  0.08% of db
  [glove] efSearch=40    recall@10=0.6800 |   4966.6 QPS (  0.201 ms/q) | compared  0.09% of db
  [glove] efSearch=48    recall@10=0.7044 |   4751.0 QPS (  0.210 ms/q) | compared  0.11% of db
  [glove] efSearch=64    recall@10=0.7399 |   3695.3 QPS (  0.271 ms/q) | compared  0.13% of db
  [glove] efSearch=80    recall@10=0.7644 |   3096.1 QPS (  0.323 ms/q) | compared  0.16% of db
  [glove] efSearch=96    recall@10=0.7827 |   2438.0 QPS (  0.410 ms/q) | compared  0.18% of db
  [glove] efSearch=128   recall@10=0.8090 |   1682.0 QPS (  0.595 ms/q) | compared  0.22% of db
  [glove] efSearch=192   recall@10=0.8419 |   1242.2 QPS (  0.805 ms/q) | compared  0.31% of db
  [glove] efSearch=256   recall@10=0.8634 |    982.0 QPS (  1.018 ms/q) | compared  0.39% of db
  [glove] efSearch=384   recall@10=0.8888 |    700.7 QPS (  1.427 ms/q) | compared  0.54% of db
  [glove] efSearch=512   recall@10=0.9050 |    531.1 QPS (  1.883 ms/q) | compared  0.69% of db
  [glove] efSearch=768   recall@10=0.9253 |    315.5 QPS (  3.169 ms/q) | compared  0.98% of db
  [glove] efSearch=1024  recall@10=0.9373 |    222.5 QPS (  4.495 ms/q) | compared  1.25% of db
[glove] recall >= 0.9: efSearch=512 (recall 0.9050, 531 QPS)
[glove] recall >= 0.95: NOT reached (max recall 0.9373)
~~~~

Execution count recorded by the notebook: `4`.

## Cell 11 > markdown

### Prompt

Create notebook cell 11 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 2: sparser graph, `M = 8`

**Decision.** Halve the links per node (8, and 16 on layer 0); everything else as the base.

**Reasoning.** This isolates the effect of connectivity. Fewer links mean less memory, a faster build and cheaper steps during the search (fewer neighbours to check per expanded node), but also fewer routes to reach any given vector.

**Alternatives not chosen.** `M = 4` (see Try 1) and `M = 12` (too close to 16 to rise above build noise measured in Try 1b).

**Hypothesis.** Lower recall at the same `efSearch`, and possibly a **recall ceiling** below 0.99 that raising `efSearch` cannot fix, because some true neighbours have no good path leading to them. Expect the ceiling to be lower on GloVe than on SIFT. QPS at equal `efSearch` should be higher than the base, so whether `M = 8` wins at a given recall target depends on how much extra `efSearch` it needs.
~~~~

## Cell 12 > code

### Prompt

Implement notebook code cell 12 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 2: M = 8, efConstruction = 100.
"""
run_try_hnsw("Try 2: M=8 (sparser)", M_SMALL, EFC_BASE)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 2: M=8 (sparser)

[sift] building HNSW: M=8, efConstruction=100, storage=flat float32 (this takes a while)
[sift] build: 27.3 s | size 593 MB = 593 bytes/vector (raw vector 512 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 124,841 > 15,653 > 1,964 > 210 > 27 > 5 > 1
[sift] layer 0 uses 12.5 of 16 link slots per node on average
  [sift] efSearch=10    recall@10=0.5699 |  20115.5 QPS (  0.050 ms/q) | compared  0.02% of db
  [sift] efSearch=12    recall@10=0.6101 |  16623.4 QPS (  0.060 ms/q) | compared  0.03% of db
  [sift] efSearch=16    recall@10=0.6709 |  15456.0 QPS (  0.065 ms/q) | compared  0.03% of db
  [sift] efSearch=20    recall@10=0.7147 |  13916.2 QPS (  0.072 ms/q) | compared  0.04% of db
  [sift] efSearch=24    recall@10=0.7470 |  11639.7 QPS (  0.086 ms/q) | compared  0.04% of db
  [sift] efSearch=32    recall@10=0.7949 |  10592.7 QPS (  0.094 ms/q) | compared  0.05% of db
  [sift] efSearch=40    recall@10=0.8296 |   7176.3 QPS (  0.139 ms/q) | compared  0.05% of db
  [sift] efSearch=48    recall@10=0.8545 |  10317.2 QPS (  0.097 ms/q) | compared  0.06% of db
  [sift] efSearch=64    recall@10=0.8875 |   7190.3 QPS (  0.139 ms/q) | compared  0.08% of db
  [sift] efSearch=80    recall@10=0.9101 |   5428.4 QPS (  0.184 ms/q) | compared  0.09% of db
  [sift] efSearch=96    recall@10=0.9254 |   4224.5 QPS (  0.237 ms/q) | compared  0.11% of db
  [sift] efSearch=128   recall@10=0.9451 |   3500.6 QPS (  0.286 ms/q) | compared  0.13% of db
  [sift] efSearch=192   recall@10=0.9650 |   2455.0 QPS (  0.407 ms/q) | compared  0.19% of db
  [sift] efSearch=256   recall@10=0.9748 |   1701.2 QPS (  0.588 ms/q) | compared  0.24% of db
  [sift] efSearch=384   recall@10=0.9844 |   1082.8 QPS (  0.924 ms/q) | compared  0.33% of db
  [sift] efSearch=512   recall@10=0.9887 |    763.2 QPS (  1.310 ms/q) | compared  0.42% of db
  [sift] efSearch=768   recall@10=0.9926 |    465.8 QPS (  2.147 ms/q) | compared  0.59% of db
[sift] recall >= 0.9: efSearch=80 (recall 0.9101, 5428 QPS)
[sift] recall >= 0.95: efSearch=192 (recall 0.9650, 2455 QPS)

[glove] building HNSW: M=8, efConstruction=100, storage=flat float32 (this takes a while)
[glove] build: 25.4 s | size 569 MB = 481 bytes/vector (raw vector 400 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 148,058 > 18,577 > 2,330 > 260 > 36 > 5 > 1
[glove] layer 0 uses 11.7 of 16 link slots per node on average
  [glove] efSearch=10    recall@10=0.2859 |  27760.2 QPS (  0.036 ms/q) | compared  0.02% of db
  [glove] efSearch=12    recall@10=0.3129 |  24006.7 QPS (  0.042 ms/q) | compared  0.02% of db
  [glove] efSearch=16    recall@10=0.3591 |  19933.3 QPS (  0.050 ms/q) | compared  0.03% of db
  [glove] efSearch=20    recall@10=0.3947 |  15271.0 QPS (  0.065 ms/q) | compared  0.03% of db
  [glove] efSearch=24    recall@10=0.4238 |  11198.3 QPS (  0.089 ms/q) | compared  0.04% of db
  [glove] efSearch=32    recall@10=0.4723 |  12099.4 QPS (  0.083 ms/q) | compared  0.05% of db
  [glove] efSearch=40    recall@10=0.5078 |   9652.9 QPS (  0.104 ms/q) | compared  0.05% of db
  [glove] efSearch=48    recall@10=0.5354 |   6532.9 QPS (  0.153 ms/q) | compared  0.06% of db
  [glove] efSearch=64    recall@10=0.5793 |   7843.1 QPS (  0.128 ms/q) | compared  0.08% of db
  [glove] efSearch=80    recall@10=0.6094 |   5255.6 QPS (  0.190 ms/q) | compared  0.09% of db
  [glove] efSearch=96    recall@10=0.6355 |   4085.2 QPS (  0.245 ms/q) | compared  0.10% of db
  [glove] efSearch=128   recall@10=0.6703 |   3833.4 QPS (  0.261 ms/q) | compared  0.13% of db
  [glove] efSearch=192   recall@10=0.7153 |   2411.1 QPS (  0.415 ms/q) | compared  0.17% of db
  [glove] efSearch=256   recall@10=0.7423 |   1725.9 QPS (  0.579 ms/q) | compared  0.22% of db
  [glove] efSearch=384   recall@10=0.7761 |   1229.1 QPS (  0.814 ms/q) | compared  0.30% of db
  [glove] efSearch=512   recall@10=0.7973 |    793.2 QPS (  1.261 ms/q) | compared  0.39% of db
  [glove] efSearch=768   recall@10=0.8217 |    465.4 QPS (  2.149 ms/q) | compared  0.54% of db
  [glove] efSearch=1024  recall@10=0.8380 |    320.5 QPS (  3.120 ms/q) | compared  0.69% of db
[glove] recall >= 0.9: NOT reached (max recall 0.8380)
[glove] recall >= 0.95: NOT reached (max recall 0.8380)
~~~~

Execution count recorded by the notebook: `5`.

## Cell 13 > markdown

### Prompt

Create notebook cell 13 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 3: denser graph, `M = 32`

**Decision.** Double the links per node (32, and 64 on layer 0); everything else as the base.

**Reasoning.** The opposite end of the connectivity axis. More links give the search more routes and make the graph more robust, so recall should be reached with a smaller `efSearch`. The price is about 250 extra bytes of links per vector, a longer build, and more distance computations per expanded node.

**Alternatives not chosen.** `M = 64`: link memory (about 520 bytes per vector) would exceed the size of the raw vector itself, the build becomes very slow, and the gain over `M = 32` is typically small. It would be worth testing only if `M = 32` still showed a ceiling.

**Hypothesis.** The highest recall at small `efSearch` of all Tries, but not necessarily the highest QPS at a given recall: each step now costs more, so there is a break-even between "fewer steps needed" and "more expensive steps". The interesting output is where that break-even falls for SIFT versus GloVe.
~~~~

## Cell 14 > code

### Prompt

Implement notebook code cell 14 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 3: M = 32, efConstruction = 100.
"""
run_try_hnsw("Try 3: M=32 (denser)", M_LARGE, EFC_BASE)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 3: M=32 (denser)

[sift] building HNSW: M=32, efConstruction=100, storage=flat float32 (this takes a while)
[sift] build: 38.3 s | size 784 MB = 784 bytes/vector (raw vector 512 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 31,254 > 978 > 27 > 1
[sift] layer 0 uses 24.7 of 64 link slots per node on average
  [sift] efSearch=10    recall@10=0.7470 |  16681.0 QPS (  0.060 ms/q) | compared  0.05% of db
  [sift] efSearch=12    recall@10=0.7809 |  13003.6 QPS (  0.077 ms/q) | compared  0.06% of db
  [sift] efSearch=16    recall@10=0.8306 |   7442.6 QPS (  0.134 ms/q) | compared  0.07% of db
  [sift] efSearch=20    recall@10=0.8647 |   9731.9 QPS (  0.103 ms/q) | compared  0.07% of db
  [sift] efSearch=24    recall@10=0.8891 |   7352.1 QPS (  0.136 ms/q) | compared  0.08% of db
  [sift] efSearch=32    recall@10=0.9220 |   7481.2 QPS (  0.134 ms/q) | compared  0.10% of db
  [sift] efSearch=40    recall@10=0.9426 |   7191.9 QPS (  0.139 ms/q) | compared  0.12% of db
  [sift] efSearch=48    recall@10=0.9553 |   5727.4 QPS (  0.175 ms/q) | compared  0.13% of db
  [sift] efSearch=64    recall@10=0.9709 |   4360.9 QPS (  0.229 ms/q) | compared  0.16% of db
  [sift] efSearch=80    recall@10=0.9802 |   3729.5 QPS (  0.268 ms/q) | compared  0.19% of db
  [sift] efSearch=96    recall@10=0.9855 |   3255.0 QPS (  0.307 ms/q) | compared  0.22% of db
  [sift] efSearch=128   recall@10=0.9916 |   2123.6 QPS (  0.471 ms/q) | compared  0.28% of db
[sift] recall >= 0.9: efSearch=32 (recall 0.9220, 7481 QPS)
[sift] recall >= 0.95: efSearch=48 (recall 0.9553, 5727 QPS)

[glove] building HNSW: M=32, efConstruction=100, storage=flat float32 (this takes a while)
[glove] build: 53.5 s | size 796 MB = 672 bytes/vector (raw vector 400 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 37,170 > 1,172 > 36 > 1
[glove] layer 0 uses 30.0 of 64 link slots per node on average
  [glove] efSearch=10    recall@10=0.5430 |  12209.2 QPS (  0.082 ms/q) | compared  0.06% of db
  [glove] efSearch=12    recall@10=0.5749 |  13355.5 QPS (  0.075 ms/q) | compared  0.07% of db
  [glove] efSearch=16    recall@10=0.6228 |  10152.6 QPS (  0.098 ms/q) | compared  0.08% of db
  [glove] efSearch=20    recall@10=0.6581 |   8724.5 QPS (  0.115 ms/q) | compared  0.09% of db
  [glove] efSearch=24    recall@10=0.6849 |   8606.3 QPS (  0.116 ms/q) | compared  0.10% of db
  [glove] efSearch=32    recall@10=0.7242 |   6611.9 QPS (  0.151 ms/q) | compared  0.12% of db
  [glove] efSearch=40    recall@10=0.7533 |   5206.0 QPS (  0.192 ms/q) | compared  0.14% of db
  [glove] efSearch=48    recall@10=0.7729 |   4754.5 QPS (  0.210 ms/q) | compared  0.16% of db
  [glove] efSearch=64    recall@10=0.8019 |   3693.4 QPS (  0.271 ms/q) | compared  0.19% of db
  [glove] efSearch=80    recall@10=0.8239 |   3245.8 QPS (  0.308 ms/q) | compared  0.23% of db
  [glove] efSearch=96    recall@10=0.8397 |   2662.6 QPS (  0.376 ms/q) | compared  0.26% of db
  [glove] efSearch=128   recall@10=0.8628 |   1933.4 QPS (  0.517 ms/q) | compared  0.33% of db
  [glove] efSearch=192   recall@10=0.8922 |   1318.0 QPS (  0.759 ms/q) | compared  0.46% of db
  [glove] efSearch=256   recall@10=0.9097 |   1074.7 QPS (  0.930 ms/q) | compared  0.58% of db
  [glove] efSearch=384   recall@10=0.9323 |    710.6 QPS (  1.407 ms/q) | compared  0.81% of db
  [glove] efSearch=512   recall@10=0.9453 |    540.2 QPS (  1.851 ms/q) | compared  1.04% of db
  [glove] efSearch=768   recall@10=0.9610 |    329.8 QPS (  3.032 ms/q) | compared  1.46% of db
  [glove] efSearch=1024  recall@10=0.9696 |    228.3 QPS (  4.380 ms/q) | compared  1.88% of db
[glove] recall >= 0.9: efSearch=256 (recall 0.9097, 1075 QPS)
[glove] recall >= 0.95: efSearch=768 (recall 0.9610, 330 QPS)
~~~~

Execution count recorded by the notebook: `6`.

## Cell 15 > markdown

### Prompt

Create notebook cell 15 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 4: cheaper construction, `efConstruction = 40`

**Decision.** Base `M = 16`, but each insertion considers only 40 candidates instead of 100.

**Reasoning.** `efConstruction` controls graph *quality* without changing graph *size*: the index file is the same size, only the choice of links is poorer. This makes it the cleanest way to see what build effort buys. It is also the tweak a practitioner reaches for first when the build is too slow.

**Alternatives not chosen.** `efConstruction = 16` (equal to `M`): the construction search could barely compare candidates and the graph would be close to random, which is not informative. `efConstruction = 20` and `30` are too close to 40 to separate from noise.

**Hypothesis.** The build is noticeably faster (roughly proportional to the candidate list size), index size is unchanged, and recall at the same `efSearch` is lower, so more `efSearch` (and hence a lower QPS) is needed to reach 0.90 and 0.95.
~~~~

## Cell 16 > code

### Prompt

Implement notebook code cell 16 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 4: M = 16, efConstruction = 40.
"""
run_try_hnsw("Try 4: efC=40 (cheaper build)", M_BASE, EFC_LOW)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 4: efC=40 (cheaper build)

[sift] building HNSW: M=16, efConstruction=40, storage=flat float32 (this takes a while)
[sift] build: 12.6 s | size 656 MB = 656 bytes/vector (raw vector 512 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 62,403 > 3,903 > 210 > 16 > 1
[sift] layer 0 uses 14.6 of 32 link slots per node on average
  [sift] efSearch=10    recall@10=0.6279 |  22109.0 QPS (  0.045 ms/q) | compared  0.03% of db
  [sift] efSearch=12    recall@10=0.6647 |  25274.9 QPS (  0.040 ms/q) | compared  0.04% of db
  [sift] efSearch=16    recall@10=0.7215 |  20713.4 QPS (  0.048 ms/q) | compared  0.04% of db
  [sift] efSearch=20    recall@10=0.7621 |  17543.2 QPS (  0.057 ms/q) | compared  0.05% of db
  [sift] efSearch=24    recall@10=0.7921 |  15708.5 QPS (  0.064 ms/q) | compared  0.05% of db
  [sift] efSearch=32    recall@10=0.8344 |  12581.6 QPS (  0.079 ms/q) | compared  0.06% of db
  [sift] efSearch=40    recall@10=0.8655 |  11434.0 QPS (  0.087 ms/q) | compared  0.07% of db
  [sift] efSearch=48    recall@10=0.8872 |  10070.4 QPS (  0.099 ms/q) | compared  0.08% of db
  [sift] efSearch=64    recall@10=0.9163 |   7579.1 QPS (  0.132 ms/q) | compared  0.10% of db
  [sift] efSearch=80    recall@10=0.9349 |   6403.9 QPS (  0.156 ms/q) | compared  0.12% of db
  [sift] efSearch=96    recall@10=0.9476 |   5419.2 QPS (  0.185 ms/q) | compared  0.14% of db
  [sift] efSearch=128   recall@10=0.9638 |   4155.9 QPS (  0.241 ms/q) | compared  0.17% of db
  [sift] efSearch=192   recall@10=0.9791 |   2814.2 QPS (  0.355 ms/q) | compared  0.24% of db
  [sift] efSearch=256   recall@10=0.9862 |   2107.1 QPS (  0.475 ms/q) | compared  0.31% of db
  [sift] efSearch=384   recall@10=0.9931 |   1320.5 QPS (  0.757 ms/q) | compared  0.42% of db
[sift] recall >= 0.9: efSearch=64 (recall 0.9163, 7579 QPS)
[sift] recall >= 0.95: efSearch=128 (recall 0.9638, 4156 QPS)

[glove] building HNSW: M=16, efConstruction=40, storage=flat float32 (this takes a while)
[glove] build: 15.5 s | size 644 MB = 544 bytes/vector (raw vector 400 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 74,078 > 4,644 > 260 > 18 > 1
[glove] layer 0 uses 13.7 of 32 link slots per node on average
  [glove] efSearch=10    recall@10=0.3769 |  26421.2 QPS (  0.038 ms/q) | compared  0.03% of db
  [glove] efSearch=12    recall@10=0.4056 |  26095.2 QPS (  0.038 ms/q) | compared  0.04% of db
  [glove] efSearch=16    recall@10=0.4490 |  20341.8 QPS (  0.049 ms/q) | compared  0.04% of db
  [glove] efSearch=20    recall@10=0.4849 |  14499.3 QPS (  0.069 ms/q) | compared  0.05% of db
  [glove] efSearch=24    recall@10=0.5117 |  15772.8 QPS (  0.063 ms/q) | compared  0.06% of db
  [glove] efSearch=32    recall@10=0.5544 |  13201.5 QPS (  0.076 ms/q) | compared  0.07% of db
  [glove] efSearch=40    recall@10=0.5862 |  10890.8 QPS (  0.092 ms/q) | compared  0.08% of db
  [glove] efSearch=48    recall@10=0.6110 |   9599.3 QPS (  0.104 ms/q) | compared  0.09% of db
  [glove] efSearch=64    recall@10=0.6473 |   7240.8 QPS (  0.138 ms/q) | compared  0.11% of db
  [glove] efSearch=80    recall@10=0.6740 |   6152.7 QPS (  0.163 ms/q) | compared  0.13% of db
  [glove] efSearch=96    recall@10=0.6953 |   5169.9 QPS (  0.193 ms/q) | compared  0.15% of db
  [glove] efSearch=128   recall@10=0.7273 |   3971.2 QPS (  0.252 ms/q) | compared  0.18% of db
  [glove] efSearch=192   recall@10=0.7661 |   2764.9 QPS (  0.362 ms/q) | compared  0.25% of db
  [glove] efSearch=256   recall@10=0.7911 |   2004.3 QPS (  0.499 ms/q) | compared  0.32% of db
  [glove] efSearch=384   recall@10=0.8212 |   1319.1 QPS (  0.758 ms/q) | compared  0.44% of db
  [glove] efSearch=512   recall@10=0.8399 |    950.5 QPS (  1.052 ms/q) | compared  0.55% of db
  [glove] efSearch=768   recall@10=0.8641 |    583.8 QPS (  1.713 ms/q) | compared  0.77% of db
  [glove] efSearch=1024  recall@10=0.8802 |    394.6 QPS (  2.534 ms/q) | compared  0.99% of db
[glove] recall >= 0.9: NOT reached (max recall 0.8802)
[glove] recall >= 0.95: NOT reached (max recall 0.8802)
~~~~

Execution count recorded by the notebook: `7`.

## Cell 17 > markdown

### Prompt

Create notebook cell 17 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 5: more careful construction, `efConstruction = 200`

**Decision.** Base `M = 16`, with 200 candidates per insertion.

**Reasoning.** The other side of Try 4. Doubling the construction effort should give better links; this tests whether the base is already near the useful limit or still leaves recall on the table.

**Alternatives not chosen.** `efConstruction = 400` or higher: build time grows roughly in proportion and each extra doubling usually buys less. If Try 5 already shows only a small gain over Try 1, higher values would only confirm diminishing returns at a cost of tens of minutes per dataset.

**Hypothesis.** Longer build, identical size, recall at the same `efSearch` somewhat higher than the base, so a smaller `efSearch` reaches 0.95 and QPS at that recall improves. The gain from 100 to 200 should be smaller than from 40 to 100 (diminishing returns); compare with the noise seen in Try 1b to decide whether the gain is real.
~~~~

## Cell 18 > code

### Prompt

Implement notebook code cell 18 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 5: M = 16, efConstruction = 200.
"""
run_try_hnsw("Try 5: efC=200 (careful build)", M_BASE, EFC_HIGH)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 5: efC=200 (careful build)

[sift] building HNSW: M=16, efConstruction=200, storage=flat float32 (this takes a while)
[sift] build: 55.8 s | size 656 MB = 656 bytes/vector (raw vector 512 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 62,403 > 3,903 > 210 > 16 > 1
[sift] layer 0 uses 22.1 of 32 link slots per node on average
  [sift] efSearch=10    recall@10=0.7165 |  23625.1 QPS (  0.042 ms/q) | compared  0.04% of db
  [sift] efSearch=12    recall@10=0.7542 |  21820.5 QPS (  0.046 ms/q) | compared  0.04% of db
  [sift] efSearch=16    recall@10=0.8084 |  18450.9 QPS (  0.054 ms/q) | compared  0.05% of db
  [sift] efSearch=20    recall@10=0.8466 |  15677.3 QPS (  0.064 ms/q) | compared  0.06% of db
  [sift] efSearch=24    recall@10=0.8744 |  13984.5 QPS (  0.072 ms/q) | compared  0.07% of db
  [sift] efSearch=32    recall@10=0.9117 |  11542.8 QPS (  0.087 ms/q) | compared  0.08% of db
  [sift] efSearch=40    recall@10=0.9345 |   9560.2 QPS (  0.105 ms/q) | compared  0.09% of db
  [sift] efSearch=48    recall@10=0.9493 |   7898.0 QPS (  0.127 ms/q) | compared  0.11% of db
  [sift] efSearch=64    recall@10=0.9682 |   6325.2 QPS (  0.158 ms/q) | compared  0.13% of db
  [sift] efSearch=80    recall@10=0.9782 |   5392.1 QPS (  0.185 ms/q) | compared  0.16% of db
  [sift] efSearch=96    recall@10=0.9842 |   4563.6 QPS (  0.219 ms/q) | compared  0.19% of db
  [sift] efSearch=128   recall@10=0.9911 |   3490.8 QPS (  0.286 ms/q) | compared  0.23% of db
[sift] recall >= 0.9: efSearch=32 (recall 0.9117, 11543 QPS)
[sift] recall >= 0.95: efSearch=64 (recall 0.9682, 6325 QPS)

[glove] building HNSW: M=16, efConstruction=200, storage=flat float32 (this takes a while)
[glove] build: 71.7 s | size 644 MB = 544 bytes/vector (raw vector 400 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 74,078 > 4,644 > 260 > 18 > 1
[glove] layer 0 uses 25.2 of 32 link slots per node on average
  [glove] efSearch=10    recall@10=0.4760 |  21748.1 QPS (  0.046 ms/q) | compared  0.04% of db
  [glove] efSearch=12    recall@10=0.5092 |  20432.8 QPS (  0.049 ms/q) | compared  0.05% of db
  [glove] efSearch=16    recall@10=0.5638 |  15476.4 QPS (  0.065 ms/q) | compared  0.06% of db
  [glove] efSearch=20    recall@10=0.6013 |  13592.4 QPS (  0.074 ms/q) | compared  0.06% of db
  [glove] efSearch=24    recall@10=0.6316 |  11216.1 QPS (  0.089 ms/q) | compared  0.07% of db
  [glove] efSearch=32    recall@10=0.6751 |  10149.3 QPS (  0.099 ms/q) | compared  0.09% of db
  [glove] efSearch=40    recall@10=0.7063 |   8860.6 QPS (  0.113 ms/q) | compared  0.10% of db
  [glove] efSearch=48    recall@10=0.7315 |   7723.6 QPS (  0.129 ms/q) | compared  0.12% of db
  [glove] efSearch=64    recall@10=0.7639 |   6303.2 QPS (  0.159 ms/q) | compared  0.14% of db
  [glove] efSearch=80    recall@10=0.7883 |   5117.4 QPS (  0.195 ms/q) | compared  0.17% of db
  [glove] efSearch=96    recall@10=0.8060 |   4336.7 QPS (  0.231 ms/q) | compared  0.19% of db
  [glove] efSearch=128   recall@10=0.8321 |   3414.7 QPS (  0.293 ms/q) | compared  0.24% of db
  [glove] efSearch=192   recall@10=0.8625 |   2331.0 QPS (  0.429 ms/q) | compared  0.33% of db
  [glove] efSearch=256   recall@10=0.8822 |   1739.2 QPS (  0.575 ms/q) | compared  0.42% of db
  [glove] efSearch=384   recall@10=0.9066 |   1140.3 QPS (  0.877 ms/q) | compared  0.60% of db
  [glove] efSearch=512   recall@10=0.9214 |    813.3 QPS (  1.230 ms/q) | compared  0.77% of db
  [glove] efSearch=768   recall@10=0.9398 |    502.8 QPS (  1.989 ms/q) | compared  1.09% of db
  [glove] efSearch=1024  recall@10=0.9514 |    347.0 QPS (  2.882 ms/q) | compared  1.40% of db
[glove] recall >= 0.9: efSearch=384 (recall 0.9066, 1140 QPS)
[glove] recall >= 0.95: efSearch=1024 (recall 0.9514, 347 QPS)
~~~~

Execution count recorded by the notebook: `8`.

## Cell 19 > markdown

### Prompt

Create notebook cell 19 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 6 (diagnostic): same graph, 8-bit scalar-quantised storage

**Decision.** Base `M = 16`, `efConstruction = 100`, but the vectors are stored as 8-bit codes (`IndexHNSWSQ`, one byte per dimension) instead of float32.

**Reasoning.** HNSW has two memory costs: the vectors and the links. This Try shrinks the first. It shows how much of the index size is raw vectors, and what compressing them does to recall, much as IVF-PQ compresses the vectors of IVF. Scalar quantisation is the gentlest compression (4x smaller vectors, small error), so it is the natural first step.

**Alternatives not chosen.**
- *`IndexHNSWPQ` (product quantisation of the vectors):* much smaller, but the graph is then built on heavily approximate distances and quality drops sharply; it needs its own tuning and belongs to the improvement part of the assignment.
- *Dropping the full vectors and re-ranking:* needs a second stage and the vectors again.

Because this index stores approximate vectors, it is **not** the HNSW index the assignment asks for. It is run as a diagnostic and **kept out of the main report curve** and tables (a second plot shows it for comparison).

**Hypothesis.** Index size drops by roughly `3d` bytes per vector (the graph does not shrink), build and search are of similar speed, and recall has a slightly lower ceiling than the flat version because distances are now approximate.
~~~~

## Cell 20 > code

### Prompt

Implement notebook code cell 20 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
Try 6 (diagnostic): M = 16, efConstruction = 100, vectors stored as 8-bit scalar-quantised codes. Tagged is_variant and kept out of the main report curve.
"""
run_try_hnsw("Try 6: 8-bit scalar-quantised vectors (diagnostic)", M_BASE, EFC_BASE, sq8=True)
~~~~

### Captured output

~~~~text
====================================================================================================
Try 6: 8-bit scalar-quantised vectors (diagnostic)

[sift] building HNSW: M=16, efConstruction=100, storage=8-bit scalar quantised (this takes a while)
[sift] build: 56.6 s | size 272 MB = 272 bytes/vector (raw vector 128 B, the rest is the graph)
[sift] nodes per layer (layer 0 first): 1,000,000 > 62,403 > 3,903 > 210 > 16 > 1
[sift] layer 0 uses 19.4 of 32 link slots per node on average
  [sift] efSearch=10    recall@10=0.6973 |  10546.9 QPS (  0.095 ms/q) | compared  0.04% of db
  [sift] efSearch=12    recall@10=0.7353 |   9686.0 QPS (  0.103 ms/q) | compared  0.04% of db
  [sift] efSearch=16    recall@10=0.7890 |   8074.5 QPS (  0.124 ms/q) | compared  0.05% of db
  [sift] efSearch=20    recall@10=0.8256 |   7125.8 QPS (  0.140 ms/q) | compared  0.06% of db
  [sift] efSearch=24    recall@10=0.8518 |   5887.5 QPS (  0.170 ms/q) | compared  0.06% of db
  [sift] efSearch=32    recall@10=0.8885 |   5388.0 QPS (  0.186 ms/q) | compared  0.07% of db
  [sift] efSearch=40    recall@10=0.9125 |   4384.4 QPS (  0.228 ms/q) | compared  0.09% of db
  [sift] efSearch=48    recall@10=0.9283 |   3767.7 QPS (  0.265 ms/q) | compared  0.10% of db
  [sift] efSearch=64    recall@10=0.9482 |   3125.5 QPS (  0.320 ms/q) | compared  0.12% of db
  [sift] efSearch=80    recall@10=0.9598 |   2590.5 QPS (  0.386 ms/q) | compared  0.15% of db
  [sift] efSearch=96    recall@10=0.9669 |   2150.5 QPS (  0.465 ms/q) | compared  0.17% of db
  [sift] efSearch=128   recall@10=0.9750 |   1758.1 QPS (  0.569 ms/q) | compared  0.21% of db
  [sift] efSearch=192   recall@10=0.9819 |   1241.3 QPS (  0.806 ms/q) | compared  0.30% of db
  [sift] efSearch=256   recall@10=0.9847 |    958.6 QPS (  1.043 ms/q) | compared  0.38% of db
  [sift] efSearch=384   recall@10=0.9867 |    660.9 QPS (  1.513 ms/q) | compared  0.52% of db
  [sift] efSearch=512   recall@10=0.9874 |    499.6 QPS (  2.002 ms/q) | compared  0.66% of db
  [sift] efSearch=768   recall@10=0.9877 |    332.2 QPS (  3.011 ms/q) | compared  0.90% of db
  [sift] recall has plateaued at ~0.988; stopping the sweep
[sift] recall >= 0.9: efSearch=40 (recall 0.9125, 4384 QPS)
[sift] recall >= 0.95: efSearch=80 (recall 0.9598, 2591 QPS)

[glove] building HNSW: M=16, efConstruction=100, storage=8-bit scalar quantised (this takes a while)
[glove] build: 74.9 s | size 289 MB = 244 bytes/vector (raw vector 100 B, the rest is the graph)
[glove] nodes per layer (layer 0 first): 1,183,514 > 74,078 > 4,644 > 260 > 18 > 1
[glove] layer 0 uses 21.1 of 32 link slots per node on average
  [glove] efSearch=10    recall@10=0.4502 |   9697.0 QPS (  0.103 ms/q) | compared  0.04% of db
  [glove] efSearch=12    recall@10=0.4813 |   9232.3 QPS (  0.108 ms/q) | compared  0.04% of db
  [glove] efSearch=16    recall@10=0.5321 |   7499.6 QPS (  0.133 ms/q) | compared  0.05% of db
  [glove] efSearch=20    recall@10=0.5693 |   6352.1 QPS (  0.157 ms/q) | compared  0.06% of db
  [glove] efSearch=24    recall@10=0.5972 |   5965.7 QPS (  0.168 ms/q) | compared  0.07% of db
  [glove] efSearch=32    recall@10=0.6409 |   4782.0 QPS (  0.209 ms/q) | compared  0.08% of db
  [glove] efSearch=40    recall@10=0.6725 |   4124.9 QPS (  0.242 ms/q) | compared  0.09% of db
  [glove] efSearch=48    recall@10=0.6964 |   3552.1 QPS (  0.282 ms/q) | compared  0.11% of db
  [glove] efSearch=64    recall@10=0.7298 |   2901.5 QPS (  0.345 ms/q) | compared  0.13% of db
  [glove] efSearch=80    recall@10=0.7551 |   2310.7 QPS (  0.433 ms/q) | compared  0.16% of db
  [glove] efSearch=96    recall@10=0.7728 |   2042.1 QPS (  0.490 ms/q) | compared  0.18% of db
  [glove] efSearch=128   recall@10=0.7992 |   1644.4 QPS (  0.608 ms/q) | compared  0.22% of db
  [glove] efSearch=192   recall@10=0.8314 |   1145.0 QPS (  0.873 ms/q) | compared  0.31% of db
  [glove] efSearch=256   recall@10=0.8504 |    888.0 QPS (  1.126 ms/q) | compared  0.39% of db
  [glove] efSearch=384   recall@10=0.8754 |    607.4 QPS (  1.646 ms/q) | compared  0.54% of db
  [glove] efSearch=512   recall@10=0.8906 |    463.6 QPS (  2.157 ms/q) | compared  0.69% of db
  [glove] efSearch=768   recall@10=0.9093 |    301.8 QPS (  3.313 ms/q) | compared  0.98% of db
  [glove] efSearch=1024  recall@10=0.9209 |    218.8 QPS (  4.570 ms/q) | compared  1.25% of db
[glove] recall >= 0.9: efSearch=768 (recall 0.9093, 302 QPS)
[glove] recall >= 0.95: NOT reached (max recall 0.9209)
~~~~

Execution count recorded by the notebook: `9`.

## Cell 21 > markdown

### Prompt

Create notebook cell 21 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Comparing the Tries: which component bought what>

For each dataset the table lists, per Try, the build time, the index size split into raw vectors and graph, the **recall ceiling** (the highest recall the sweep reached), and, for recall 0.90 and 0.95, the `efSearch` needed, the **number of distances computed per query** (as a percentage of the database) and the QPS.

How to read it:
- *Same `M`, different `efConstruction`* (Tries 1, 1b, 4, 5): build effort versus graph quality. Size must be unchanged.
- *Different `M`* (Tries 1, 2, 3): connectivity versus memory, build time and cost per step.
- *Distance computations per query* show the cost directly. Compare it with IVF-Flat, which had to compare several percent of the database for the same recall.
- *Hops per query* is the number of graph nodes expanded; distances per query divided by hops is roughly the number of links followed per step.
- The graph overhead column is the index size per vector minus the raw vector bytes: it is the price of the links.
~~~~

## Cell 22 > code

### Prompt

Implement notebook code cell 22 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell compares all Tries: build time, size split into vectors and graph, recall ceiling, efSearch / distance computations / QPS at recall 0.90 and 0.95, and the structure of each graph (nodes per layer).
"""
RES = pd.concat(RESULTS, ignore_index=True)
BLD = pd.DataFrame(BUILDS)

for ds in DATASETS:
    ev = EVAL[ds]
    rows = []
    for tr, g in RES[RES["dataset"] == ds].groupby("try", sort=False):
        b = BLD[(BLD["dataset"] == ds) & (BLD["try"] == tr)].iloc[0]
        raw = ev.d if b["is_variant"] else 4 * ev.d
        row = {"try": tr, "build s": round(b["build_s"], 1), "bytes/vec": round(b["bytes_per_vector"]),
               "graph B/vec": round(b["bytes_per_vector"] - raw), "ceiling": round(g["recall"].max(), 4)}
        for t in (0.90, 0.95):
            r = best_at_recall(g, t)
            row[f"ef@{t}"] = int(r["param"]) if r is not None else None
            row[f"dists%@{t}"] = round(r["dist_comps_per_query"] / ev.n * 100, 3) if r is not None else None
            row[f"QPS@{t}"] = round(r["qps"], 1) if r is not None else None
        rows.append(row)
    print(f"\n===== {ds}: Tries side by side (exact baseline: {BASELINE[ds]['qps']:.1f} QPS, {4 * ev.d} bytes/vector) =====")
    print(int_cols(pd.DataFrame(rows)).to_string(index=False))

    print(f"\n----- {ds}: graph structure and work per query at recall 0.95 (or at the ceiling) -----")
    rows = []
    for tr, g in RES[RES["dataset"] == ds].groupby("try", sort=False):
        b = BLD[(BLD["dataset"] == ds) & (BLD["try"] == tr)].iloc[0]
        r = best_at_recall(g, 0.95)
        r = g.loc[g["recall"].idxmax()] if r is None else r
        rows.append({"try": tr, "links used L0": round(b["mean_links_layer0"], 1), "slots L0": 2 * int(b["M"]),
                     "at recall": round(r["recall"], 4), "efSearch": int(r["param"]),
                     "dists/query": int(r["dist_comps_per_query"]), "hops/query": int(r["hops_per_query"])})
    print(int_cols(pd.DataFrame(rows)).to_string(index=False))
    print("nodes per layer (layer 0 first), base:", BLD[(BLD["dataset"] == ds)].iloc[0]["layers"])
~~~~

### Captured output

~~~~text

===== sift: Tries side by side (exact baseline: 29.7 QPS, 512 bytes/vector) =====
                                               try  build s  bytes/vec  graph B/vec  ceiling  ef@0.9  dists%@0.9  QPS@0.9  ef@0.95  dists%@0.95  QPS@0.95
                       Try 1: M=16, efC=100 (base)     32.5        656          144   0.9930      40       0.088   5499.7       64        0.124    4063.6
      Try 1b: M=16, efC=100 (rebuild, noise check)     34.9        656          144   0.9930      40       0.088   5890.2       64        0.124    4730.8
                              Try 2: M=8 (sparser)     27.3        593           81   0.9926      80       0.092   5428.4      192        0.186    2455.0
                              Try 3: M=32 (denser)     38.3        784          272   0.9916      32       0.099   7481.2       48        0.131    5727.4
                     Try 4: efC=40 (cheaper build)     12.6        656          144   0.9931      64       0.102   7579.1      128        0.175    4155.9
                    Try 5: efC=200 (careful build)     55.8        656          144   0.9911      32       0.081  11542.8       64        0.135    6325.2
Try 6: 8-bit scalar-quantised vectors (diagnostic)     56.6        272          144   0.9877      40       0.087   4384.4       80        0.147    2590.5

----- sift: graph structure and work per query at recall 0.95 (or at the ceiling) -----
                                               try  links used L0  slots L0  at recall  efSearch  dists/query  hops/query
                       Try 1: M=16, efC=100 (base)           19.6        32     0.9568        64         1242          75
      Try 1b: M=16, efC=100 (rebuild, noise check)           19.6        32     0.9568        64         1242          75
                              Try 2: M=8 (sparser)           12.5        16     0.9650       192         1857         206
                              Try 3: M=32 (denser)           24.7        64     0.9553        48         1313          57
                     Try 4: efC=40 (cheaper build)           14.6        32     0.9638       128         1745         139
                    Try 5: efC=200 (careful build)           22.1        32     0.9682        64         1349          75
Try 6: 8-bit scalar-quantised vectors (diagnostic)           19.4        32     0.9598        80         1467          91
nodes per layer (layer 0 first), base: 1,000,000 > 62,403 > 3,903 > 210 > 16 > 1

===== glove: Tries side by side (exact baseline: 32.6 QPS, 400 bytes/vector) =====
                                               try  build s  bytes/vec  graph B/vec  ceiling  ef@0.9  dists%@0.9  QPS@0.9  ef@0.95  dists%@0.95  QPS@0.95
                       Try 1: M=16, efC=100 (base)     39.6        544          144   0.9373     512       0.689    607.7     <NA>          NaN       NaN
      Try 1b: M=16, efC=100 (rebuild, noise check)     46.4        544          144   0.9373     512       0.689    531.1     <NA>          NaN       NaN
                              Try 2: M=8 (sparser)     25.4        481           81   0.8380    <NA>         NaN      NaN     <NA>          NaN       NaN
                              Try 3: M=32 (denser)     53.5        672          272   0.9696     256       0.576   1074.7      768        1.465     329.8
                     Try 4: efC=40 (cheaper build)     15.5        544          144   0.8802    <NA>         NaN      NaN     <NA>          NaN       NaN
                    Try 5: efC=200 (careful build)     71.7        544          144   0.9514     384       0.599   1140.3     1024        1.400     347.0
Try 6: 8-bit scalar-quantised vectors (diagnostic)     74.9        244          144   0.9209     768       0.977    301.8     <NA>          NaN       NaN

----- glove: graph structure and work per query at recall 0.95 (or at the ceiling) -----
                                               try  links used L0  slots L0  at recall  efSearch  dists/query  hops/query
                       Try 1: M=16, efC=100 (base)           21.1        32     0.9373      1024        14809        1037
      Try 1b: M=16, efC=100 (rebuild, noise check)           21.1        32     0.9373      1024        14809        1037
                              Try 2: M=8 (sparser)           11.7        16     0.8380      1024         8187        1049
                              Try 3: M=32 (denser)           30.0        64     0.9610       768        17335         777
                     Try 4: efC=40 (cheaper build)           13.7        32     0.8802      1024        11705        1041
                    Try 5: efC=200 (careful build)           25.2        32     0.9514      1024        16564        1036
Try 6: 8-bit scalar-quantised vectors (diagnostic)           21.1        32     0.9209      1024        14841        1038
nodes per layer (layer 0 first), base: 1,183,514 > 74,078 > 4,644 > 260 > 18 > 1
~~~~

Execution count recorded by the notebook: `10`.

## Cell 23 > markdown

### Prompt

Create notebook cell 23 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Report item 1: recall@10 against queries per second

Same layout as in the other notebooks: thin lines are the individual build settings, the bold black line is the best trade-off curve, the red star is exact FAISS search (same single-query, single-thread protocol). The main figure uses **only the flat-storage HNSW settings (Tries 1 to 5, including the rebuild)**, because that is the index the assignment asks for. The second figure adds the scalar-quantised variant (Try 6).
~~~~

## Cell 24 > code

### Prompt

Implement notebook code cell 24 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell plots recall@10 against QPS for the flat-storage HNSW settings (the curve used in the report) and then again including the scalar-quantised diagnostic (Try 6). Both figures are saved to data/processed/plots.
"""
RES_MAIN = RES[~RES["is_variant"]]
BLD_MAIN = BLD[~BLD["is_variant"]]
fig = plot_tradeoff(RES_MAIN, BASELINE, "HNSW", "./data/processed/plots/hnsw_tradeoff.png")
plt.show()

fig = plot_tradeoff(RES, BASELINE, "HNSW (with SQ8 diagnostic)", "./data/processed/plots/hnsw_with_sq8.png")
plt.show()
~~~~

### Captured output

~~~~text
<Figure size 1400x500 with 2 Axes>
<Figure size 1400x500 with 2 Axes>
~~~~

Execution count recorded by the notebook: `11`.

## Cell 25 > markdown

### Prompt

Create notebook cell 25 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Report items 2 and 3: tables at recall 0.90 and 0.95

Both tables use the **flat-storage HNSW settings only**. The first gives the fastest measured configuration reaching each recall target, with the `efSearch` used. The second gives build time, index size and bytes per vector for the configuration that reaches recall 0.95. Exact search stores `4 x d` bytes per vector (512 for SIFT, 400 for GloVe) and needs no build. HNSW's bytes per vector are *higher* than that, because the links are stored on top of the full vectors.
~~~~

## Cell 26 > code

### Prompt

Implement notebook code cell 26 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell prints the report tables for HNSW: QPS at recall 0.90 and 0.95, and build time / index size / bytes per vector for the configuration that reaches recall 0.95.
"""
print("QPS at recall 0.90 and 0.95 (best HNSW configuration; one query at a time, one thread; param = efSearch)")
print(qps_table(RES_MAIN).to_string(index=False))
print("\nexact FAISS baseline QPS:", {k: round(v["qps"], 1) for k, v in BASELINE.items()})

print(f"\nBuild time, index size and bytes per vector at recall 0.95 (build threads: {MAX_THREADS})")
print(build_table(RES_MAIN, BLD_MAIN).to_string(index=False))
print("\nexact search for comparison: bytes per vector =", {k: 4 * EVAL[k].d for k in DATASETS}, "(no build step)")
~~~~

### Captured output

~~~~text
QPS at recall 0.90 and 0.95 (best HNSW configuration; one query at a time, one thread; param = efSearch)
dataset  target_recall     qps  recall                        setting  param
  glove           0.90  1140.3  0.9066 Try 5: efC=200 (careful build)    384
  glove           0.95   347.0  0.9514 Try 5: efC=200 (careful build)   1024
   sift           0.90 11542.8  0.9117 Try 5: efC=200 (careful build)     32
   sift           0.95  6325.2  0.9682 Try 5: efC=200 (careful build)     64

exact FAISS baseline QPS: {'sift': 29.7, 'glove': 32.6}

Build time, index size and bytes per vector at recall 0.95 (build threads: 20)
dataset                        setting                               build  param  recall  build_s  size_mb  bytes_per_vector
  glove Try 5: efC=200 (careful build) M=16, efConstruction=200, sq8=False   1024  0.9514     71.7    644.2             544.3
   sift Try 5: efC=200 (careful build) M=16, efConstruction=200, sq8=False     64  0.9682     55.8    656.3             656.3

exact search for comparison: bytes per vector = {'sift': 512, 'glove': 400} (no build step)
~~~~

Execution count recorded by the notebook: `12`.

## Cell 27 > markdown

### Prompt

Create notebook cell 27 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Why an index fails to reach high recall: the HNSW case

HNSW computes **exact distances** to the vectors it visits, so unlike IVF-PQ it makes no distance errors, and unlike IVF it has no cell boundaries. A true neighbour is missed only if **the walk never reaches it**. There are two causes:

1. **The search stops too early.** With a small `efSearch` the candidate list fills up with decent but not best vectors and the walk ends before it reaches the true neighbours. Raising `efSearch` fixes this, at a linear cost in time.
2. **The graph has no good path to it.** With too few links (small `M`) or poor construction (small `efConstruction`), some vectors are reachable only through long detours or hardly at all, and the greedy walk cannot find them however long it searches. This shows up as a **plateau**: recall stops improving as `efSearch` grows. Only a better graph (larger `M`, larger `efConstruction`) lifts it.

The checks below make this concrete. The first prints each Try's recall ceiling and whether 0.95 is reached. The second compares Tries at the *same* `efSearch`: if a sparser or hastily built graph is far behind at a large `efSearch`, the limit is the graph, not the search budget. The sentence or two for the report can then say which settings fail to reach 0.95, whether this is a plateau or just a budget that was cut off, and what lifts it.
~~~~

## Cell 28 > code

### Prompt

Implement notebook code cell 28 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell prints the recall ceiling of every Try and compares Tries at the same large efSearch to separate "search stopped too early" from "the graph has no good path".
"""
for ds in DATASETS:
    print(f"[{ds}]")
    for tr, g in RES[RES["dataset"] == ds].groupby("try", sort=False):
        top, last = g["recall"].max(), g.iloc[-1]
        ended = "reached the stop recall" if top >= STOP_RECALL else "plateaued / budget exhausted"
        print(f"  {tr:<56} ceiling recall@10 = {top:.4f} (sweep ended at efSearch={int(last['param'])}: {ended}) "
              f"-> reaches 0.95: {'yes' if top >= 0.95 else 'NO'}")

print("\nRecall at the same efSearch (the largest efSearch that every Try reached):")
for ds in DATASETS:
    g = RES[RES["dataset"] == ds]
    common = set.intersection(*[set(x["param"]) for _, x in g.groupby("try")])
    if common:
        ef = max(common)
        vals = {tr: float(x.loc[x["param"] == ef, "recall"].iloc[0]) for tr, x in g.groupby("try", sort=False)}
        print(f"  [{ds}] efSearch={ef}: " + " | ".join(f"{k.split(':')[0]} {v:.4f}" for k, v in vals.items()))
~~~~

### Captured output

~~~~text
[sift]
  Try 1: M=16, efC=100 (base)                              ceiling recall@10 = 0.9930 (sweep ended at efSearch=192: reached the stop recall) -> reaches 0.95: yes
  Try 1b: M=16, efC=100 (rebuild, noise check)             ceiling recall@10 = 0.9930 (sweep ended at efSearch=192: reached the stop recall) -> reaches 0.95: yes
  Try 2: M=8 (sparser)                                     ceiling recall@10 = 0.9926 (sweep ended at efSearch=768: reached the stop recall) -> reaches 0.95: yes
  Try 3: M=32 (denser)                                     ceiling recall@10 = 0.9916 (sweep ended at efSearch=128: reached the stop recall) -> reaches 0.95: yes
  Try 4: efC=40 (cheaper build)                            ceiling recall@10 = 0.9931 (sweep ended at efSearch=384: reached the stop recall) -> reaches 0.95: yes
  Try 5: efC=200 (careful build)                           ceiling recall@10 = 0.9911 (sweep ended at efSearch=128: reached the stop recall) -> reaches 0.95: yes
  Try 6: 8-bit scalar-quantised vectors (diagnostic)       ceiling recall@10 = 0.9877 (sweep ended at efSearch=768: plateaued / budget exhausted) -> reaches 0.95: yes
[glove]
  Try 1: M=16, efC=100 (base)                              ceiling recall@10 = 0.9373 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: NO
  Try 1b: M=16, efC=100 (rebuild, noise check)             ceiling recall@10 = 0.9373 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: NO
  Try 2: M=8 (sparser)                                     ceiling recall@10 = 0.8380 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: NO
  Try 3: M=32 (denser)                                     ceiling recall@10 = 0.9696 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: yes
  Try 4: efC=40 (cheaper build)                            ceiling recall@10 = 0.8802 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: NO
  Try 5: efC=200 (careful build)                           ceiling recall@10 = 0.9514 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: yes
  Try 6: 8-bit scalar-quantised vectors (diagnostic)       ceiling recall@10 = 0.9209 (sweep ended at efSearch=1024: plateaued / budget exhausted) -> reaches 0.95: NO

Recall at the same efSearch (the largest efSearch that every Try reached):
  [sift] efSearch=128: Try 1 0.9854 | Try 1b 0.9854 | Try 2 0.9451 | Try 3 0.9916 | Try 4 0.9638 | Try 5 0.9911 | Try 6 0.9750
  [glove] efSearch=1024: Try 1 0.9373 | Try 1b 0.9373 | Try 2 0.8380 | Try 3 0.9696 | Try 4 0.8802 | Try 5 0.9514 | Try 6 0.9209
~~~~

Execution count recorded by the notebook: `13`.

## Cell 29 > markdown

### Prompt

Create notebook cell 29 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Saving the results

The sweep points and build records are saved for the final cross-index comparison and the tweak experiments.
~~~~

## Cell 30 > code

### Prompt

Implement notebook code cell 30 of 31. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell saves all sweep points and build records of this notebook as csv files in data/processed.
"""
print("saved:", save_df(RES, "results_hnsw.csv"))
print("saved:", save_df(BLD, "builds_hnsw.csv"))
print(f"{len(RES)} sweep points over {RES['try'].nunique()} Tries, {len(BLD)} index builds")
~~~~

### Captured output

~~~~text
saved: ./data/processed/results_hnsw.csv
saved: ./data/processed/builds_hnsw.csv
225 sweep points over 7 Tries, 14 index builds
~~~~

Execution count recorded by the notebook: `14`.

## Cell 31 > markdown

### Prompt

Create notebook cell 31 of 31 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Observations (fill in after running)

Use these prompts to turn the output above into the explanations the report asks for.

- **Noise** (Try 1 vs 1b): how far apart are two identical builds> Treat any difference smaller than that as not significant.
- **Links per node** (Tries 1 to 3): how do the ceiling, QPS at recall 0.95, graph bytes per vector and build time change with `M`> Is there a break-even where more links stop paying off>
- **Construction effort** (Tries 4, 1, 5): how much recall at equal `efSearch` does a longer build buy, and what does it cost in build time> Is the index size really unchanged>
- **Storage** (Try 6): how much of the index size is raw vectors, and what does compressing them do to the ceiling>
- **Work per query:** what share of the database does HNSW compare at recall 0.95, compared with IVF-Flat and IVF-PQ>
- **SIFT vs GloVe:** which dataset needs the larger `efSearch`, and does any setting plateau below 0.95 on one of them>
- **Cost of the speed:** compare bytes per vector and build time with exact search (512 / 400 bytes, no build), IVF-Flat and IVF-PQ. HNSW's strength is speed at high recall; the price is memory and build time.
~~~~
