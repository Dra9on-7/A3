# AI Log: complete creation of `SIFT_experiment.ipynb`

This log reconstructs the notebook from an empty file to its final executed state. Each cell has a implementation prompt followed by a `Response` containing the complete source for that cell. Outputs are included when the notebook captured them.

## Versioned implementation milestones

1. **Version 1 > scaling experiment configuration and data loading**
2. **Version 2 > per-scale exact references**
3. **Version 3 > HNSW and IVF-Flat scaling sweeps**
4. **Version 4 > target selection, plots, exports, and validation**

## Creation sequence

- Notebook cells: 21
- Markdown cells: 11
- Code cells: 10
- Execution order: top to bottom; each response depends on prior cells.

## Cell 1 > markdown

### Prompt

Create notebook cell 1 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
# SIFT scaling experiment: cost of reaching recall@10 = 0.95

This experiment measures how the cost of reaching **recall@10 = 0.95** changes as the SIFT database grows from **100K to 300K to 1M vectors** for:

- **Exact search** (`IndexFlatL2`), the quality and latency reference.
- **HNSW**, with a fixed graph construction policy and an `efSearch` sweep.
- **IVF-Flat**, with a scale-aware `nlist` and an `nprobe` sweep.

The comparison is made at the fastest measured operating point whose recall is at least 0.95. All methods use the same 10,000 queries, `k=10`, one query per `search()` call, one FAISS thread for timing, and three timing repeats reported as the median. Build time and serialized index size are retained because scaling cost is not only query latency.
~~~~

## Cell 2 > markdown

### Prompt

Create notebook cell 2 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Experimental decisions and alternatives

**Database sizes.** The first 100K and first 300K vectors of the official SIFT training set are used, plus the complete 1M set. This keeps the data distribution consistent while making the experiment reproducible. A random subset could have been chosen, but then differences between sizes would mix scaling with sampling noise.

**Ground truth.** Exact top-10 neighbours are recomputed independently for each subset. Reusing the 1M ground truth would be incorrect: a neighbour outside a smaller subset cannot be returned by an index built on that subset.

**HNSW.** `M=16` and `efConstruction=100` are held fixed so only database size changes. A different choice such as `M=8` would reduce memory, while `M=32` could improve recall, but either would change the index design rather than isolate scaling. `efSearch` is swept because it is the query-time cost knob.

**IVF-Flat.** `nlist` is set close to `sqrt(N)` (316, 548, and 1000), keeping average list size approximately comparable across scales. A fixed `nlist=1024` could have been used for all sizes, but it would make the 100K case unusually fine and the 1M case unusually coarse. `nprobe` is swept because it controls query work.

**Exact search.** Exact search is measured rather than treated as free. It is the baseline cost at recall 1.0 and shows how the ANN methods scale relative to exhaustive comparison.
~~~~

## Cell 3 > code

### Prompt

Implement notebook code cell 3 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
import gc
import time

import faiss
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ann_utils import K, MAX_THREADS, exact_dists, time_single_query, index_size_bytes

N_TIME_Q = 500
REPS = 3
TARGET_RECALL = 0.95
SEED = 1234
SIZES = (100_000, 300_000, 1_000_000)
EFSEARCH_GRID = [10, 16, 24, 32, 40, 48, 64, 80, 96, 128, 192, 256, 384, 512]
NPROBE_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

with h5py.File('./data/raw/sift-128-euclidean.hdf5', 'r') as f:
    train_all = np.ascontiguousarray(f['train'][:], dtype=np.float32)
    test = np.ascontiguousarray(f['test'][:], dtype=np.float32)


class ScaledSIFT:
    def __init__(self, n):
        self.name = f'sift_{n // 1000}k'
        self.n = n
        self.train = train_all[:n]
        self.test = test
        self.d = self.train.shape[1]
        self.metric = 'l2'
        self.gt_ids = None
        self.gt_d = None

    def recall(self, ids):
        ret = np.asarray(ids)[:, :K]
        ret_d = exact_dists(ret, self.train, self.test, self.metric)
        from ann_utils import recall_from_dists
        return recall_from_dists(ret, ret_d, self.gt_d, K)


EVALS = {n: ScaledSIFT(n) for n in SIZES}
RESULTS = []
BUILDS = []
print(f'FAISS {faiss.__version__} | build threads: {MAX_THREADS} | timing threads: 1')
print(f'train shape: {train_all.shape}, test shape: {test.shape}')
~~~~

### Captured output

~~~~text
FAISS 1.15.1 | build threads: 20 | timing threads: 1
train shape: (1000000, 128), test shape: (10000, 128)
~~~~

Execution count recorded by the notebook: `1`.

## Cell 4 > markdown

### Prompt

Create notebook cell 4 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Exact baseline: one try per scale

The exact index is built separately for each database size. Its recall is 1.0 by construction, and its single-query QPS is the reference against which ANN latency is compared.
~~~~

## Cell 5 > code

### Prompt

Implement notebook code cell 5 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
BASELINES = {}
for n, ev in EVALS.items():
    exact = faiss.IndexFlatL2(ev.d)
    exact.add(ev.train)
    _, ev.gt_ids = exact.search(ev.test, K)
    ev.gt_d = exact_dists(ev.gt_ids, ev.train, ev.test, ev.metric)
    qps, _, _ = time_single_query(exact, ev.test, K, N_TIME_Q, REPS)
    BASELINES[n] = {'qps': qps, 'ms_per_query': 1e3 / qps}
    RESULTS.append({'index': 'Exact', 'size': n, 'try': 'exact', 'param': 0,
                    'recall': 1.0, 'qps': qps, 'ms_per_query': 1e3 / qps})
    print(f'{ev.name}: recall=1.0000 | {qps:,.1f} QPS | {1e3 / qps:.3f} ms/query')
    del exact
    gc.collect()
~~~~

### Captured output

~~~~text
sift_100k: recall=1.0000 | 282.2 QPS | 3.544 ms/query
sift_300k: recall=1.0000 | 72.0 QPS | 13.893 ms/query
sift_1000k: recall=1.0000 | 22.4 QPS | 44.730 ms/query
~~~~

Execution count recorded by the notebook: `2`.

## Cell 6 > markdown

### Prompt

Create notebook cell 6 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 1: HNSW at 100K

**Decision.** Build HNSW with `M=16`, `efConstruction=100`, then sweep `efSearch`.

**Reasoning.** This is the base HNSW design used in the earlier benchmark. Holding construction parameters fixed makes the result a scaling measurement, while `efSearch` exposes the cheapest point that reaches the target.

**Alternative not chosen.** Re-tuning `M` for 100K might produce a faster index, but it would answer an engineering-tuning question rather than the requested fixed-design scaling question.
~~~~

## Cell 7 > code

### Prompt

Implement notebook code cell 7 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
def run_hnsw_try(n):
    ev = EVALS[n]
    index = faiss.IndexHNSWFlat(ev.d, 16)
    index.hnsw.efConstruction = 100
    faiss.omp_set_num_threads(MAX_THREADS)
    t0 = time.perf_counter()
    index.add(ev.train)
    build_s = time.perf_counter() - t0
    print(f'[{ev.name}] HNSW build: {build_s:.1f}s')
    rows = []
    for ef in EFSEARCH_GRID:
        index.hnsw.efSearch = ef
        _, ids = index.search(ev.test, K)
        recall = ev.recall(ids)
        qps, qps_reps, extra = time_single_query(index, ev.test, K, N_TIME_Q, REPS, collect_stats='hnsw')
        row = {'index': 'HNSW', 'size': n, 'try': 'M16_efC100', 'param': ef,
               'recall': recall, 'qps': qps, 'ms_per_query': 1e3 / qps,
               'qps_min': min(qps_reps), 'qps_max': max(qps_reps), **extra}
        rows.append(row)
        print(f'  efSearch={ef:<4} recall={recall:.4f} | {qps:,.1f} QPS')
        if recall >= 0.99:
            break
    size_mb = index_size_bytes(index) / 1e6
    BUILDS.append({'index': 'HNSW', 'size': n, 'try': 'M16_efC100',
                    'build_s': build_s, 'size_mb': size_mb,
                    'bytes_per_vector': size_mb * 1e6 / n})
    RESULTS.extend(rows)
    del index
    gc.collect()

run_hnsw_try(100_000)
~~~~

### Captured output

~~~~text
[sift_100k] HNSW build: 1.3s
  efSearch=10   recall=0.7795 | 23,630.9 QPS
  efSearch=16   recall=0.8601 | 15,309.7 QPS
  efSearch=24   recall=0.9153 | 11,149.7 QPS
  efSearch=32   recall=0.9457 | 9,600.6 QPS
  efSearch=40   recall=0.9625 | 9,106.5 QPS
  efSearch=48   recall=0.9734 | 7,260.0 QPS
  efSearch=64   recall=0.9845 | 6,545.2 QPS
  efSearch=80   recall=0.9903 | 5,057.3 QPS
~~~~

Execution count recorded by the notebook: `3`.

## Cell 8 > markdown

### Prompt

Create notebook cell 8 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 2: HNSW at 300K

**Decision.** Repeat exactly the 100K HNSW protocol at 300K.

**Reasoning.** The same `M`, `efConstruction`, timing protocol, and search grid make changes in build time, memory, and target latency attributable to the larger database.

**Alternative not chosen.** Increasing `efConstruction` at 300K could improve graph quality, but it would add a second changing variable and make the scaling curve ambiguous.
~~~~

## Cell 9 > code

### Prompt

Implement notebook code cell 9 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
run_hnsw_try(300_000)
~~~~

### Captured output

~~~~text
[sift_300k] HNSW build: 7.0s
  efSearch=10   recall=0.7364 | 15,284.0 QPS
  efSearch=16   recall=0.8266 | 11,942.9 QPS
  efSearch=24   recall=0.8875 | 11,628.1 QPS
  efSearch=32   recall=0.9215 | 11,061.2 QPS
  efSearch=40   recall=0.9435 | 8,859.9 QPS
  efSearch=48   recall=0.9578 | 8,073.6 QPS
  efSearch=64   recall=0.9735 | 4,951.7 QPS
  efSearch=80   recall=0.9826 | 4,392.0 QPS
  efSearch=96   recall=0.9878 | 4,243.8 QPS
  efSearch=128  recall=0.9930 | 2,975.2 QPS
~~~~

Execution count recorded by the notebook: `4`.

## Cell 10 > markdown

### Prompt

Create notebook cell 10 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 3: HNSW at 1M

**Decision.** Repeat the fixed HNSW protocol on the complete SIFT training set.

**Reasoning.** This is the requested large-scale endpoint and is directly comparable with the existing 1M HNSW benchmark.

**Alternative not chosen.** A wider `efSearch` grid above 512 was not added because the target is 0.95, not 0.99; the sweep stops once the measured target is reached (or at 0.99).
~~~~

## Cell 11 > code

### Prompt

Implement notebook code cell 11 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
run_hnsw_try(1_000_000)
~~~~

### Captured output

~~~~text
[sift_1000k] HNSW build: 31.3s
  efSearch=10   recall=0.7003 | 13,423.0 QPS
  efSearch=16   recall=0.7910 | 11,532.2 QPS
  efSearch=24   recall=0.8566 | 6,916.6 QPS
  efSearch=32   recall=0.8946 | 8,314.5 QPS
  efSearch=40   recall=0.9197 | 7,750.6 QPS
  efSearch=48   recall=0.9354 | 7,159.0 QPS
  efSearch=64   recall=0.9568 | 5,099.0 QPS
  efSearch=80   recall=0.9692 | 4,611.5 QPS
  efSearch=96   recall=0.9766 | 3,449.5 QPS
  efSearch=128  recall=0.9854 | 2,674.6 QPS
  efSearch=192  recall=0.9930 | 2,007.3 QPS
~~~~

Execution count recorded by the notebook: `5`.

## Cell 12 > markdown

### Prompt

Create notebook cell 12 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 4: IVF-Flat at 100K

**Decision.** Use `nlist=316`, approximately `sqrt(100K)`, and sweep `nprobe`.

**Reasoning.** Keeping the average list size near `sqrt(N)` preserves the balance between centroid work and list scanning as the database grows.

**Alternative not chosen.** A fixed `nlist=1024` would make this small database unusually fragmented and would not be a fair scale-aware comparison.
~~~~

## Cell 13 > code

### Prompt

Implement notebook code cell 13 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
def run_ivf_try(n):
    ev = EVALS[n]
    nlist = int(round(np.sqrt(n)))
    quantizer = faiss.IndexFlatL2(ev.d)
    index = faiss.IndexIVFFlat(quantizer, ev.d, nlist, faiss.METRIC_L2)
    index.cp.niter, index.cp.seed = 25, SEED
    faiss.omp_set_num_threads(MAX_THREADS)
    t0 = time.perf_counter()
    index.train(ev.train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.add(ev.train)
    add_s = time.perf_counter() - t0
    print(f'[{ev.name}] IVF-Flat nlist={nlist}: train={train_s:.1f}s, add={add_s:.1f}s')
    rows = []
    for nprobe in dict.fromkeys(min(v, nlist) for v in NPROBE_GRID):
        index.nprobe = nprobe
        _, ids = index.search(ev.test, K)
        recall = ev.recall(ids)
        qps, qps_reps, extra = time_single_query(index, ev.test, K, N_TIME_Q, REPS, collect_stats='ivf')
        row = {'index': 'IVF-Flat', 'size': n, 'try': f'nlist{nlist}', 'param': nprobe,
               'recall': recall, 'qps': qps, 'ms_per_query': 1e3 / qps,
               'qps_min': min(qps_reps), 'qps_max': max(qps_reps), **extra}
        rows.append(row)
        print(f'  nprobe={nprobe:<4} recall={recall:.4f} | {qps:,.1f} QPS')
        if recall >= 0.99:
            break
    size_mb = index_size_bytes(index) / 1e6
    BUILDS.append({'index': 'IVF-Flat', 'size': n, 'try': f'nlist{nlist}',
                'build_s': train_s + add_s, 'train_s': train_s, 'add_s': add_s,
                'size_mb': size_mb, 'bytes_per_vector': size_mb * 1e6 / n})
    RESULTS.extend(rows)
    del index
    gc.collect()

run_ivf_try(100_000)
~~~~

### Captured output

~~~~text
[sift_100k] IVF-Flat nlist=316: train=1.6s, add=0.1s
  nprobe=1    recall=0.4062 | 29,081.4 QPS
  nprobe=2    recall=0.5855 | 15,262.0 QPS
  nprobe=4    recall=0.7566 | 11,097.0 QPS
  nprobe=8    recall=0.8898 | 6,595.7 QPS
  nprobe=12   recall=0.9411 | 3,933.1 QPS
  nprobe=16   recall=0.9661 | 3,397.7 QPS
  nprobe=24   recall=0.9869 | 2,402.6 QPS
  nprobe=32   recall=0.9943 | 1,744.1 QPS
~~~~

Execution count recorded by the notebook: `6`.

## Cell 14 > markdown

### Prompt

Create notebook cell 14 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 5: IVF-Flat at 300K

**Decision.** Repeat IVF-Flat with the scale-aware `nlist=round(sqrt(300K))=548`.

**Reasoning.** This preserves approximately the same average number of vectors per inverted list as the 100K and 1M tries.

**Alternative not chosen.** Reusing the 100K trained centroids would make the partition mismatched to the larger data distribution and would conflate poor training with scaling.
~~~~

## Cell 15 > code

### Prompt

Implement notebook code cell 15 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
run_ivf_try(300_000)
~~~~

### Captured output

~~~~text
[sift_300k] IVF-Flat nlist=548: train=4.5s, add=0.5s
  nprobe=1    recall=0.3871 | 14,854.8 QPS
  nprobe=2    recall=0.5574 | 11,406.5 QPS
  nprobe=4    recall=0.7291 | 6,522.3 QPS
  nprobe=8    recall=0.8662 | 4,164.4 QPS
  nprobe=12   recall=0.9222 | 2,419.0 QPS
  nprobe=16   recall=0.9503 | 1,776.0 QPS
  nprobe=24   recall=0.9775 | 1,390.0 QPS
  nprobe=32   recall=0.9882 | 1,210.0 QPS
  nprobe=48   recall=0.9964 | 746.1 QPS
~~~~

Execution count recorded by the notebook: `7`.

## Cell 16 > markdown

### Prompt

Create notebook cell 16 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Try 6: IVF-Flat at 1M

**Decision.** Repeat IVF-Flat with `nlist=round(sqrt(1M))=1000`.

**Reasoning.** This is close to the `nlist=1024` base used in the earlier 1M benchmark, so the result can be compared with the previous run while following the same scaling rule.

**Alternative not chosen.** `nlist=4096` would reduce list sizes but increase the fixed centroid-comparison cost; it is a useful tuning experiment, not the cleanest scaling baseline.
~~~~

## Cell 17 > code

### Prompt

Implement notebook code cell 17 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
run_ivf_try(1_000_000)
~~~~

### Captured output

~~~~text
[sift_1000k] IVF-Flat nlist=1000: train=16.7s, add=2.8s
  nprobe=1    recall=0.3773 | 10,282.4 QPS
  nprobe=2    recall=0.5431 | 6,487.9 QPS
  nprobe=4    recall=0.7055 | 3,503.3 QPS
  nprobe=8    recall=0.8435 | 1,643.0 QPS
  nprobe=12   recall=0.9029 | 1,107.2 QPS
  nprobe=16   recall=0.9344 | 1,002.9 QPS
  nprobe=24   recall=0.9663 | 816.0 QPS
  nprobe=32   recall=0.9807 | 654.0 QPS
  nprobe=48   recall=0.9924 | 401.4 QPS
~~~~

Execution count recorded by the notebook: `8`.

## Cell 18 > markdown

### Prompt

Create notebook cell 18 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Results: cost of reaching recall 0.95

For each scale and index, the selected row is the fastest measured point with recall at least 0.95. `not reached` is reported explicitly rather than silently substituting a lower-recall point.
~~~~

## Cell 19 > code

### Prompt

Implement notebook code cell 19 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
results_df = pd.DataFrame(RESULTS)
builds_df = pd.DataFrame(BUILDS)
selected = []
for (index, n), group in results_df.groupby(['index', 'size'], sort=True):
    eligible = group[group.recall >= TARGET_RECALL]
    if eligible.empty:
        selected.append({'index': index, 'size': n, 'status': 'not reached',
                         'param': np.nan, 'recall': group.recall.max(),
                         'qps': np.nan, 'ms_per_query': np.nan})
    else:
        row = eligible.loc[eligible.qps.idxmax()]
        selected.append({'index': index, 'size': n, 'status': 'reached',
                         'param': row.param, 'recall': row.recall,
                         'qps': row.qps, 'ms_per_query': row.ms_per_query})
cost_df = pd.DataFrame(selected)
cost_df = cost_df.merge(builds_df[['index', 'size', 'build_s', 'size_mb', 'bytes_per_vector']],
                        on=['index', 'size'], how='left')
cost_df['exact_qps'] = cost_df['size'].map({n: BASELINES[n]['qps'] for n in SIZES})
cost_df['speedup_vs_exact'] = cost_df['exact_qps'] / cost_df['qps']
display(cost_df.sort_values(['size', 'index']).round(4))
results_df.to_csv('./data/processed/results_sift_scaling.csv', index=False)
cost_df.to_csv('./data/processed/results_sift_scaling_at_095.csv', index=False)
builds_df.to_csv('./data/processed/builds_sift_scaling.csv', index=False)
print('Saved scaling sweep, target-cost, and build tables under data/processed/.')
~~~~

### Captured output

~~~~text
      index     size   status  param  recall        qps  ms_per_query  \
0     Exact   100000  reached      0  1.0000   282.1611        3.5441   
3      HNSW   100000  reached     40  0.9625  9106.4729        0.1098   
6  IVF-Flat   100000  reached     16  0.9661  3397.6878        0.2943   
1     Exact   300000  reached      0  1.0000    71.9784       13.8931   
4      HNSW   300000  reached     48  0.9578  8073.5533        0.1239   
7  IVF-Flat   300000  reached     16  0.9503  1776.0147        0.5631   
2     Exact  1000000  reached      0  1.0000    22.3563       44.7301   
5      HNSW  1000000  reached     64  0.9568  5099.0022        0.1961   
8  IVF-Flat  1000000  reached     24  0.9663   816.0044        1.2255   

   build_s   size_mb  bytes_per_vector  exact_qps  speedup_vs_exact  
0      NaN       NaN               NaN   282.1611            1.0000  
3   1.3141   65.6333          656.3327   282.1611            0.0310  
6   1.6416   52.1645          521.6446   282.1611            0.0830  
1      NaN       NaN               NaN    71.9784            1.0000  
4   7.0443  196.8839          656.2797    71.9784            0.0089  
7   4.9960  156.2851          520.9503    71.9784            0.0405  
2      NaN       NaN               NaN    22.3563            1.0000  
5  31.3436  656.2584          656.2584    22.3563            0.0044  
8  19.4749  520.5201          520.5201    22.3563            0.0274  
Saved scaling sweep, target-cost, and build tables under data/processed/.
~~~~

Execution count recorded by the notebook: `9`.

## Cell 20 > markdown

### Prompt

Create notebook cell 20 of 21 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Scaling plot

The left panel shows query latency at the target, while the right panel shows build time. Exact search has no ANN build phase, so its build-time bar is intentionally absent.
~~~~

## Cell 21 > code

### Prompt

Implement notebook code cell 21 of 21. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
ann_cost = cost_df[cost_df['index'] != 'Exact'].copy()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for index, group in ann_cost.groupby('index'):
    group = group.sort_values('size')
    axes[0].plot(group['size'], group['ms_per_query'], 'o-', label=index)
    axes[1].plot(group['size'], group['build_s'], 'o-', label=index)
axes[0].set_title('Query cost at recall@10 >= 0.95')
axes[0].set_xlabel('database vectors')
axes[0].set_ylabel('milliseconds per query (log scale)')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[1].set_title('ANN build cost')
axes[1].set_xlabel('database vectors')
axes[1].set_ylabel('seconds (log scale)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
for ax in axes:
    ax.grid(alpha=0.3, which='both')
    ax.legend()
fig.tight_layout()
fig.savefig('./data/processed/plots/sift_scaling_at_095.png', dpi=150)
plt.show()
~~~~

### Captured output

~~~~text
<Figure size 1300x500 with 2 Axes>
~~~~

Execution count recorded by the notebook: `10`.
