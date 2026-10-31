# AI Log: complete creation of `faiss_index_analysis.ipynb`

This log reconstructs the notebook from an empty file to its final executed state. Each cell has a implementation prompt followed by a `Response` containing the complete source for that cell. Outputs are included when the notebook captured them.

## Versioned implementation milestones

1. **Version 1 > dataset loading and FAISS metric setup**
2. **Version 2 > exact search and tie-aware recall implementation**
3. **Version 3 > timing experiments and reference verification**
4. **Version 4 > persisted references and validation**

## Creation sequence

- Notebook cells: 19
- Markdown cells: 11
- Code cells: 8
- Execution order: top to bottom; each response depends on prior cells.

## Cell 1 > markdown

### Prompt

Create notebook cell 1 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
# FAISS exact search: the baseline for all ANN experiments

Every approximate index we test later (HNSW, IVF-Flat, IVF-PQ) will be judged on two things: **how many of the true nearest neighbours it finds (recall@10)** and **how fast it answers queries (QPS / latency)**. Both need a trustworthy reference, so this notebook builds one.

We already have two reference points from the dataset notebooks: a random indexer (fast, useless accuracy) and a pure-Python brute-force indexer (exact, about 1.7 seconds per query on SIFT 1M). Here we add the third and most important one, **exact search with FAISS** (`IndexFlatL2` / `IndexFlatIP`). It returns exactly the same answers as brute force but is heavily optimised, so it is the fastest *exact* method and the fair yardstick for the approximate ones.

The notebook does the following, in order:
1. Load SIFT 1M (Euclidean) and GloVe 100 (angular) and prepare them for FAISS.
2. Build the exact FAISS indexes.
3. Define a **tie-aware recall@k** function and unit test it.
4. Check that FAISS exact results agree with the neighbours stored in the HDF5 files.
5. Time exact search **one query at a time** and **with all queries in one batch**, and explain the difference.
6. Save the reference results so the HNSW / IVF notebooks can reuse them.
~~~~

## Cell 2 > markdown

### Prompt

Create notebook cell 2 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## 1. Loading the datasets

Both files have the same four components we saw in `sift_analysis.ipynb`: `train` (the vectors we index), `test` (the queries), `neighbors` (ids of the 100 true nearest neighbours per query) and `distances` (their distances).

The two datasets use different distance measures, which changes how FAISS must be set up:

| Dataset | Dim | Metric | FAISS index | Preparation |
|---|---|---|---|---|
| SIFT 1M | 128 | Euclidean (L2) | `IndexFlatL2` | none |
| GloVe 100 | 100 | Angular (cosine) | `IndexFlatIP` | L2-normalise train **and** test |

FAISS has no dedicated cosine index. For unit-length vectors, cosine similarity is exactly the inner product, so we normalise every vector and use inner product. Ranking by largest inner product is then identical to ranking by smallest angular distance (`1 - cos`).
~~~~

## Cell 3 > code

### Prompt

Implement notebook code cell 3 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell loads both datasets from the hdf5 files into float32 arrays (the type FAISS requires). For the angular dataset (GloVe) the vectors are L2-normalised so that inner product equals cosine similarity. A summary table of the shapes is printed at the end.
"""

import os
import json
import time

import h5py
import numpy as np
import pandas as pd
import faiss # generally used for similarity search and clustering of dense vectors
import tqdm

K = 10          # we report recall@10
K_REF = 100     # depth of the reference lists we store (the files contain 100 neighbours)

DATASETS = {
    "sift":  dict(path="./data/raw/sift-128-euclidean.hdf5", metric="l2"),
    "glove": dict(path="./data/raw/glove-100-angular.hdf5",    metric="angular"),
}

def load_dataset(name):
    cfg = DATASETS[name]
    with h5py.File(cfg["path"], "r") as f:
        train   = np.ascontiguousarray(f["train"][:],     dtype=np.float32) # load the training data from the HDF5 file and convert it to a contiguous array of type float32, looks like [1.0, 2.0, 3.0, ...]
        test    = np.ascontiguousarray(f["test"][:],      dtype=np.float32)
        gt_ids  = f["neighbors"][:] # load the ground truth neighbor indices from the HDF5 file, looks like [[0, 1, 2, ...], [3, 4, 5, ...], ...], .ascontiguousarray is not needed here because these are integer indices, not floating-point numbers
        gt_dist = f["distances"][:] # load the ground truth distances from the HDF5 file, looks like [[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...], ...], .ascontiguousarray is not needed here because these are floating-point numbers, but they are not used in the same way as the training and test data
    norm_before = float(np.linalg.norm(train[:1000], axis=1).mean()) # compute the mean L2 norm of the first 1000 training vectors before any normalization, this is used to check if the normalization worked correctly
    if cfg["metric"] == "angular":
        faiss.normalize_L2(train)   # in place
        faiss.normalize_L2(test)
    norm_after = float(np.linalg.norm(train[:1000], axis=1).mean())
    return dict(train=train, test=test, gt_ids=gt_ids, gt_dist=gt_dist,
                metric=cfg["metric"], norm_before=norm_before, norm_after=norm_after)

data = {}
rows = []
for name in tqdm.tqdm(DATASETS):
    data[name] = load_dataset(name)
    d = data[name]
    rows.append({
        "dataset": name,
        "metric": d["metric"],
        "train": d["train"].shape,
        "test": d["test"].shape,
        "neighbors": d["gt_ids"].shape,
        "distances": d["gt_dist"].shape,
        "mean |x| before": round(d["norm_before"], 3),
        "mean |x| after": round(d["norm_after"], 3),
    })

print("FAISS version:", faiss.__version__)
print("OpenMP threads available to FAISS:", faiss.omp_get_max_threads())
print(pd.DataFrame(rows).to_string(index=False))
~~~~

### Captured output

~~~~text
100%|██████████| 2/2 [00:00<00:00,  2.55it/s]
FAISS version: 1.15.1
OpenMP threads available to FAISS: 20
dataset  metric          train         test    neighbors    distances  mean |x| before  mean |x| after
   sift      l2 (1000000, 128) (10000, 128) (10000, 100) (10000, 100)          508.531         508.531
  glove angular (1183514, 100) (10000, 100) (10000, 100) (10000, 100)            5.705           1.000
~~~~

Execution count recorded by the notebook: `1`.

## Cell 4 > markdown

### Prompt

Create notebook cell 4 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## 2. Building the exact FAISS indexes

A flat index does no clever indexing at all: `add()` simply stores the raw vectors, and `search()` compares each query against **every** stored vector. That is why it is exact, and also why its cost grows linearly with the dataset size. Memory use is `N x d x 4` bytes (about 512 MB for SIFT 1M).

Note on distances: `IndexFlatL2` returns **squared** L2 distances (it skips the square root because ranking is unaffected), and `IndexFlatIP` returns **cosine similarities** (larger is better). The HDF5 files store true Euclidean distance and `1 - cos`, so we convert when comparing (`to_file_units` below).
~~~~

## Cell 5 > code

### Prompt

Implement notebook code cell 5 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell builds one flat (exact) FAISS index per dataset and reports how long adding the vectors took and how much memory the stored vectors need. Flat indexes need no training step.
"""

def build_flat(base, metric):
    d = base.shape[1]
    index = faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)
    index.add(base)
    # print(f"IndexFlat{metric.upper()} ntotal={index.ntotal:,}, d={index.d}")
    # print first 5 vectors in the index
    for i in range(min(5, index.ntotal)):
        print(f"  {i}: {index.reconstruct(i)}")
    return index

def to_file_units(D, metric):
    """FAISS distances -> the units used in the hdf5 'distances' field."""
    if metric == "l2":
        return np.sqrt(np.maximum(D, 0))   # IndexFlatL2 returns squared L2
    return 1.0 - D                         # IndexFlatIP returns cosine similarity

indexes = {}
for name in DATASETS:
    d = data[name]
    t0 = time.perf_counter()
    indexes[name] = build_flat(d["train"], d["metric"])
    build_s = time.perf_counter() - t0
    idx = indexes[name]
    print(f"[{name}] {type(idx).__name__}: ntotal={idx.ntotal:,}, d={idx.d}, "
          f"build time={build_s:.2f} s, stored vectors={d['train'].nbytes / 1e6:.0f} MB")
~~~~

### Captured output

~~~~text
  0: [  0.  16.  35.   5.  32.  31.  14.  10.  11.  78.  55.  10.  45.  83.
  11.   6.  14.  57. 102.  75.  20.   8.   3.   5.  67.  17.  19.  26.
   5.   0.   1.  22.  60.  26.   7.   1.  18.  22.  84.  53.  85. 119.
 119.   4.  24.  18.   7.   7.   1.  81. 106. 102.  72.  30.   6.   0.
   9.   1.   9. 119.  72.   1.   4.  33. 119.  29.   6.   1.   0.   1.
  14.  52. 119.  30.   3.   0.   0.  55.  92. 111.   2.   5.   4.   9.
  22.  89.  96.  14.   1.   0.   1.  82.  59.  16.  20.   5.  25.  14.
  11.   4.   0.   0.   1.  26.  47.  23.   4.   0.   0.   4.  38.  83.
  30.  14.   9.   4.   9.  17.  23.  41.   0.   0.   2.   8.  19.  25.
  23.   1.]
  1: [ 14.  35.  19.  20.   3.   1.  13.  11.  16. 119.  85.   5.   0.   5.
  24.  26.   0.  27. 119.  13.   3.   9.  19.   0.   0.  11.  73.   9.
  10.   3.   5.   0.  92.  38.  17.  39.  32.   7.  15.  47. 119. 111.
  53.  27.   8.   0.   0.  52.   5.   7.  63.  51.  84.  43.   0.   1.
  12.   8.  20.  25.  33.  30.   2.   5.  59.  23.  25. 105.  25.  23.
   5.  18. 119.  15.   7.  13.  14.  19.  95. 119.   5.   0.   0.  14.
 119. 103.  93.  39.  11.   4.   1.   4.  13.  43.  62.  18.   2.   0.
   0.   8.  44.  65.   7.   1.   3.   0.   0.   1.  19.  45.  94.  95.
  13.   7.   0.   0.   3.  52. 119.  52.  15.   2.   0.   0.   0.  11.
  21.  33.]
  2: [  0.   1.   5.   3.  44.  40.  20.  14.  10. 100.  63.   7.  44.  47.
   9.   6.   7.  70. 114.  82.  11.   5.   3.   4.  30.  16.  33.  55.
   5.   0.   0.   4. 104.  46.  12.   2.   8.   5.  40.  66.  85. 117.
 117.   5.   7.   3.   1.   6.   0.  62. 108.  86.  51.  22.   2.   0.
   7.   1.  15. 117.  95.   2.   1.  12. 117.  33.   4.   2.   0.   0.
   3.  56. 117.  20.   3.   0.   0.  40.  78.  93.   0.   1.   2.   5.
  24.  86.  96.  16.   0.   0.   0.  77.  75.  10.  19.   7.  22.   8.
  12.   7.   0.   0.   3.  44.  35.  11.   2.   0.   0.  10.  55. 117.
  17.   9.  12.   2.   1.  14.  34.  31.   0.   1.   6.  10.  12.   4.
  23.  10.]
  3: [ 12.  47.  14.  25.   2.   3.   4.   7.  14. 122.  90.   7.   0.   0.
   6.  14.   0.  24. 122.  22.   2.   4.   6.   0.   0.  10.  93.  10.
   6.   6.   0.   0. 122.  31.   9.  23.  19.   9.   8.  56. 122. 100.
  29.  19.   3.   0.   0.  25.   3.   9.  43.  59.  76.  32.   0.   0.
   8.   6.  10.   7.  24.  58.   1.   1.  81.  23.  32.  68.  14.  19.
  10.  23. 122.  13.   2.   1.   4.   9.  86. 122.   3.   0.   0.   8.
 122.  95.  68.  30.   9.   2.   0.   2.  26.  50.  44.  13.   0.   0.
   0.   3.  12.  82.  18.   7.   6.   0.   0.   0.   2.  20. 112. 122.
   6.   5.   1.   0.   3.  69. 122.  43.  15.   1.   0.   0.   0.  27.
  29.  21.]
  4: [  1.   1.   0.   0.  14.  16.  30.  50.   2.  40.  81.  25.  65.  37.
   7.   3.  13. 121. 121.  15.   2.   4.   4.   6.   0.  11.  71. 106.
   2.   2.   0.   0.  92.  41.   1.   0.   1.   2.  24.  65. 121. 121.
 110.   5.   2.   1.   1.  12.   6.  96. 121.  13.   4.   4.   1.   0.
   0.   4.  46. 118.  46.   9.   0.   0. 121.  33.   8.   7.   0.   0.
   0.  42. 121.  20.   0.   0.   0.   2.  33. 105.  12.   6.   5.   1.
   4.  60. 107.  18.   0.   0.   0.  15.  37.  37.  30.  15.  21.   2.
  12.  16.   3.   0.   0.  23.  40.   7.   8.   0.   0.   2.  57. 121.
  14.   2.   1.   0.   0.  25.  70.  48.   0.   0.   2.   0.   1.   4.
  28.  34.]
[sift] IndexFlatL2: ntotal=1,000,000, d=128, build time=0.17 s, stored vectors=512 MB
  0: [-0.02701984  0.11539876  0.02164138 -0.05349847  0.0081553  -0.13311078
  0.00997753 -0.12772731  0.04484392 -0.14000343  0.00365088 -0.00347016
  0.19274133 -0.0091836   0.17964269  0.16808899 -0.04258849  0.07681806
  0.16111052  0.16021168  0.06209341  0.09982534 -0.08135276  0.0545022
 -0.12762241  0.2999767  -0.02182544  0.04700636 -0.00892968 -0.07953602
  0.07486065  0.08699372  0.16990334  0.03116115 -0.05877941 -0.12503797
 -0.00860472  0.13129164  0.02388226  0.11466681  0.16952427 -0.01274627
  0.05322667  0.07371148 -0.0951905   0.00873418 -0.08447365 -0.10203069
  0.11073054  0.06100146  0.1627365  -0.04964087  0.09163094  0.01329725
 -0.06053416 -0.04960034  0.12522155 -0.02717719 -0.07755716 -0.10515158
  0.04178979  0.14842671  0.11977372 -0.18136406 -0.01711503  0.00191046
 -0.03167613  0.11943993  0.04487968 -0.13046673 -0.10171837  0.10232872
  0.03546935 -0.00172886 -0.03930071 -0.01425688  0.23591395 -0.1471941
  0.01291483  0.11545121 -0.08364633 -0.06449904  0.09019089  0.02742514
 -0.09444425  0.05785435  0.09333323 -0.01794233  0.15519299 -0.04964326
 -0.04161813  0.12772255 -0.03942708  0.03238184 -0.13355185  0.00404451
  0.03044589  0.22428156 -0.05390139 -0.00503204]
  1: [-0.12779517 -0.30363294 -0.03123757 -0.1325016   0.05398661 -0.15170735
 -0.02111989 -0.21606454  0.0637705  -0.03499398 -0.00326048  0.04389076
  0.1992852   0.17372462  0.04274689 -0.00072762 -0.09824325  0.06366009
  0.15676555  0.10706172 -0.02571592 -0.07658801  0.04940727 -0.03015275
 -0.13768819  0.1230605   0.02074759 -0.1604244   0.21505034  0.09487198
  0.00659196 -0.05482109  0.02087083 -0.02153199  0.05153967  0.03830492
  0.12350727 -0.04651486 -0.06142755  0.10049376  0.01904398  0.17211987
  0.06140957 -0.05017499 -0.13313067 -0.03163683 -0.09736513  0.09845508
 -0.04796684  0.06770921  0.00238313 -0.06586053  0.12539446  0.07090717
  0.14871609  0.09463575  0.16760087 -0.05744133 -0.02049083  0.10913121
 -0.04376751 -0.0868713   0.02974579  0.20004265  0.15801084 -0.07986942
 -0.00421101  0.0154827  -0.03324929 -0.00353817 -0.06780293 -0.12006539
  0.07501791 -0.0318371   0.04838922  0.11108388  0.00893863  0.16812722
  0.03848208 -0.17525236 -0.04114341 -0.01764078  0.04278797 -0.24537383
 -0.07731336 -0.03070992  0.01625812 -0.06157262 -0.11494043  0.0796653
  0.05246787 -0.0109162   0.08233048 -0.10789619 -0.09541888 -0.02493921
  0.01267399  0.0419137  -0.08681224 -0.08048436]
  2: [-0.01278196  0.07506908  0.02545223  0.03167944 -0.13281444 -0.04144568
  0.19012932 -0.03892067 -0.06586256  0.07940629  0.07272352  0.18630165
  0.09307302  0.04509893  0.04168413  0.02794462 -0.01641889 -0.11812867
 -0.01313587 -0.11621483  0.11076821  0.113297    0.03280892 -0.13622798
  0.03342511  0.20810063  0.08066379 -0.08257386  0.01540863 -0.03929842
  0.02556894 -0.08573139 -0.17825723  0.02527779 -0.19104546  0.06299619
 -0.02143253 -0.02877039 -0.08072276  0.1361778   0.17705244  0.10713254
  0.09743783 -0.03064534  0.11477913  0.13123316 -0.16907078  0.07837094
 -0.17160584  0.06648251  0.1186633   0.09392263  0.06835871  0.12353261
 -0.18725543  0.00775414 -0.16878213  0.07122885 -0.03036798  0.01195079
 -0.01621057 -0.00949542 -0.12667759  0.19034268  0.0817142  -0.02107863
  0.24449502 -0.00408119  0.04288891  0.01669624 -0.12365183  0.21251816
 -0.07271599 -0.0135475   0.03222786  0.01385873  0.13415727  0.09253212
 -0.06789061  0.10309026 -0.01105925 -0.02702096 -0.05054428  0.02539701
 -0.08953147 -0.11423573 -0.04430202 -0.01822731 -0.06196333 -0.14970645
 -0.02188683 -0.17440444  0.04658482  0.19311617 -0.00622344 -0.03862825
 -0.08737917 -0.00536403  0.05934419  0.00855507]
  3: [-0.09574772 -0.06161184  0.17393948  0.12349547  0.05121102 -0.02027508
  0.10061233 -0.06346466 -0.03330095 -0.06686322 -0.0800916   0.11112946
  0.15978862 -0.09589668  0.0269141  -0.04292824  0.06052995  0.22076935
  0.1426586  -0.00974163 -0.01419269  0.00269311 -0.03742338  0.14170475
 -0.07131623  0.11639259  0.03130179  0.08709649 -0.02536313  0.19291186
  0.0946384  -0.0548513   0.14775449 -0.02700688  0.09704261 -0.04292694
 -0.06425256 -0.03950224 -0.23194109  0.03763898  0.04316997 -0.07092033
  0.0728633  -0.02017839  0.11718311 -0.08803073 -0.02719764 -0.10166287
  0.13185272  0.11895752 -0.05011997  0.01725936 -0.09016578  0.25323927
 -0.15252371 -0.06446946  0.08617139 -0.11197616 -0.16046806  0.01166892
 -0.08847369  0.01532162 -0.08844624  0.12638967 -0.17174433 -0.11580461
  0.15686175  0.07577971  0.11621228 -0.0745175  -0.24225046  0.05167226
  0.07722484  0.08005372  0.09073155 -0.16676605  0.02262964 -0.06319418
 -0.01248361 -0.0258701  -0.02656654 -0.00805084 -0.1853203   0.00909811
  0.00249019 -0.07886337 -0.01649237 -0.03808977  0.07713208  0.05408431
  0.06138057 -0.09248114  0.04644048  0.13274123  0.05093139  0.07458021
  0.0253971   0.1163743  -0.10560369  0.12038567]
  4: [-0.03532146  0.10536979 -0.08331273  0.02858244 -0.08905803 -0.08540283
 -0.08278441 -0.16949396 -0.05949347 -0.23176652  0.07217157  0.05757064
  0.21033548  0.0820474  -0.09526541 -0.0344503  -0.06177736  0.10131712
  0.05188331  0.02947512  0.08636508 -0.085007   -0.05124071  0.06595922
  0.02717137  0.17408161 -0.02718296 -0.14858629 -0.02363873 -0.13928184
  0.01220906  0.01511451  0.03616114 -0.1550007   0.0049871  -0.07183868
  0.13472733  0.06791186  0.09089805 -0.01922003  0.221581    0.20861307
 -0.00449737 -0.00563335 -0.02656852 -0.0041605   0.06173926  0.06560314
  0.2025183   0.15107718 -0.03762521  0.1772449   0.00388988 -0.08269165
  0.03597068 -0.10652251  0.00664925 -0.02083315  0.00550731  0.01695437
 -0.18075603 -0.04421848  0.07734218  0.08786393 -0.03772789  0.06795493
  0.14230272 -0.04638974 -0.0940448   0.08999874 -0.00088425 -0.07110167
 -0.05707544  0.07304272  0.16351183 -0.00513616  0.31513888 -0.1805076
  0.03597233  0.06036794 -0.09407628 -0.16838433 -0.01589855  0.11662689
 -0.06500526 -0.07939751  0.03950663 -0.10013957 -0.09288879  0.09023889
 -0.04330593 -0.11095115 -0.09621441  0.00665439  0.05514599  0.06704568
  0.05708207  0.11278785 -0.00634567  0.01541941]
[glove] IndexFlatIP: ntotal=1,183,514, d=100, build time=0.15 s, stored vectors=473 MB
~~~~

Execution count recorded by the notebook: `2`.

## Cell 6 > markdown

### Prompt

Create notebook cell 6 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## 3. Recall@10 that handles ties correctly

**Recall@k** for one query is the fraction of the true k nearest neighbours that the index returned. The obvious implementation, `len(set(returned) & set(true_ids)) / k`, has a flaw: **real datasets contain duplicate and equidistant points** (SIFT descriptors are integer valued, so exact ties are common). If points 9 and 10 are both at distance 232.87 from the query, then which of them lands in the "true" top 10 is arbitrary, and an index that returns the other one has made no mistake.

The tie-aware definition used here is:

> A returned point counts as correct if its **true distance** is less than or equal to the distance of the k-th exact neighbour.

So any point tied with the 10th neighbour is accepted. Design details:

- **True distances are recomputed in float64** from the raw vectors for both the returned ids and the ground-truth ids. We never trust distances reported by the index under test, because IVF-PQ returns only approximate (compressed) distances, and HNSW/IVF float32 values can differ slightly from the file.
- **A tiny tolerance** (`rtol=1e-6`) absorbs float32 rounding so that points that are truly tied are not separated by noise in the last bit.
- **Duplicate ids in a result list are counted once**, otherwise an index could inflate recall by returning the same neighbour repeatedly.
- **Missing results (`-1`)** returned when an index finds fewer than k candidates count as misses.
~~~~

## Cell 7 > code

### Prompt

Implement notebook code cell 7 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell defines the tie-aware recall@k. exact_dists recomputes true float64 distances for a set of ids, recall_from_dists contains the counting logic (separated out so it can be unit tested with hand-made distances), and recall_at_k is the wrapper we will call for every index in the project.
"""

def exact_dists(ids, base, queries, metric, chunk=500):
    """True distance (float64) from each query to each id; id -1 gives inf.
    l2 -> squared L2 (same ranking as L2); angular -> 1 - cosine
    (base and queries must already be L2-normalised for angular)."""
    ids = np.asarray(ids)
    nq, k = ids.shape
    out = np.full((nq, k), np.inf)
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        sub = ids[s:e]
        valid = sub >= 0
        vecs = base[np.where(valid, sub, 0)].astype(np.float64)   # (chunk, k, d)
        q = queries[s:e].astype(np.float64)[:, None, :]
        if metric == "l2":
            dist = ((vecs - q) ** 2).sum(-1)
        else:
            dist = 1.0 - (vecs * q).sum(-1)
        out[s:e] = np.where(valid, dist, np.inf)
    return out


def recall_from_dists(ret_ids, ret_d, gt_d, k=10, rtol=1e-6, atol=1e-7, per_query=False):
    """Core logic. ret_d / gt_d are the TRUE distances of the retrieved / exact top-k."""
    ret_ids, ret_d, gt_d = ret_ids[:, :k], ret_d[:, :k], gt_d[:, :k]

    # threshold = distance of the k-th exact neighbour (max over the top-k is robust
    # to float32 ordering noise), plus a tiny tolerance so true ties are accepted
    thr = gt_d.max(axis=1, keepdims=True)
    thr = thr + rtol * np.abs(thr) + atol

    hit = (ret_d <= thr) & (ret_ids >= 0)

    # a duplicated id in the result list must only be counted once
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
    """I: (nq, >=k) ids returned by the index under test; gt_I: exact ids (>=k columns)."""
    return recall_from_dists(
        I, exact_dists(I[:, :k], base, queries, metric),
        exact_dists(gt_I[:, :k], base, queries, metric), k=k, **kw)

print("recall_at_k defined (tie-aware, float64 distances, duplicates counted once)")
~~~~

### Captured output

~~~~text
recall_at_k defined (tie-aware, float64 distances, duplicates counted once)
~~~~

Execution count recorded by the notebook: `3`.

## Cell 8 > markdown

### Prompt

Create notebook cell 8 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
### Unit tests for recall@k

Each test targets one rule from above using tiny hand-made inputs, so the expected answer is obvious. Tests 1 to 6 exercise `recall_from_dists` directly; tests 7 and 8 run the full `recall_at_k` wrapper on real vectors containing duplicates (L2) and parallel vectors (angular).
~~~~

## Cell 9 > code

### Prompt

Implement notebook code cell 9 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell contains the unit tests for recall@k. If every assertion passes the final line is printed; any failure raises an AssertionError that names the failing test.
"""

def test_recall():
    k = 10
    gt_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]])   # exact top-10 distances

    # 1. perfect result -> recall 1
    ids = np.arange(10)[None, :]
    assert recall_from_dists(ids, gt_d, gt_d, k) == 1.0, "test 1 (perfect) failed"
    print("test 1 passed: perfect result gives recall 1.0")

    # 2. tie at the 10th neighbour: id 10 is at the same distance (10.0) as id 9,
    #    so returning 10 instead of 9 must NOT be a mistake
    ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]])
    d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]])
    assert recall_from_dists(ids, d, gt_d, k) == 1.0, "test 2 (tie) failed"
    print("test 2 passed: a point tied with the 10th neighbour is not a mistake")

    # 3. a genuinely farther point IS a mistake
    d_far = d.copy(); d_far[0, 9] = 10.5
    assert np.isclose(recall_from_dists(ids, d_far, gt_d, k), 0.9), "test 3 (farther point) failed"
    print("test 3 passed: a strictly farther point costs 1/10 recall")

    # 4. duplicate ids are counted once
    ids = np.array([[0, 0, 2, 3, 4, 5, 6, 7, 8, 9]])
    d = np.array([[1, 1, 3, 4, 5, 6, 7, 8, 9, 10.0]])
    assert np.isclose(recall_from_dists(ids, d, gt_d, k), 0.9), "test 4 (duplicates) failed"
    print("test 4 passed: a repeated id is counted once")

    # 5. -1 padding (index returned fewer than k results) is a miss
    ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, -1, -1]])
    d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, np.inf, np.inf]])
    assert np.isclose(recall_from_dists(ids, d, gt_d, k), 0.8), "test 5 (-1 padding) failed"
    print("test 5 passed: missing results (-1) count as misses")

    # 6. per-query output: one perfect query and one completely wrong query
    ids2 = np.vstack([np.arange(10), np.arange(10) + 100])
    d_r = np.vstack([gt_d[0], gt_d[0] + 50])
    gt2 = np.vstack([gt_d[0], gt_d[0]])
    pq = recall_from_dists(ids2, d_r, gt2, k, per_query=True)
    assert np.allclose(pq, [1.0, 0.0]), "test 6 (per query) failed"
    assert np.isclose(recall_from_dists(ids2, d_r, gt2, k), 0.5), "test 6 (mean) failed"
    print("test 6 passed: per-query recalls [1.0, 0.0] average to 0.5")

    # 7. end-to-end wrapper with real duplicate vectors (L2, k=3); ids 2 and 3 are identical points
    base = np.array([[0.], [1.], [2.], [2.], [3.]], dtype=np.float32)
    qry = np.array([[0.]], dtype=np.float32)
    gt = np.array([[0, 1, 2]])
    assert recall_at_k(np.array([[0, 1, 3]]), gt, base, qry, "l2", k=3) == 1.0, "test 7a failed"
    assert np.isclose(recall_at_k(np.array([[0, 1, 4]]), gt, base, qry, "l2", k=3), 2 / 3), "test 7b failed"
    print("test 7 passed: duplicate vectors are interchangeable, a farther vector is not (L2)")

    # 8. angular: parallel vectors have the same cosine, so they tie
    base = np.array([[1, 0], [2, 0], [0, 1]], dtype=np.float32)
    faiss.normalize_L2(base)
    qry = np.array([[1, 0]], dtype=np.float32)
    assert recall_at_k(np.array([[1]]), np.array([[0]]), base, qry, "angular", k=1) == 1.0, "test 8a failed"
    assert recall_at_k(np.array([[2]]), np.array([[0]]), base, qry, "angular", k=1) == 0.0, "test 8b failed"
    print("test 8 passed: parallel vectors tie, an orthogonal one does not (angular)")

    print("\nall recall tests passed")

test_recall()
~~~~

### Captured output

~~~~text
test 1 passed: perfect result gives recall 1.0
test 2 passed: a point tied with the 10th neighbour is not a mistake
test 3 passed: a strictly farther point costs 1/10 recall
test 4 passed: a repeated id is counted once
test 5 passed: missing results (-1) count as misses
test 6 passed: per-query recalls [1.0, 0.0] average to 0.5
test 7 passed: duplicate vectors are interchangeable, a farther vector is not (L2)
test 8 passed: parallel vectors tie, an orthogonal one does not (angular)

all recall tests passed
~~~~

Execution count recorded by the notebook: `4`.

## Cell 10 > markdown

### Prompt

Create notebook cell 10 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## 4. Does FAISS exact agree with the neighbours stored in the data files>

Before trusting FAISS as the reference we check it against the ground truth shipped with each dataset. We run all test queries through the flat index asking for 100 neighbours, then compare with `neighbors` and `distances` from the file in four ways:

1. **Rank-exact id match**: the same id at the same position. Expected to be slightly below 100%, because tied points can appear in a different order.
2. **Set overlap@10**: the same 10 ids regardless of order. Also affected by ties at rank 10.
3. **Tie-aware recall@10** in both directions (FAISS against file, and file against FAISS). This is the number that matters and should be 1.0.
4. **Maximum distance difference** after converting FAISS distances to the file's units. This should be tiny, only float32 rounding.

We also count how many queries have a tie across the 10th/11th neighbour in the file, to show that the tie handling is not an academic detail.
~~~~

## Cell 11 > code

### Prompt

Implement notebook code cell 11 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell runs all test queries through the flat index in one batch (k=100) and stores the result as our reference. The reference lists are reused later by the HNSW, IVF-Flat and IVF-PQ notebooks.
"""

exact = {}
for name in DATASETS:
    d = data[name]
    t0 = time.perf_counter()
    D, I = indexes[name].search(d["test"], K_REF)
    elapsed = time.perf_counter() - t0
    exact[name] = dict(D=D, I=I)
    print(f"[{name}] searched {len(d['test']):,} queries with k={K_REF} in {elapsed:.1f} s "
          f"| ids shape {I.shape} | distances shape {D.shape}")
    print(f"[{name}] query 0 top-{K} ids:       {I[0, :K]}")
    print(f"[{name}] query 0 top-{K} distances: {np.round(to_file_units(D[0, :K], d['metric']), 4)}")
~~~~

### Captured output

~~~~text
[sift] searched 10,000 queries with k=100 in 11.1 s | ids shape (10000, 100) | distances shape (10000, 100)
[sift] query 0 top-10 ids:       [932085 934876 561813 708177 706771 695756 435345 701258 455537 872728]
[sift] query 0 top-10 distances: [232.8712 234.7147 243.9898 255.4604 256.3143 258.8629 261.2413 264.2802
 267.2845 268.069 ]
[glove] searched 10,000 queries with k=100 in 6.0 s | ids shape (10000, 100) | distances shape (10000, 100)
[glove] query 0 top-10 ids:       [  97478  262700  846101  671078  232287  727732  544474 1133489  723915
  660281]
[glove] query 0 top-10 distances: [0.4304 0.4324 0.4331 0.4334 0.4376 0.4398 0.4496 0.4501 0.452  0.4522]
~~~~

Execution count recorded by the notebook: `5`.

## Cell 12 > code

### Prompt

Implement notebook code cell 12 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell compares the FAISS exact results with the neighbours and distances stored in the hdf5 files, using strict rank-by-rank matching, set overlap, and the tie-aware recall@10 in both directions. It prints a per-dataset report and a summary table.
"""

summary = []
for name in DATASETS:
    d = data[name]
    D, I = exact[name]["D"], exact[name]["I"]
    train, test, gt_ids, gt_dist, metric = d["train"], d["test"], d["gt_ids"], d["gt_dist"], d["metric"]

    strict = (I[:, :K] == gt_ids[:, :K]).mean()
    overlap_pq = np.array([len(set(a) & set(b)) / K for a, b in zip(I[:, :K], gt_ids[:, :K])])

    recall_pq = recall_at_k(I, gt_ids, train, test, metric, k=K, per_query=True)   # faiss vs file
    recall_fa_vs_file = float(recall_pq.mean())
    recall_file_vs_fa = recall_at_k(gt_ids, I, train, test, metric, k=K)           # file vs faiss

    dist_err = np.abs(to_file_units(D[:, :K], metric) - gt_dist[:, :K]).max()

    # how many queries have a tie across the 10th / 11th true neighbour in the file>
    d11 = exact_dists(gt_ids[:, :K + 1], train, test, metric)
    tie_at_10 = int((d11[:, K] <= d11[:, K - 1] * (1 + 1e-6) + 1e-7).sum())

    set_mismatch = overlap_pq < 1.0
    explained = int((set_mismatch & (recall_pq == 1.0)).sum())

    print(f"[{name}] rank-exact id match:            {strict:.4f}")
    print(f"[{name}] set overlap@{K}:                  {overlap_pq.mean():.4f}")
    print(f"[{name}] tie-aware recall@{K} faiss->file: {recall_fa_vs_file:.4f}")
    print(f"[{name}] tie-aware recall@{K} file->faiss: {recall_file_vs_fa:.4f}")
    print(f"[{name}] max |distance diff| vs file:    {dist_err:.2e}")
    print(f"[{name}] queries with a tie at rank {K}/{K + 1} in the file: {tie_at_10} of {len(test)}")
    print(f"[{name}] queries whose top-{K} id set differs from the file: {int(set_mismatch.sum())}, "
          f"of which explained by ties (tie-aware recall = 1.0): {explained}")
    status = "PASS" if min(recall_fa_vs_file, recall_file_vs_fa) >= 0.9999 else "CHECK"
    print(f"[{name}] agreement with data file: {status}\n")

    summary.append({
        "dataset": name,
        "rank-exact match": round(float(strict), 4),
        "set overlap@10": round(float(overlap_pq.mean()), 4),
        "recall@10 faiss->file": round(recall_fa_vs_file, 4),
        "recall@10 file->faiss": round(recall_file_vs_fa, 4),
        "max dist diff": float(f"{dist_err:.2e}"),
        "queries w/ tie at rank 10": tie_at_10,
    })

agreement_df = pd.DataFrame(summary)
print(agreement_df.to_string(index=False))
~~~~

### Captured output

~~~~text
[sift] rank-exact id match:            0.9857
[sift] set overlap@10:                  0.9993
[sift] tie-aware recall@10 faiss->file: 1.0000
[sift] tie-aware recall@10 file->faiss: 1.0000
[sift] max |distance diff| vs file:    0.00e+00
[sift] queries with a tie at rank 10/11 in the file: 137 of 10000
[sift] queries whose top-10 id set differs from the file: 65, of which explained by ties (tie-aware recall = 1.0): 65
[sift] agreement with data file: PASS

[glove] rank-exact id match:            1.0000
[glove] set overlap@10:                  1.0000
[glove] tie-aware recall@10 faiss->file: 1.0000
[glove] tie-aware recall@10 file->faiss: 1.0000
[glove] max |distance diff| vs file:    5.29e-07
[glove] queries with a tie at rank 10/11 in the file: 2 of 10000
[glove] queries whose top-10 id set differs from the file: 0, of which explained by ties (tie-aware recall = 1.0): 0
[glove] agreement with data file: PASS

dataset  rank-exact match  set overlap@10  recall@10 faiss->file  recall@10 file->faiss  max dist diff  queries w/ tie at rank 10
   sift            0.9857          0.9993                    1.0                    1.0   0.000000e+00                        137
  glove            1.0000          1.0000                    1.0                    1.0   5.290000e-07                          2
~~~~

Execution count recorded by the notebook: `6`.

## Cell 13 > markdown

### Prompt

Create notebook cell 13 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
**How to read this.** A tie-aware recall of 1.0 in both directions means FAISS and the file agree on every query once equally-good neighbours are treated as interchangeable, so FAISS exact is a valid reference. Any gap between the rank-exact match / set overlap and 1.0 is explained by ties and shows up in the "explained by ties" count: those queries differ only in which of several equidistant points was listed.

If a dataset shows a large distance difference but a recall of 1.0, the file simply uses a different distance convention for the stored `distances` (for example cosine similarity instead of `1 - cos`). Recall is unaffected because `recall_at_k` recomputes distances itself.
~~~~

## Cell 14 > markdown

### Prompt

Create notebook cell 14 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## 5. Timing exact search: one query at a time vs one batch

Two ways of running the same exact search:

- **Single-query mode**: call `index.search(query, k)` once per query inside a Python loop. This models an online service that answers requests as they arrive.
- **Batch mode**: call `index.search(all_queries, k)` once. This models offline evaluation, and it is what we used above to build the reference.

Both return identical results, but the cost per query is very different. To see where the gain comes from we also sweep the batch size (1, 10, 100, 1000, all) and add one extra run restricted to a **single CPU thread**, which separates the effect of multi-threading from the effect of batching the computation itself.

Methodology: one warm-up call first (so one-off start-up costs are not timed), `time.perf_counter()` for timing, per-query cost reported in milliseconds, and the best of three repeats for the full batch. Single-query mode uses the first 1000 queries to keep the runtime reasonable (a single query scans the whole 1M database); per-query cost is what we compare, so this does not bias the result.
~~~~

## Cell 15 > code

### Prompt

Implement notebook code cell 15 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell times exact FAISS search for different batch sizes. Batch size 1 is the single-query mode and batch size equal to the number of queries is the full-batch mode. It also times a batch of 1000 restricted to one OpenMP thread to separate the effect of parallelism from the effect of batching. Results are printed as a table per dataset.
"""

N_SINGLE = 1000   # number of queries used for the small-batch measurements

def time_search(index, queries, batch_size, n_queries, k=K, reps=1):
    """Best-of-reps average time per query (ms) when searching n_queries in chunks of batch_size."""
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for s in range(0, n_queries, batch_size):
            index.search(queries[s:s + batch_size], k)
        best = min(best, time.perf_counter() - t0)
    return best / n_queries * 1e3

max_threads = faiss.omp_get_max_threads()
print("OpenMP threads used by FAISS:", max_threads)

timing = {}
for name in DATASETS:
    queries, index = data[name]["test"], indexes[name]
    nq = len(queries)
    index.search(queries[:20], K)   # warm-up

    batch_sizes = [1, 10, 100, 1000, nq]
    rows = []
    for b in tqdm.tqdm(batch_sizes, desc=name):
        n_q = min(nq, max(b, N_SINGLE))
        ms = time_search(index, queries, b, n_q, reps=3 if b == nq else 1)
        rows.append({"mode": "single query" if b == 1 else ("full batch" if b == nq else "mini batch"),
                     "batch size": b, "threads": max_threads, "queries timed": n_q, "ms / query": ms})

    # same batch of 1000, but restricted to one thread
    faiss.omp_set_num_threads(1)
    ms_1t = time_search(index, queries, 1000, min(nq, 1000))
    faiss.omp_set_num_threads(max_threads)
    rows.append({"mode": "mini batch, 1 thread", "batch size": 1000, "threads": 1,
                 "queries timed": min(nq, 1000), "ms / query": ms_1t})

    df = pd.DataFrame(rows)
    df["QPS"] = 1e3 / df["ms / query"]
    df["speedup vs single"] = df["ms / query"].iloc[0] / df["ms / query"]
    timing[name] = df

    single_ms = df.loc[df["batch size"] == 1, "ms / query"].iloc[0]
    batch_ms = df.loc[df["mode"] == "full batch", "ms / query"].iloc[0]
    print(f"\n[{name}] single query: {single_ms:.2f} ms/query ({1e3 / single_ms:,.0f} QPS) | "
          f"full batch: {batch_ms:.4f} ms/query ({1e3 / batch_ms:,.0f} QPS) | "
          f"batch is {single_ms / batch_ms:,.0f}x faster per query")
    print(df.round({"ms / query": 4, "QPS": 0, "speedup vs single": 1}).to_string(index=False))
~~~~

### Captured output

~~~~text
OpenMP threads used by FAISS: 20

sift: 100%|██████████| 5/5 [01:11<00:00, 14.24s/it]


[sift] single query: 23.38 ms/query (43 QPS) | full batch: 1.1971 ms/query (835 QPS) | batch is 20x faster per query
                mode  batch size  threads  queries timed  ms / query   QPS  speedup vs single
        single query           1       20           1000     23.3794  43.0                1.0
          mini batch          10       20           1000      2.9847 335.0                7.8
          mini batch         100       20           1000      6.6095 151.0                3.5
          mini batch        1000       20           1000      1.3961 716.0               16.7
          full batch       10000       20          10000      1.1971 835.0               19.5
mini batch, 1 thread        1000        1           1000      2.5159 397.0                9.3

glove: 100%|██████████| 5/5 [00:57<00:00, 11.45s/it]


[glove] single query: 13.87 ms/query (72 QPS) | full batch: 0.9572 ms/query (1,045 QPS) | batch is 14x faster per query
                mode  batch size  threads  queries timed  ms / query    QPS  speedup vs single
        single query           1       20           1000     13.8728   72.0                1.0
          mini batch          10       20           1000      1.6085  622.0                8.6
          mini batch         100       20           1000      5.3544  187.0                2.6
          mini batch        1000       20           1000      6.0451  165.0                2.3
          full batch       10000       20          10000      0.9572 1045.0               14.5
mini batch, 1 thread        1000        1           1000     43.2488   23.0                0.3
~~~~

Execution count recorded by the notebook: `7`.

## Cell 16 > markdown

### Prompt

Create notebook cell 16 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
### Why single-query and batch timings differ so much

Searching one query at a time scans the entire database (about 512 MB for SIFT 1M) for every single query. FAISS only parallelises and uses its fast matrix path when it has several queries to work on, so a lone query runs essentially on one core and is limited by **memory bandwidth**: the whole dataset streams through the CPU to produce 10 numbers, and nothing is reused. With all queries in one batch, FAISS turns the distance computation into a large **matrix multiplication (BLAS GEMM)**. Each block of database vectors is loaded into cache once and reused against many queries at once, and the work is spread over all CPU cores, so arithmetic throughput rises sharply and the cost per query drops by one to two orders of magnitude.

The sweep above shows both ingredients: per-query cost falls steadily as the batch size grows (data reuse), and the one-thread row isolates how much of the gain remains without parallelism. **For all later comparisons we must therefore report QPS for one clearly stated mode and thread count**, and use the same mode for every algorithm; mixing a single-query HNSW number with this batch baseline would be misleading.
~~~~

## Cell 17 > markdown

### Prompt

Create notebook cell 17 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## 6. Saving the baseline

The HNSW, IVF-Flat and IVF-PQ notebooks all need the exact top-100 neighbours (for recall) and the baseline timing (for speedup). We save both so they do not have to re-run the exact search.
~~~~

## Cell 18 > code

### Prompt

Implement notebook code cell 18 of 19. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell saves the FAISS exact reference (top-100 ids and distances per query) and the baseline timing table for each dataset into ./data/processed. Later notebooks can load these with np.load and pd.read_csv instead of recomputing them.
"""

OUT_DIR = "./data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

for name in DATASETS:
    npz_path = f"{OUT_DIR}/faiss_exact_{name}.npz"
    csv_path = f"{OUT_DIR}/faiss_exact_timing_{name}.csv"
    np.savez_compressed(npz_path, ids=exact[name]["I"], dists=exact[name]["D"])
    timing[name].to_csv(csv_path, index=False)
    print(f"[{name}] saved reference neighbours -> {npz_path}")
    print(f"[{name}] saved baseline timing      -> {csv_path}")

with open(f"{OUT_DIR}/faiss_exact_meta.json", "w") as f:
    json.dump({"k": K, "k_ref": K_REF, "faiss_version": faiss.__version__,
               "omp_threads": faiss.omp_get_max_threads(),
               "metrics": {n: DATASETS[n]["metric"] for n in DATASETS}}, f, indent=2)
print("saved run metadata ->", f"{OUT_DIR}/faiss_exact_meta.json")
~~~~

### Captured output

~~~~text
[sift] saved reference neighbours -> ./data/processed/faiss_exact_sift.npz
[sift] saved baseline timing      -> ./data/processed/faiss_exact_timing_sift.csv
[glove] saved reference neighbours -> ./data/processed/faiss_exact_glove.npz
[glove] saved baseline timing      -> ./data/processed/faiss_exact_timing_glove.csv
saved run metadata -> ./data/processed/faiss_exact_meta.json
~~~~

Execution count recorded by the notebook: `8`.

## Cell 19 > markdown

### Prompt

Create notebook cell 19 of 19 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Conclusion

FAISS exact search reproduces the ground-truth neighbours stored in both data files: once ties at the 10th neighbour are handled, tie-aware recall@10 is 1.0 in both directions, and the remaining differences in rank order are explained by equidistant points. It is therefore our **reference for correctness (recall = 1.0 by definition)** and our **reference for speed of an exact method**. It is far faster than the pure-Python brute-force indexer from the dataset notebooks, but it still compares every query with all 1M+ vectors, so its cost grows linearly with the dataset.

We also established the rules for the rest of the project:
- Recall@10 is always computed with `recall_at_k`, which recomputes true distances in float64, accepts ties, counts duplicates once and treats missing results as misses.
- QPS must be reported for a stated query mode (single query or batch) and thread count, because the two differ by one to two orders of magnitude for exact search, and the same effect applies to the approximate indexes.

The next notebooks (HNSW, IVF-Flat and IVF-PQ) will measure how much speed can be gained over this baseline, and how much recall has to be given up in exchange.
~~~~
