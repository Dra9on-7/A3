# Hypothesis for Part 4: how does hard geometry (intrinsic dimension) affect HNSW and IVF-Flat?

## Hypothesis (what I expect to break, and why)

Nearest-neighbour search is easy only while the nearest neighbour is much closer than a typical point.
As m grows, distances concentrate and the contrast between the 10th neighbour and an average point shrinks.
**Expecting IVF-Flat to break first and hardest.** It prunes by cell: it scans the cells whose centroids are closest to the query. With low m the true neighbours sit in the few nearest cells, but with high m they spread over many cells, and the nearest centroids no longer identify them, so nprobe has to grow towards nlist and the search degenerates into
an exhaustive scan plus the cost of ranking the centroids. 
**HNSW should degrade more gradually**:
a greedy walk needs only some distance gradient to follow, so it needs a far larger efSearch as the contrast fades, but it still visits a small share of the database. Both converge towards the cost of exact search at m = 128.

## What would count as the hypothesis being wrong

- IVF-Flat staying cheap (< 20% work) at m = 128, or HNSW degrading as much as or more than IVF-Flat.
- No monotone trend in the cost with m (then m is not what drives the difficulty here).
- A noise-control difference as large as the differences between settings (then nothing can be attributed to m).
