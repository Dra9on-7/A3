# Hypothesis for Part 3: how does the cost of reaching recall 0.95 grow with database size?

Setup: SIFT, 100K / 300K / 1M vectors, HNSW (M=16, efConstruction=100), IVF-Flat (nlist about sqrt(N)) and exact search.

**Hypothesis (in short).** Exact search has to look at every vector, so its cost grows in step with N. IVF-Flat
still scans a fixed share of the vectors, so it grows almost as fast, just from a lower start. HNSW only follows a short path
through a graph, so its cost grows slowest. Therefore the advantage over exact search should get bigger as N grows, most of all for HNSW.

**Wrong if:** IVF-Flat grows as slowly as HNSW, or HNSW's cost grows about as fast as exact search.

*Note on honesty:* an earlier run of this notebook had already been seen when this file was written, so these
predictions come from theory and the Part 2 results, but they are not blind.
