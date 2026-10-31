#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

notebooks=(
  "SIFT_analysis.ipynb"
  "glove_analysis.ipynb"
  "faiss_index_analysis.ipynb"
  "IVF-Flat.ipynb"
  "IVF-PQ.ipynb"
  "HNSW.ipynb"
  "SIFT_experiment.ipynb"
  "Geometry_experiment.ipynb"
)

for notebook in "${notebooks[@]}"; do
  echo "Executing ${notebook}..."
  jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "$notebook"
done

echo "All notebooks executed successfully."