"""Head-to-head speed benchmark: R FastCAR vs pyfastcar.

Runs both on the raw 10x dataset bundled in ``data/pbmc1k_raw.h5ad`` and
reports wall time for the FastCAR pipeline:

  * ``determine.background.to.remove`` — build the per-gene ambient profile
  * ``remove.background``             — apply the correction

FastCAR is fully deterministic, so the two implementations produce
bit-identical output; this benchmark only compares how fast they get there.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import pyfastcar as fc


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
DATA = HERE.parent / "data" / "pbmc1k_raw.h5ad"
RSCRIPT = "/scratch/users/steorra/env/CMAP/bin/Rscript"


R_BENCH_SCRIPT = r"""
suppressMessages({ library(Matrix); library(qlcMatrix); library(FastCAR) })
args <- commandArgs(trailingOnly = TRUE)
full_path <- args[[1]]; cell_path <- args[[2]]
fullDense <- as.matrix(read.table(full_path, sep = "\t", header = TRUE,
                                  row.names = 1, check.names = FALSE))
cellDense <- as.matrix(read.table(cell_path, sep = "\t", header = TRUE,
                                  row.names = 1, check.names = FALSE))
fullSparse <- as(as(fullDense, "CsparseMatrix"), "dgCMatrix")
cellSparse <- as(as(cellDense, "CsparseMatrix"), "dgCMatrix")
for (rep in 1:3) {
  t0 <- proc.time()[[3]]
  amb <- determine.background.to.remove(fullSparse, cellSparse, 100, 0.005)
  t1 <- proc.time()[[3]]
  corr <- remove.background(cellSparse, amb)
  t2 <- proc.time()[[3]]
  cat(sprintf("R_DETERMINE=%.4f\nR_REMOVE=%.4f\n", t1 - t0, t2 - t1))
}
"""


def _dump_tsvs() -> tuple[Path, Path]:
    WORK.mkdir(exist_ok=True)
    full_tsv, cell_tsv = WORK / "full_counts.tsv", WORK / "cell_counts.tsv"
    raw = ad.read_h5ad(DATA)
    tot = np.asarray(raw.X.sum(1)).ravel()
    cells = raw[tot >= 1000].copy()
    if not full_tsv.exists():
        fullX = raw.X.T.toarray() if sp.issparse(raw.X) else np.asarray(raw.X).T
        pd.DataFrame(fullX, index=raw.var_names,
                     columns=raw.obs_names).to_csv(full_tsv, sep="\t")
    if not cell_tsv.exists():
        cellX = cells.X.T.toarray() if sp.issparse(cells.X) else np.asarray(cells.X).T
        pd.DataFrame(cellX, index=cells.var_names,
                     columns=cells.obs_names).to_csv(cell_tsv, sep="\t")
    return full_tsv, cell_tsv


def time_r(full_tsv: Path, cell_tsv: Path) -> tuple[float, float]:
    script = WORK / "_bench_r.R"
    script.write_text(R_BENCH_SCRIPT)
    env = os.environ.copy()
    gcc = "/share/software/user/open/gcc/14.2.0"
    if os.path.isdir(gcc):
        env["PATH"] = f"{gcc}/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{gcc}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(
        [RSCRIPT, str(script), str(full_tsv), str(cell_tsv)],
        env=env, capture_output=True, text=True, check=True,
    )
    det, rem = [], []
    for line in proc.stdout.splitlines():
        if line.startswith("R_DETERMINE="):
            det.append(float(line.split("=")[1]))
        elif line.startswith("R_REMOVE="):
            rem.append(float(line.split("=")[1]))
    return float(np.mean(det)), float(np.mean(rem))


def time_python(full_tsv: Path, cell_tsv: Path, runs: int = 3) -> tuple[float, float]:
    full = pd.read_csv(full_tsv, sep="\t", index_col=0)
    cell = pd.read_csv(cell_tsv, sep="\t", index_col=0)
    det, rem = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        profile = fc.determine_background_to_remove(full, cell, 100, 0.005)
        t1 = time.perf_counter()
        fc.remove_background(cell, profile)
        t2 = time.perf_counter()
        det.append(t1 - t0)
        rem.append(t2 - t1)
    return float(np.mean(det)), float(np.mean(rem))


def main() -> None:
    full_tsv, cell_tsv = _dump_tsvs()
    print(f"Dataset: pbmc1k raw subset — {DATA.name}")

    print("\n[R] timing FastCAR (3 runs)…")
    r_det, r_rem = time_r(full_tsv, cell_tsv)
    print(f"  determine.background.to.remove: {r_det*1000:8.2f} ms")
    print(f"  remove.background:              {r_rem*1000:8.2f} ms")

    print("\n[py] timing pyfastcar (3 runs)…")
    py_det, py_rem = time_python(full_tsv, cell_tsv)
    print(f"  determine_background_to_remove: {py_det*1000:8.2f} ms")
    print(f"  remove_background:              {py_rem*1000:8.2f} ms")

    print("\nSpeed-ups (R time / Python time):")
    print(f"  determine background: {r_det/py_det:5.2f}x")
    print(f"  remove background:    {r_rem/py_rem:5.2f}x")
    print(f"  end-to-end:           {(r_det+r_rem)/(py_det+py_rem):5.2f}x")


if __name__ == "__main__":
    main()
