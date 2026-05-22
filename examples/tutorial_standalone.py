"""Minimal end-to-end example -- drop this into a Jupyter cell or run as a script.

Demonstrates the standalone FastCAR pipeline on a real raw 10x dataset
(bundled in ``data/pbmc1k_raw.h5ad``) that still contains empty droplets,
which FastCAR needs in order to estimate the ambient-RNA background.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

import pyfastcar as fc


DATA = Path(__file__).resolve().parents[1] / "data" / "pbmc1k_raw.h5ad"


def main() -> None:
    # Raw, unfiltered AnnData (cells x genes): real cells + empty droplets.
    raw = ad.read_h5ad(DATA)
    tot = np.asarray(raw.X.sum(1)).ravel() if sp.issparse(raw.X) else raw.X.sum(1)

    # The filtered "cell" AnnData is just the real cells (here total UMI >= 1000).
    cells = raw[tot >= 1000].copy()
    print(f"raw droplets: {raw.n_obs}   real cells: {cells.n_obs}")

    # One-shot AnnData correction (transposes internally to genes x droplets).
    corrected = fc.correct_anndata(
        raw, cells,
        empty_droplet_cutoff=100,
        contamination_chance_cutoff=0.005,
    )

    info = corrected.uns["fastcar"]
    print(f"genes flagged for correction: {info['n_genes_corrected']}")
    profile = info["ambient_rna_profile"]
    print("top ambient genes (gMax subtracted):")
    print(profile.sort_values(ascending=False).head(10))

    before = corrected.layers  # noqa: F841 -- placeholder for layer demos
    removed = int(np.asarray(raw[tot >= 1000].X.sum())) - int(
        np.asarray(corrected.X.sum()))
    print(f"total counts removed as ambient RNA: {removed}")

    # Choosing the empty-droplet cutoff from the raw matrix.
    full_gd = sp.csc_matrix(raw.X).T  # genes x droplets
    import pandas as pd
    full_df = pd.DataFrame.sparse.from_spmatrix(
        full_gd, index=pd.Index(raw.var_names))
    desc = fc.describe_ambient_rna_sequence(
        full_df, start=50, stop=500, by=25, contamination_chance_cutoff=0.005)
    print("\nempty-droplet cutoff sweep:")
    print(desc)
    print("recommended cutoff:", fc.recommend_empty_cutoff(desc))


if __name__ == "__main__":
    main()
