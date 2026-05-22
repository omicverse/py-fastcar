"""Self-contained smoke tests for pyfastcar (no R required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import pyfastcar as fc


EMPTY_CUTOFF = 100   # cell column sums far exceed this; empties stay below.


def _toy():
    """A tiny deterministic genes x droplets matrix.

    3 genes x 6 droplets: droplets 0-1 are real cells (total UMIs well above
    EMPTY_CUTOFF), droplets 2-4 are empty (low counts), droplet 5 is an
    all-zero unused barcode.  GENE0 contaminates every empty droplet; GENE1
    only one; GENE2 none.
    """
    genes = pd.Index(["GENE0", "GENE1", "GENE2"], name="gene")
    cols = pd.Index([f"d{i}" for i in range(6)])
    data = np.array(
        [
            [80, 95,  3,  2,  4,  0],   # GENE0 -- ambient, in all 3 empties
            [70, 88,  0,  1,  0,  0],   # GENE1 -- in 1 of 3 empties
            [60, 90,  0,  0,  0,  0],   # GENE2 -- never in empties
        ],
        dtype=np.int64,
    )
    return pd.DataFrame(data, index=genes, columns=cols)


def test_determine_background_basic():
    full = _toy()
    profile = fc.determine_background_to_remove(
        full, full.iloc[:, :2], empty_droplet_cutoff=EMPTY_CUTOFF,
        contamination_chance_cutoff=0.005)
    # 3 empty droplets all non-zero -> nEmpty = 3.
    # GENE0 in 3/3 -> frC 1.0 > 0.005, gMax = 4 -> corrected.
    assert profile["GENE0"] == 4
    # GENE1 in 1/3 -> frC 0.333 > 0.005, gMax = 1 -> corrected.
    assert profile["GENE1"] == 1
    # GENE2 in 0/3 -> frC 0 -> not corrected.
    assert profile["GENE2"] == 0


def test_remove_background_basic():
    full = _toy()
    cells = full.iloc[:, :2].copy()
    profile = fc.determine_background_to_remove(
        full, cells, empty_droplet_cutoff=EMPTY_CUTOFF,
        contamination_chance_cutoff=0.005)
    corrected = fc.remove_background(cells, profile)
    # GENE0: 80-4, 95-4 ; GENE1: 70-1, 88-1 ; GENE2: unchanged.
    np.testing.assert_array_equal(
        corrected.to_numpy(),
        np.array([[76, 91], [69, 87], [60, 90]]),
    )


def test_high_contamination_floors_at_zero():
    genes = pd.Index(["G"], name="gene")
    # one big real cell (>cutoff) + two low empty droplets.
    full = pd.DataFrame([[200, 2, 3]], index=genes,
                        columns=["c", "e1", "e2"])
    profile = fc.determine_background_to_remove(
        full, full.iloc[:, :1], empty_droplet_cutoff=EMPTY_CUTOFF,
        contamination_chance_cutoff=0.005)
    assert profile["G"] == 3            # gMax over empties
    corrected = fc.remove_background(
        pd.DataFrame([[1]], index=genes, columns=["c"]), profile)
    assert corrected.iloc[0, 0] == 0    # 1 - 3 floored at 0


def test_sparse_roundtrip():
    full = _toy()
    full_sp = sp.csc_matrix(full.to_numpy())
    profile = fc.determine_background_to_remove(
        pd.DataFrame.sparse.from_spmatrix(full_sp, index=full.index),
        None, empty_droplet_cutoff=EMPTY_CUTOFF,
        contamination_chance_cutoff=0.005)
    cells_sp = sp.csr_matrix(full.iloc[:, :2].to_numpy())
    corrected = fc.remove_background(
        pd.DataFrame.sparse.from_spmatrix(
            cells_sp, index=full.index, columns=full.columns[:2]),
        profile)
    assert isinstance(corrected, pd.DataFrame)
    assert (corrected.to_numpy() >= 0).all()


def test_unused_barcodes_excluded_from_nempty():
    # droplet d5 is an all-zero unused barcode -- must not count as empty.
    full = _toy()
    _, table = fc.determine_background_to_remove(
        full, None, empty_droplet_cutoff=EMPTY_CUTOFF,
        contamination_chance_cutoff=0.005, return_table=True)
    # nEmpty = 3 (d2,d3,d4); GENE0 occurs in all 3 -> frC == 1.0.
    assert abs(table.loc["GENE0", "frC"] - 1.0) < 1e-12


def test_describe_and_recommend():
    full = _toy()
    desc = fc.describe_ambient_rna_sequence(
        full, start=50, stop=150, by=50, contamination_chance_cutoff=0.005)
    assert list(desc.index) == [50, 100, 150]
    assert set(desc.columns) == {
        "nEmptyDroplets", "genesInBackground", "genesContaminating"}
    assert isinstance(fc.recommend_empty_cutoff(desc), int)


def test_anndata_wrapper():
    import anndata as ad

    full = _toy()
    # AnnData is cells x genes -> transpose the genes x droplets toy matrix.
    full_ad = ad.AnnData(
        X=sp.csr_matrix(full.to_numpy().T.astype(np.float64)),
        obs=pd.DataFrame(index=full.columns),
        var=pd.DataFrame(index=full.index),
    )
    cell_ad = full_ad[:2].copy()
    out = fc.correct_anndata(
        full_ad, cell_ad, empty_droplet_cutoff=EMPTY_CUTOFF,
        contamination_chance_cutoff=0.005)
    X = np.asarray(out.X.todense() if sp.issparse(out.X) else out.X)
    # genes x cells expectation [[76,91],[69,87],[60,90]] transposed.
    np.testing.assert_array_equal(
        X, np.array([[76, 69, 60], [91, 87, 90]], dtype=X.dtype))
    assert out.uns["fastcar"]["n_genes_corrected"] == 2


def test_version():
    assert fc.__version__ == "0.1.0"
