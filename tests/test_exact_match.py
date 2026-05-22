"""Exact R-parity tests.

These tests run the R reference (``FastCAR::determine.background.to.remove`` +
``remove.background`` + ``describe.ambient.RNA.sequence``) via the
``r_reference_driver.R`` script on a deterministic synthetic raw matrix,
save every intermediate to a directory, then check that the Python port
produces identical per-gene ``gMax``/``frC``, selected genes, the corrected
integer matrix and the threshold-profiling table.

FastCAR is fully deterministic -- there is no RNG anywhere in the algorithm --
so the agreement is **bit-exact**.

The tests are skipped if Rscript + the required R packages aren't available,
so running ``pytest`` with just Python still works.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyfastcar as fc


HERE = Path(__file__).parent
DRIVER = HERE / "r_reference_driver.R"

_R_FILES = [
    "full_matrix.tsv",
    "cell_matrix.tsv",
    "ambient_profile.tsv",
    "corrected_matrix.tsv",
    "ambient_description.tsv",
    "params.tsv",
]


def _find_rscript() -> str | None:
    """Prefer the CMAP env's Rscript if available; else fall back to PATH."""
    candidates = [
        os.environ.get("PYFASTCAR_RSCRIPT", ""),
        "/scratch/users/steorra/env/CMAP/bin/Rscript",
        shutil.which("Rscript") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


@pytest.fixture(scope="module")
def r_ref_dir(tmp_path_factory) -> Path:
    """Run the FastCAR R driver and return the directory of TSV outputs."""
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript not found -- skipping R-parity tests")
    outdir = tmp_path_factory.mktemp("r_ref")
    env = os.environ.copy()
    # Make gcc visible so R can load compiled FastCAR / Matrix shared libs.
    gcc = "/share/software/user/open/gcc/14.2.0"
    if os.path.isdir(gcc):
        env["PATH"] = f"{gcc}/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{gcc}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(
        [rscript, str(DRIVER), str(outdir)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0 or not all(
        (outdir / f).exists() for f in _R_FILES
    ):
        pytest.skip(
            "R reference driver failed (likely missing FastCAR/Matrix in the "
            f"env):\nSTDERR:\n{proc.stderr[-1500:]}\nSTDOUT:\n{proc.stdout[-500:]}"
        )
    return outdir


def _matrix(p: Path) -> pd.DataFrame:
    """Read an R ``write.table(..., col.names = NA)`` dense matrix."""
    return pd.read_csv(p, sep="\t", index_col=0)


def _params(p: Path) -> dict:
    df = pd.read_csv(p, sep="\t")
    out: dict = {}
    for k, v in zip(df["key"], df["value"]):
        fv = float(v)
        out[k] = int(fv) if fv.is_integer() else fv
    return out


# --------------------------------------------------------------------------
# ambient profile
# --------------------------------------------------------------------------
def test_ambient_profile_bit_exact(r_ref_dir: Path):
    """determine_background_to_remove must match R FastCAR exactly."""
    params = _params(r_ref_dir / "params.tsv")
    full = _matrix(r_ref_dir / "full_matrix.tsv")
    cell = _matrix(r_ref_dir / "cell_matrix.tsv")
    r_profile = pd.read_csv(r_ref_dir / "ambient_profile.tsv", sep="\t")

    profile = fc.determine_background_to_remove(
        full, cell,
        empty_droplet_cutoff=params["emptyDropletCutoff"],
        contamination_chance_cutoff=params["contaminationChanceCutoff"],
    )
    assert list(profile.index) == list(r_profile["gene"])
    np.testing.assert_array_equal(
        profile.to_numpy().astype(np.int64),
        r_profile["profile"].to_numpy().astype(np.int64),
    )
    assert int((profile > 0).sum()) == params["nGenesCorrected"]


def test_diagnostics_table(r_ref_dir: Path):
    """The frC / gMax diagnostic table must be internally consistent."""
    params = _params(r_ref_dir / "params.tsv")
    full = _matrix(r_ref_dir / "full_matrix.tsv")
    cell = _matrix(r_ref_dir / "cell_matrix.tsv")

    profile, table = fc.determine_background_to_remove(
        full, cell,
        empty_droplet_cutoff=params["emptyDropletCutoff"],
        contamination_chance_cutoff=params["contaminationChanceCutoff"],
        return_table=True,
    )
    frAA = params["contaminationChanceCutoff"]
    # R FastCAR zeroes genes with frC < frAA.
    assert (table["selected"] == ~(table["frC"] < frAA)).all()
    expected = np.where(table["selected"], table["gMax"], 0.0)
    np.testing.assert_array_equal(table["corrected"].to_numpy(), expected)
    np.testing.assert_array_equal(profile.to_numpy(),
                                  table["corrected"].to_numpy())


# --------------------------------------------------------------------------
# corrected matrix
# --------------------------------------------------------------------------
def test_corrected_matrix_bit_exact(r_ref_dir: Path):
    """remove_background must reproduce R FastCAR's corrected matrix exactly."""
    params = _params(r_ref_dir / "params.tsv")
    full = _matrix(r_ref_dir / "full_matrix.tsv")
    cell = _matrix(r_ref_dir / "cell_matrix.tsv")
    r_corrected = _matrix(r_ref_dir / "corrected_matrix.tsv")

    profile = fc.determine_background_to_remove(
        full, cell,
        empty_droplet_cutoff=params["emptyDropletCutoff"],
        contamination_chance_cutoff=params["contaminationChanceCutoff"],
    )
    corrected = fc.remove_background(cell, profile)
    assert list(corrected.index) == list(r_corrected.index)
    assert list(corrected.columns) == list(r_corrected.columns)
    np.testing.assert_array_equal(
        corrected.to_numpy().astype(np.int64),
        r_corrected.to_numpy().astype(np.int64),
    )
    assert (corrected.to_numpy() >= 0).all()


def test_corrected_matrix_sparse_input(r_ref_dir: Path):
    """Sparse input must give the same result as DataFrame input."""
    params = _params(r_ref_dir / "params.tsv")
    full = _matrix(r_ref_dir / "full_matrix.tsv")
    cell = _matrix(r_ref_dir / "cell_matrix.tsv")
    r_corrected = _matrix(r_ref_dir / "corrected_matrix.tsv")

    genes = pd.Index(full.index, name="gene")
    full_sp = pd.DataFrame.sparse.from_spmatrix(
        sp.csc_matrix(full.to_numpy()), index=genes)
    cell_sp = pd.DataFrame.sparse.from_spmatrix(
        sp.csc_matrix(cell.to_numpy()), index=genes,
        columns=pd.Index(cell.columns))

    profile = fc.determine_background_to_remove(
        full_sp, cell_sp,
        empty_droplet_cutoff=params["emptyDropletCutoff"],
        contamination_chance_cutoff=params["contaminationChanceCutoff"],
    )
    corrected = fc.remove_background(cell_sp, profile)
    np.testing.assert_array_equal(
        np.asarray(corrected.to_numpy(), dtype=np.int64),
        r_corrected.to_numpy().astype(np.int64),
    )


# --------------------------------------------------------------------------
# threshold profiling
# --------------------------------------------------------------------------
def test_describe_ambient_rna_sequence(r_ref_dir: Path):
    """describe_ambient_rna_sequence must match R column-for-column."""
    params = _params(r_ref_dir / "params.tsv")
    full = _matrix(r_ref_dir / "full_matrix.tsv")
    r_desc = pd.read_csv(
        r_ref_dir / "ambient_description.tsv", sep="\t", index_col=0)

    desc = fc.describe_ambient_rna_sequence(
        full, start=50, stop=200, by=25,
        contamination_chance_cutoff=params["contaminationChanceCutoff"],
    )
    assert list(desc.index) == list(r_desc.index)
    for col in ["nEmptyDroplets", "genesInBackground", "genesContaminating"]:
        np.testing.assert_array_equal(
            desc[col].to_numpy().astype(np.int64),
            r_desc[col].to_numpy().astype(np.int64),
        )
    assert fc.recommend_empty_cutoff(desc) == params["recommendedCutoff"]
