"""Shared fixtures and the R-reference build hook for pyfastcar tests."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
R_OUT = os.path.join(HERE, "R_out")
R_DRIVER = os.path.join(HERE, "r_reference_driver.R")

# CMAP path-based R environment (override with PYFASTCAR_RSCRIPT).
_RSCRIPT = os.environ.get(
    "PYFASTCAR_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript")

_R_FILES = [
    "full_matrix.tsv",
    "cell_matrix.tsv",
    "ambient_profile.tsv",
    "corrected_matrix.tsv",
    "ambient_description.tsv",
    "params.tsv",
]


def _r_outputs_present() -> bool:
    return all(os.path.exists(os.path.join(R_OUT, f)) for f in _R_FILES)


def _build_r_reference() -> bool:
    """Run the FastCAR R driver if R is available; return success."""
    if not shutil.which(_RSCRIPT) and not os.path.exists(_RSCRIPT):
        return False
    env = dict(os.environ)
    gcc = "/share/software/user/open/gcc/14.2.0/bin"
    if os.path.isdir(gcc):
        env["PATH"] = gcc + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = (
            "/share/software/user/open/gcc/14.2.0/lib64" + os.pathsep
            + env.get("LD_LIBRARY_PATH", ""))
    try:
        subprocess.run([_RSCRIPT, R_DRIVER, R_OUT], check=True, env=env,
                       capture_output=True, timeout=900)
    except Exception:
        return False
    return _r_outputs_present()


@pytest.fixture(scope="session")
def r_reference():
    """Path to the directory of FastCAR reference outputs.

    Skips the dependent test if the R reference cannot be produced.
    """
    if not _r_outputs_present():
        if not _build_r_reference():
            pytest.skip("FastCAR R reference unavailable")
    return R_OUT


def _read_matrix_tsv(path: str) -> pd.DataFrame:
    """Read an R ``write.table(..., col.names = NA)`` dense matrix."""
    return pd.read_csv(path, sep="\t", index_col=0)


@pytest.fixture(scope="session")
def r_full_matrix(r_reference) -> pd.DataFrame:
    """Full genes x droplets matrix shared by R and Python."""
    return _read_matrix_tsv(os.path.join(r_reference, "full_matrix.tsv"))


@pytest.fixture(scope="session")
def r_cell_matrix(r_reference) -> pd.DataFrame:
    """Filtered genes x cells matrix shared by R and Python."""
    return _read_matrix_tsv(os.path.join(r_reference, "cell_matrix.tsv"))


@pytest.fixture(scope="session")
def r_ambient_profile(r_reference) -> pd.Series:
    """R FastCAR per-gene ambient-RNA profile."""
    df = pd.read_csv(os.path.join(r_reference, "ambient_profile.tsv"), sep="\t")
    return pd.Series(df["profile"].to_numpy(),
                     index=pd.Index(df["gene"], name="gene"))


@pytest.fixture(scope="session")
def r_corrected_matrix(r_reference) -> pd.DataFrame:
    """R FastCAR corrected genes x cells matrix."""
    return _read_matrix_tsv(os.path.join(r_reference, "corrected_matrix.tsv"))


@pytest.fixture(scope="session")
def r_ambient_description(r_reference) -> pd.DataFrame:
    """R FastCAR threshold-profiling table."""
    return pd.read_csv(
        os.path.join(r_reference, "ambient_description.tsv"), sep="\t",
        index_col=0)


@pytest.fixture(scope="session")
def r_params(r_reference) -> dict:
    """R FastCAR scalar parameters / results."""
    df = pd.read_csv(os.path.join(r_reference, "params.tsv"), sep="\t")
    out = {}
    for k, v in zip(df["key"], df["value"]):
        fv = float(v)
        out[k] = int(fv) if fv.is_integer() else fv
    return out
