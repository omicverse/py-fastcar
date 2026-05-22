"""10x CellRanger loaders -- ports of R ``read.cell.matrix`` / ``read.full.matrix``.

R FastCAR delegates to ``Seurat::Read10X``.  Here the loaders use scanpy /
anndata when available (the natural omicverse path); they fall back to a
self-contained Matrix-Market reader so the package has no hard scanpy
dependency.

Both loaders return a **genes x droplets** :class:`pandas.DataFrame` -- the
orientation FastCAR's core functions expect -- to mirror R FastCAR exactly.
"""
from __future__ import annotations

import gzip
import os

import pandas as pd
import scipy.io
import scipy.sparse as sp

__all__ = ["read_cell_matrix", "read_full_matrix", "read_10x"]


def _open_maybe_gz(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _find(folder: str, *names: str) -> str | None:
    for name in names:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    return None


def read_10x(folder_location: str, *, as_anndata: bool = False):
    """Read a 10x CellRanger ``matrix.mtx`` directory.

    Parameters
    ----------
    folder_location
        Directory containing ``matrix.mtx[.gz]``, a features/genes file and
        a barcodes file (the standard CellRanger ``filtered``/``raw`` output).
    as_anndata
        When ``True`` return a cells x genes :class:`anndata.AnnData`;
        otherwise return a genes x droplets :class:`pandas.DataFrame`
        (FastCAR's native orientation).

    Returns
    -------
    pandas.DataFrame or anndata.AnnData
    """
    mtx = _find(folder_location, "matrix.mtx.gz", "matrix.mtx")
    if mtx is None:
        raise FileNotFoundError(
            f"no matrix.mtx[.gz] found in {folder_location!r}")
    feat = _find(folder_location, "features.tsv.gz", "features.tsv",
                 "genes.tsv.gz", "genes.tsv")
    if feat is None:
        raise FileNotFoundError(
            f"no features.tsv/genes.tsv found in {folder_location!r}")
    barc = _find(folder_location, "barcodes.tsv.gz", "barcodes.tsv")
    if barc is None:
        raise FileNotFoundError(
            f"no barcodes.tsv found in {folder_location!r}")

    with _open_maybe_gz(mtx) as fh:
        matrix = scipy.io.mmread(fh)            # genes x droplets
    matrix = sp.csc_matrix(matrix)

    with _open_maybe_gz(feat) as fh:
        feat_rows = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    # CellRanger gene id is column 0, symbol column 1; Seurat uses the symbol.
    genes = [r[1] if len(r) > 1 else r[0] for r in feat_rows]
    genes = _make_unique(genes)

    with _open_maybe_gz(barc) as fh:
        barcodes = [line.strip() for line in fh if line.strip()]

    if as_anndata:
        import anndata as ad
        return ad.AnnData(
            X=matrix.T.tocsr(),
            obs=pd.DataFrame(index=pd.Index(barcodes)),
            var=pd.DataFrame(index=pd.Index(genes)),
        )
    return pd.DataFrame.sparse.from_spmatrix(
        matrix, index=pd.Index(genes), columns=pd.Index(barcodes))


def _make_unique(names) -> list[str]:
    """Reproduce Seurat/make.unique disambiguation of repeated gene names."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def read_cell_matrix(cell_folder_location: str, *, as_anndata: bool = False):
    """Load the filtered (real-cell) 10x matrix -- R ``read.cell.matrix``."""
    return read_10x(cell_folder_location, as_anndata=as_anndata)


def read_full_matrix(full_folder_location: str, *, as_anndata: bool = False):
    """Load the full / unfiltered 10x matrix -- R ``read.full.matrix``."""
    return read_10x(full_folder_location, as_anndata=as_anndata)
