"""Core FastCAR algorithm -- deterministic per-gene ambient-RNA correction.

This is a faithful, bit-exact port of ``FastCAR_Base.R`` from the R package
``FastCAR`` (LungCellAtlas/FastCAR, v0.1.0).

R FastCAR works on gene x droplet sparse matrices (``dgCMatrix``).  Here the
matrices are accepted as :class:`scipy.sparse` matrices, dense
:class:`numpy.ndarray`, :class:`pandas.DataFrame`, or :class:`anndata.AnnData`
objects.  All FastCAR-internal computation uses CSC sparse arithmetic so the
results match R exactly.

Orientation note
----------------
R FastCAR stores data **genes x droplets** (the 10x CellRanger convention).
:class:`anndata.AnnData` stores data **cells (obs) x genes (var)**.  The
AnnData wrappers in this module transparently transpose, so the user always
passes AnnData objects in the natural cells x genes orientation.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "determine_background_to_remove",
    "remove_background",
    "correct_anndata",
    "describe_ambient_rna_sequence",
    "recommend_empty_cutoff",
]


# --------------------------------------------------------------------------
# matrix coercion helpers
# --------------------------------------------------------------------------
def _as_csc(matrix, *, name: str = "matrix"):
    """Return ``(csc_matrix, gene_names, droplet_names)`` from any input.

    Accepts scipy sparse matrices, numpy arrays and pandas DataFrames.  The
    returned matrix is a CSC matrix with an integer/float dtype, oriented
    *genes x droplets* exactly as the caller supplied it.
    """
    genes = None
    droplets = None
    if isinstance(matrix, pd.DataFrame):
        genes = np.asarray(matrix.index, dtype=object)
        droplets = np.asarray(matrix.columns, dtype=object)
        mat = sp.csc_matrix(matrix.to_numpy())
    elif sp.issparse(matrix):
        mat = matrix.tocsc()
    elif isinstance(matrix, np.ndarray):
        mat = sp.csc_matrix(matrix)
    else:  # last resort -- e.g. list of lists
        mat = sp.csc_matrix(np.asarray(matrix))
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional")
    if genes is None:
        genes = np.array([f"gene{i}" for i in range(mat.shape[0])], dtype=object)
    if droplets is None:
        droplets = np.array(
            [f"droplet{i}" for i in range(mat.shape[1])], dtype=object)
    return mat, genes, droplets


def _col_sums(csc) -> np.ndarray:
    """Per-droplet (column) total UMI counts."""
    return np.asarray(csc.sum(axis=0)).ravel()


def _row_max_over_cols(csc, col_mask: np.ndarray) -> np.ndarray:
    """Per-gene maximum over the selected columns.

    Mirrors ``qlcMatrix::rowMax`` on a column subset of a sparse matrix:
    a gene whose every selected entry is zero gets a max of 0.
    """
    n_genes = csc.shape[0]
    if not col_mask.any():
        return np.zeros(n_genes, dtype=csc.dtype)
    sub = csc[:, col_mask]
    if sub.nnz == 0:
        return np.zeros(n_genes, dtype=sub.dtype)
    # max of stored (positive) values per row; rows with no stored value -> 0.
    sub = sub.tocsr()
    out = np.zeros(n_genes, dtype=sub.dtype)
    # csr.max(axis=1) treats missing entries as 0 implicitly for >=0 data;
    # FastCAR counts are non-negative so this matches qlcMatrix::rowMax.
    rmax = sub.max(axis=1)
    out[:] = np.asarray(rmax.todense()).ravel()
    return out


def _row_nonzero_counts(csc, col_mask: np.ndarray) -> np.ndarray:
    """Per-gene count of non-zero entries over the selected columns.

    Mirrors ``Matrix::rowSums(M[, mask] != 0)`` in R.
    """
    n_genes = csc.shape[0]
    if not col_mask.any():
        return np.zeros(n_genes, dtype=np.int64)
    sub = csc[:, col_mask]
    if sub.nnz == 0:
        return np.zeros(n_genes, dtype=np.int64)
    sub = sub.tocsr()
    indicator = sub.copy()
    indicator.data = (indicator.data != 0).astype(np.int64)
    indicator.eliminate_zeros()
    return np.asarray(indicator.sum(axis=1), dtype=np.int64).ravel()


# --------------------------------------------------------------------------
# 1. ambient-RNA profile
# --------------------------------------------------------------------------
def determine_background_to_remove(
    full_cell_matrix,
    cell_matrix=None,
    empty_droplet_cutoff: int = 100,
    contamination_chance_cutoff: float = 0.005,
    *,
    return_table: bool = False,
):
    """Compute the per-gene ambient-RNA profile to subtract.

    Faithful port of R ``determine.background.to.remove``.

    Parameters
    ----------
    full_cell_matrix
        The **full / unfiltered** gene x droplet count matrix (scipy sparse,
        numpy array, pandas DataFrame).  Contains both real cells and empty
        droplets.
    cell_matrix
        The filtered gene x droplet matrix of real cells.  Only its shape is
        used (R uses it solely for ``ncol``); may be ``None``.
    empty_droplet_cutoff
        Droplets with a total UMI count strictly below this value (``thE``)
        are treated as empty.  Default 100.
    contamination_chance_cutoff
        Allowable contamination fraction (``frAA``).  A gene is corrected
        only when its ``frC`` (fraction of non-zero empty droplets that
        contain the gene) is **greater than** this value.  Default 0.005.
    return_table
        When ``True`` also return a per-gene :class:`pandas.DataFrame` with
        the diagnostic columns ``gMax``, ``frC`` and ``occurrences``.

    Returns
    -------
    ambient_profile : pandas.Series
        Per-gene amount to subtract (``gMax`` for corrected genes, 0
        otherwise), indexed by gene name -- the R named ``backGroundMax``
        vector.
    table : pandas.DataFrame, optional
        Only when ``return_table=True``.
    """
    full, genes, _ = _as_csc(full_cell_matrix, name="full_cell_matrix")

    col_totals = _col_sums(full)
    empty_mask = col_totals < empty_droplet_cutoff

    # gMax: highest count of every gene across the empty droplets.
    background_max = _row_max_over_cols(full, empty_mask).astype(np.float64)

    # nEmpty: empty droplets that are *not* unused barcodes (>0 reads).
    # R: table((colSums < cutoff) & (colSums > 0))[2]
    n_empty = int(np.count_nonzero(empty_mask & (col_totals > 0)))

    # occurrences: number of empty droplets in which each gene appears.
    occurrences = _row_nonzero_counts(full, empty_mask)

    # frC: probability a background read of a gene ends up in a cell.
    with np.errstate(divide="ignore", invalid="ignore"):
        fr_c = occurrences / n_empty if n_empty else np.zeros_like(
            occurrences, dtype=np.float64)
    fr_c = np.asarray(fr_c, dtype=np.float64)

    # genes whose contamination chance is too low are not corrected.
    # R: backGroundMax[probabilityCellContamination < cutoff] = 0
    # so a gene is *selected* exactly when NOT (frC < cutoff), i.e.
    # frC >= cutoff -- the strict-< test is what FastCAR itself uses.
    low = fr_c < contamination_chance_cutoff
    corrected = background_max.copy()
    corrected[low] = 0.0

    profile = pd.Series(corrected, index=pd.Index(genes, name="gene"),
                        name="ambient_rna_profile")

    if return_table:
        table = pd.DataFrame(
            {
                "gMax": background_max,
                "frC": fr_c,
                "occurrences": occurrences,
                "corrected": corrected,
                "selected": ~low,
            },
            index=pd.Index(genes, name="gene"),
        )
        return profile, table
    return profile


# --------------------------------------------------------------------------
# 2. apply the correction
# --------------------------------------------------------------------------
def remove_background(gene_cell_matrix, ambient_rna_profile):
    """Subtract the ambient-RNA profile from a gene x cell matrix.

    Faithful port of R ``remove.background``.  For every gene with a
    positive value in ``ambient_rna_profile`` the value is subtracted from
    that gene's count in *every* cell; results below zero are floored at
    zero and explicit zeros are dropped.

    Parameters
    ----------
    gene_cell_matrix
        The gene x cell count matrix to correct (scipy sparse, numpy array
        or pandas DataFrame).
    ambient_rna_profile
        The per-gene amount to subtract -- a :class:`pandas.Series` from
        :func:`determine_background_to_remove`, or any gene -> value mapping
        / array aligned to the matrix rows.

    Returns
    -------
    corrected
        The corrected matrix, in the same container type as the input
        (sparse stays sparse, DataFrame stays DataFrame, ndarray stays
        ndarray).
    """
    mat, genes, droplets = _as_csc(gene_cell_matrix, name="gene_cell_matrix")

    profile = _align_profile(ambient_rna_profile, genes)

    csc = mat.astype(np.result_type(mat.dtype, np.float64)).tocsc().copy()
    # subtract profile[gene] from every stored entry of that gene's row.
    if csc.nnz:
        # row index of each stored value:
        row_of_value = csc.indices  # rows for a CSC matrix
        csc.data = csc.data - profile[row_of_value]
        # floor negatives at zero, then drop explicit zeros (R's drop0).
        np.clip(csc.data, 0, None, out=csc.data)
        csc.eliminate_zeros()

    return _restore_container(gene_cell_matrix, csc, genes, droplets)


def _align_profile(ambient_rna_profile, genes: np.ndarray) -> np.ndarray:
    """Return a float vector of subtract-amounts aligned to ``genes``."""
    if isinstance(ambient_rna_profile, pd.Series):
        # align by gene name; genes absent from the profile -> 0.
        aligned = ambient_rna_profile.reindex(genes).fillna(0.0)
        return np.asarray(aligned.to_numpy(), dtype=np.float64)
    if isinstance(ambient_rna_profile, dict):
        return np.array([float(ambient_rna_profile.get(g, 0.0)) for g in genes],
                        dtype=np.float64)
    arr = np.asarray(ambient_rna_profile, dtype=np.float64).ravel()
    if arr.shape[0] != genes.shape[0]:
        raise ValueError(
            "ambient_rna_profile length does not match the number of genes")
    return arr


def _restore_container(original, csc, genes, droplets):
    """Cast ``csc`` back to the container type of ``original``."""
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame(csc.toarray(), index=pd.Index(genes),
                            columns=pd.Index(droplets))
    if sp.issparse(original):
        # preserve the original sparse format.
        fmt = original.format
        return csc.asformat(fmt)
    if isinstance(original, np.ndarray):
        return csc.toarray()
    return csc


# --------------------------------------------------------------------------
# 3. AnnData-friendly one-shot wrapper
# --------------------------------------------------------------------------
def correct_anndata(
    full_adata,
    cell_adata=None,
    empty_droplet_cutoff: int = 100,
    contamination_chance_cutoff: float = 0.005,
    *,
    layer: str | None = None,
    inplace: bool = False,
):
    """Run the full FastCAR correction on AnnData objects.

    AnnData stores data **cells x genes**; FastCAR works **genes x cells**.
    This wrapper transposes internally so the user always works in the
    natural AnnData orientation.

    Parameters
    ----------
    full_adata
        AnnData of the **full / unfiltered** matrix (real cells + empty
        droplets), cells x genes.  Used to build the ambient profile.
    cell_adata
        AnnData of the filtered real cells, cells x genes.  This is the
        object that gets corrected.  When ``None`` the correction is applied
        to ``full_adata`` itself (rarely wanted -- usually you pass both).
    empty_droplet_cutoff, contamination_chance_cutoff
        See :func:`determine_background_to_remove`.
    layer
        Optional ``.layers`` key to read counts from (and write to) instead
        of ``.X``.
    inplace
        When ``True`` modify ``cell_adata`` in place and return it; when
        ``False`` (default) operate on a copy.

    Returns
    -------
    corrected_adata : anndata.AnnData
        The corrected cell AnnData.  ``.uns['fastcar']`` records the
        parameters, the ambient profile and the per-gene diagnostic table.
    """
    full_mat = _get_counts(full_adata, layer)            # cells x genes
    full_genes = np.asarray(full_adata.var_names, dtype=object)

    # build the profile on the genes x droplets matrix.
    full_gd = sp.csc_matrix(full_mat).T
    profile, table = determine_background_to_remove(
        _named_csc(full_gd, full_genes),
        cell_matrix=None,
        empty_droplet_cutoff=empty_droplet_cutoff,
        contamination_chance_cutoff=contamination_chance_cutoff,
        return_table=True,
    )

    target = full_adata if cell_adata is None else cell_adata
    out = target if inplace else target.copy()

    cell_mat = _get_counts(out, layer)                   # cells x genes
    cell_genes = np.asarray(out.var_names, dtype=object)
    cell_gd = sp.csc_matrix(cell_mat).T                  # genes x cells
    corrected_gd = remove_background(
        _named_csc(cell_gd, cell_genes), profile)
    corrected = sp.csc_matrix(corrected_gd).T            # cells x genes

    _set_counts(out, corrected, layer)
    out.uns["fastcar"] = {
        "empty_droplet_cutoff": empty_droplet_cutoff,
        "contamination_chance_cutoff": contamination_chance_cutoff,
        "ambient_rna_profile": profile,
        "diagnostics": table,
        "n_genes_corrected": int((profile > 0).sum()),
    }
    return out


def _named_csc(csc, genes):
    """Wrap a genes x droplets CSC matrix as a DataFrame so gene names ride."""
    return pd.DataFrame.sparse.from_spmatrix(
        csc, index=pd.Index(np.asarray(genes, dtype=object)))


def _get_counts(adata, layer):
    x = adata.layers[layer] if layer is not None else adata.X
    return sp.csc_matrix(x) if not sp.issparse(x) else x.tocsc()


def _set_counts(adata, value, layer):
    if layer is not None:
        adata.layers[layer] = value
    else:
        adata.X = value


# --------------------------------------------------------------------------
# 4. threshold profiling
# --------------------------------------------------------------------------
def describe_ambient_rna_sequence(
    full_cell_matrix,
    start: int,
    stop: int,
    by: int,
    contamination_chance_cutoff: float = 0.005,
):
    """Profile how the empty-droplet cutoff affects the ambient correction.

    Faithful port of R ``describe.ambient.RNA.sequence``.  For every cutoff
    in ``range(start, stop + 1, by)`` it reports the number of non-empty
    empty droplets, the number of genes seen in the background, and the
    number of genes that would be corrected.

    Returns
    -------
    pandas.DataFrame
        Indexed by the empty-droplet cutoff, with columns
        ``nEmptyDroplets``, ``genesInBackground`` and ``genesContaminating``.
    """
    full, _, _ = _as_csc(full_cell_matrix, name="full_cell_matrix")
    col_totals = _col_sums(full)

    cutoffs = list(_r_seq(start, stop, by))
    n_empty_droplets = []
    genes_in_background = []
    genes_contaminating = []
    for cutoff in cutoffs:
        empty_mask = col_totals < cutoff
        n_empty = int(np.count_nonzero(empty_mask & (col_totals > 0)))
        occurrences = _row_nonzero_counts(full, empty_mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            fr_c = (occurrences / n_empty if n_empty
                    else np.zeros_like(occurrences, dtype=np.float64))
        fr_c = np.asarray(fr_c, dtype=np.float64)
        n_empty_droplets.append(n_empty)
        genes_in_background.append(int(np.count_nonzero(occurrences != 0)))
        genes_contaminating.append(
            int(np.count_nonzero(fr_c > contamination_chance_cutoff)))

    return pd.DataFrame(
        {
            "nEmptyDroplets": n_empty_droplets,
            "genesInBackground": genes_in_background,
            "genesContaminating": genes_contaminating,
        },
        index=pd.Index(cutoffs, name="emptyDropletCutoff"),
    )


def recommend_empty_cutoff(ambient_profile: pd.DataFrame) -> int:
    """Suggest an empty-droplet cutoff from a profiling table.

    Faithful port of R ``recommend.empty.cutoff``: returns the *first*
    cutoff at which the number of genes flagged for correction reaches its
    maximum.

    Parameters
    ----------
    ambient_profile
        The :class:`pandas.DataFrame` returned by
        :func:`describe_ambient_rna_sequence`.
    """
    col = ambient_profile["genesContaminating"]
    highest = col.max()
    first = col.to_numpy().tolist().index(highest)
    return int(ambient_profile.index[first])


def _r_seq(start, stop, by) -> Iterable[int]:
    """Reproduce R's ``seq(start, stop, by)`` (inclusive of an exact stop)."""
    n = int(np.floor((stop - start) / by + 1e-10)) + 1
    return [start + i * by for i in range(max(n, 0))]
