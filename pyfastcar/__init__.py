"""pyfastcar -- pure-Python port of the R package FastCAR.

FastCAR (Gocht / Berg et al., *BMC Genomics* 2023) is a deterministic,
per-gene correction for ambient RNA in droplet-based single-cell RNA-seq.
It inspects the "empty" droplets (libraries with very few UMIs), builds an
ambient-RNA profile, and subtracts that profile -- gene by gene -- from the
counts of every real cell.

The algorithm
-------------
1. *Empty* droplets are the columns of the full (unfiltered) gene x droplet
   matrix whose total UMI count is below ``empty_droplet_cutoff``.
2. For each gene, ``gMax`` is the highest count seen in any single empty
   droplet, and ``frC`` is the fraction of non-zero empty droplets that
   contain the gene.
3. A gene is corrected only when ``frC`` exceeds
   ``contamination_chance_cutoff``; for those genes ``gMax`` is subtracted
   from every cell's count, with negatives floored at zero.

Main entry points
-----------------
- :func:`determine_background_to_remove` -- build the ambient-RNA profile.
- :func:`remove_background`              -- apply the correction to a matrix.
- :func:`correct_anndata`                -- AnnData-friendly one-shot wrapper.
- :func:`describe_ambient_rna_sequence`  -- profile the empty-droplet cutoff.
- :func:`recommend_empty_cutoff`         -- suggest a cutoff from a profile.
- :func:`read_cell_matrix` / :func:`read_full_matrix` -- 10x loaders.

The port is **bit-exact** with R FastCAR 0.1.0.
"""
from __future__ import annotations

from .core import (
    correct_anndata,
    describe_ambient_rna_sequence,
    determine_background_to_remove,
    recommend_empty_cutoff,
    remove_background,
)
from .io import read_cell_matrix, read_full_matrix

__version__ = "0.1.0"

# R-FastCAR-compatible aliases (dotted R names -> snake_case here).
determine_background_to_remove.r_name = "determine.background.to.remove"
remove_background.r_name = "remove.background"

__all__ = [
    "determine_background_to_remove",
    "remove_background",
    "correct_anndata",
    "describe_ambient_rna_sequence",
    "recommend_empty_cutoff",
    "read_cell_matrix",
    "read_full_matrix",
    "__version__",
]
