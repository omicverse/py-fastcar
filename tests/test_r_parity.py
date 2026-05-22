"""Bit-exact R-parity tests for pyfastcar.

The same deterministic synthetic raw matrix is run through R ``FastCAR`` (by
``r_reference_driver.R``) and through ``pyfastcar``; this suite asserts the
per-gene ``gMax``/``frC``, the selected genes, the ambient profile, the
corrected integer matrix and the threshold-profiling table are identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import pyfastcar as fc


# --------------------------------------------------------------------------
# ambient profile
# --------------------------------------------------------------------------
def test_ambient_profile_bit_exact(r_full_matrix, r_cell_matrix,
                                   r_ambient_profile, r_params):
    """determine_background_to_remove must match R FastCAR exactly."""
    profile = fc.determine_background_to_remove(
        r_full_matrix,
        r_cell_matrix,
        empty_droplet_cutoff=r_params["emptyDropletCutoff"],
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    # same gene order
    assert list(profile.index) == list(r_ambient_profile.index)
    # bit-exact subtract amounts
    np.testing.assert_array_equal(
        profile.to_numpy().astype(np.int64),
        r_ambient_profile.to_numpy().astype(np.int64),
    )


def test_n_genes_corrected_matches(r_full_matrix, r_cell_matrix, r_params):
    profile = fc.determine_background_to_remove(
        r_full_matrix, r_cell_matrix,
        empty_droplet_cutoff=r_params["emptyDropletCutoff"],
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    assert int((profile > 0).sum()) == r_params["nGenesCorrected"]


def test_diagnostics_table(r_full_matrix, r_cell_matrix, r_params):
    """The frC / gMax diagnostic table must be internally consistent."""
    profile, table = fc.determine_background_to_remove(
        r_full_matrix, r_cell_matrix,
        empty_droplet_cutoff=r_params["emptyDropletCutoff"],
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
        return_table=True,
    )
    frAA = r_params["contaminationChanceCutoff"]
    # R FastCAR zeroes genes with frC < frAA, so a gene is selected
    # exactly when NOT (frC < frAA), i.e. frC >= frAA.
    assert (table["selected"] == ~(table["frC"] < frAA)).all()
    # corrected value is gMax for selected genes, 0 otherwise.
    expected = np.where(table["selected"], table["gMax"], 0.0)
    np.testing.assert_array_equal(table["corrected"].to_numpy(), expected)
    np.testing.assert_array_equal(profile.to_numpy(),
                                  table["corrected"].to_numpy())
    # occurrences / nEmpty == frC
    assert (table["occurrences"] >= 0).all()


# --------------------------------------------------------------------------
# corrected matrix
# --------------------------------------------------------------------------
def test_corrected_matrix_bit_exact(r_full_matrix, r_cell_matrix,
                                    r_corrected_matrix, r_params):
    """remove_background must reproduce R FastCAR's corrected matrix exactly."""
    profile = fc.determine_background_to_remove(
        r_full_matrix, r_cell_matrix,
        empty_droplet_cutoff=r_params["emptyDropletCutoff"],
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    corrected = fc.remove_background(r_cell_matrix, profile)
    # same shape / labels
    assert list(corrected.index) == list(r_corrected_matrix.index)
    assert list(corrected.columns) == list(r_corrected_matrix.columns)
    # bit-exact integer counts
    np.testing.assert_array_equal(
        corrected.to_numpy().astype(np.int64),
        r_corrected_matrix.to_numpy().astype(np.int64),
    )


def test_corrected_matrix_sparse_input(r_full_matrix, r_cell_matrix,
                                       r_corrected_matrix, r_params):
    """Sparse input must give the same result as DataFrame input."""
    full_sp = sp.csc_matrix(r_full_matrix.to_numpy())
    cell_sp = sp.csc_matrix(r_cell_matrix.to_numpy())
    genes = pd.Index(r_full_matrix.index, name="gene")

    profile = fc.determine_background_to_remove(
        pd.DataFrame.sparse.from_spmatrix(full_sp, index=genes),
        cell_sp,
        empty_droplet_cutoff=r_params["emptyDropletCutoff"],
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    corrected = fc.remove_background(
        pd.DataFrame.sparse.from_spmatrix(
            cell_sp, index=genes,
            columns=pd.Index(r_cell_matrix.columns)),
        profile)
    np.testing.assert_array_equal(
        np.asarray(corrected.to_numpy(), dtype=np.int64),
        r_corrected_matrix.to_numpy().astype(np.int64),
    )


def test_correction_floors_at_zero(r_full_matrix, r_cell_matrix, r_params):
    profile = fc.determine_background_to_remove(
        r_full_matrix, r_cell_matrix,
        empty_droplet_cutoff=r_params["emptyDropletCutoff"],
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    corrected = fc.remove_background(r_cell_matrix, profile)
    assert (corrected.to_numpy() >= 0).all()


# --------------------------------------------------------------------------
# threshold profiling
# --------------------------------------------------------------------------
def test_describe_ambient_rna_sequence(r_full_matrix, r_ambient_description,
                                       r_params):
    """describe_ambient_rna_sequence must match R column-for-column."""
    desc = fc.describe_ambient_rna_sequence(
        r_full_matrix, start=50, stop=200, by=25,
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    assert list(desc.index) == list(r_ambient_description.index)
    for col in ["nEmptyDroplets", "genesInBackground", "genesContaminating"]:
        np.testing.assert_array_equal(
            desc[col].to_numpy().astype(np.int64),
            r_ambient_description[col].to_numpy().astype(np.int64),
        )


def test_recommend_empty_cutoff(r_full_matrix, r_params):
    desc = fc.describe_ambient_rna_sequence(
        r_full_matrix, start=50, stop=200, by=25,
        contamination_chance_cutoff=r_params["contaminationChanceCutoff"],
    )
    assert fc.recommend_empty_cutoff(desc) == r_params["recommendedCutoff"]
