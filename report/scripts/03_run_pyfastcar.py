"""pyfastcar run on one task, mirroring 02_run_fastcar.R.

Usage: python 03_run_pyfastcar.py <task_dir> <out_dir>
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pyfastcar


def dump(m, prefix):
    m = sp.csc_matrix(m)
    m.sort_indices()
    m.indptr.astype(np.int32).tofile(f"{prefix}_p.bin")
    m.indices.astype(np.int32).tofile(f"{prefix}_i.bin")
    m.data.astype(np.float64).tofile(f"{prefix}_x.bin")


def main():
    td, od = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    od.mkdir(parents=True, exist_ok=True)
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")), dtype=np.float64)
    genes = pd.read_csv(td / "genes.tsv", sep="\t").id.to_numpy()
    droplets = (td / "droplets.tsv").read_text().split()
    cells = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str).cell
    toc = tod[:, pd.Index(droplets).get_indexer(cells)].tocsc()
    res = {"engine": "pyfastcar", "version": pyfastcar.__version__, "python": sys.version.split()[0]}
    t0 = time.perf_counter()
    amb = pyfastcar.determine_background_to_remove(tod, toc, 100, 0.05)
    fixed = pyfastcar.remove_background(toc, np.asarray(amb))
    res["seconds_fixed"] = time.perf_counter() - t0
    np.asarray(amb, dtype=np.float64).tofile(od / "profile_fixed.bin")
    dump(fixed, od / "fixed")
    t0 = time.perf_counter()
    desc = pyfastcar.describe_ambient_rna_sequence(tod, 10, 500, 10, 0.05)
    cutoff = pyfastcar.recommend_empty_cutoff(desc)
    amb2 = pyfastcar.determine_background_to_remove(tod, toc, cutoff, 0.05)
    rec = pyfastcar.remove_background(toc, np.asarray(amb2))
    res["seconds_recommended"] = time.perf_counter() - t0
    np.asarray(amb2, dtype=np.float64).tofile(od / "profile_recommended.bin")
    dump(rec, od / "recommended")
    desc.to_csv(od / "describe.tsv", sep="\t")
    res.update(cutoff=int(cutoff), genes_fixed=int((np.asarray(amb) > 0).sum()),
               genes_recommended=int((np.asarray(amb2) > 0).sum()), dim=list(toc.shape),
               seconds=res["seconds_fixed"])
    (od / "result.json").write_text(json.dumps(res))
    print(td.name, f"cutoff={cutoff} genes={res['genes_fixed']}/{res['genes_recommended']} "
          f"fixed {res['seconds_fixed']:.1f}s recommended {res['seconds_recommended']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
