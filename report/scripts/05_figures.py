"""Figure and the numbers quoted in the pyfastcar technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir>
Compares FastCAR 0.1.0 and pyfastcar on five 10x Genomics human-mouse mixtures. In a mixture, counts of mouse
genes in human cells (and of human genes in mouse cells) can only come from ambient RNA, which shows how much
of the contamination a correction removes and how much of the cells' own RNA it takes with it.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY, C_TRUE = "#eb6834", "#2a78d6", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "FastCAR", "py": "pyfastcar"}
CFG = ("fixed", "recommended")
LABEL = {"hgmm_1k_v3": "hgmm 1k (v3)", "hgmm_5k_v3": "hgmm 5k (v3)", "hgmm_6k": "hgmm 6k (v2)",
         "hgmm_10k_v3": "hgmm 10k (v3)", "hgmm_12k": "hgmm 12k (v2)"}
EXAMPLE = "hgmm_12k"


def result(engine, task, run=1):
    d = RUNS / ENG[engine] / task / f"run{run}"
    r = json.load(open(d / "result.json"))
    mats = {c: sp.csc_matrix((np.fromfile(d / f"{c}_x.bin"), np.fromfile(d / f"{c}_i.bin", np.int32),
                              np.fromfile(d / f"{c}_p.bin", np.int32)), shape=tuple(r["dim"])) for c in CFG}
    prof = {c: np.fromfile(d / f"profile_{c}.bin") for c in CFG}
    desc = pd.read_csv(d / "describe.tsv", sep="\t", index_col=0)
    return r, mats, prof, desc


rows, descs, cellwise, umap_frac, prof_pairs = [], {}, {}, {}, []
for t in LABEL:
    td = TASKS / t
    genes = pd.read_csv(td / "genes.tsv", sep="\t")
    species = pd.read_csv(td / "species.tsv", sep="\t")
    droplets = (td / "droplets.tsv").read_text().split()
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")))
    cells = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str).cell
    toc = tod[:, pd.Index(droplets).get_indexer(cells)].tocsc().astype(np.float64)
    sp_of = species.set_index("cell").species.reindex(cells).to_numpy()
    mouse_gene = (genes.genome == "mouse").to_numpy()
    res = {e: result(e, t) for e in ENG}
    hum, mou = sp_of == "human", sp_of == "mouse"
    pure = hum | mou

    def split(m):
        """Per cell: counts of the other species' genes and of the own species' genes."""
        m_cnt = np.asarray(m[mouse_gene].sum(axis=0)).ravel()
        h_cnt = np.asarray(m[~mouse_gene].sum(axis=0)).ravel()
        return (np.where(hum, m_cnt, np.where(mou, h_cnt, np.nan)),
                np.where(hum, h_cnt, np.where(mou, m_cnt, np.nan)))

    o0, s0 = split(toc)
    row = dict(dataset=t, label=LABEL[t], droplets=tod.shape[1], cells=toc.shape[1],
               human=int(hum.sum()), mouse=int(mou.sum()), other_share=o0[pure].sum() / (o0 + s0)[pure].sum())
    r0 = res["R"][0]
    row["cutoff_R"], row["cutoff_py"] = r0["cutoff"], res["py"][0]["cutoff"]
    for c in CFG:
        row[f"genes_{c}"] = r0[f"genes_{c}"]
        prof = res["R"][2][c]
        row[f"genes_{c}_mt"] = int(((prof > 0) & genes.symbol.str.contains(r"^(?:mt-|MT-|hg19_MT-|mm10_mt-)",
                                                                           regex=True).to_numpy()).sum())
        for e in ENG:
            o1, s1 = split(res[e][1][c])
            row[f"removed_{c}_{e}"] = 1 - o1[pure].sum() / o0[pure].sum()
            row[f"lost_{c}_{e}"] = 1 - s1[pure].sum() / s0[pure].sum()
            row[f"sec_{c}_{e}"] = min(json.load(open(RUNS / ENG[e] / t / f"run{k}" / "result.json"))[f"seconds_{c}"]
                                      for k in (1, 2))            # faster of the two runs
            if t == EXAMPLE and c == "recommended":
                cellwise[e] = (o0[pure] / (o0 + s0)[pure], o1[pure] / (o1 + s1)[pure])
                umap_frac[e] = np.where(pure, o1 / (o1 + s1), np.nan)
        d = abs(res["py"][1][c] - res["R"][1][c])
        row[f"maxdiff_{c}"] = d.max()
        row[f"same_profile_{c}"] = np.array_equal(res["py"][2][c], res["R"][2][c])
        row[f"entries_{c}"] = res["R"][1][c].nnz
        row[f"changed_{c}"] = int((res["R"][1][c] != toc).nnz)
    for c in CFG:
        sel = (res["R"][2][c] > 0) | (res["py"][2][c] > 0)
        prof_pairs.append(np.column_stack([res["R"][2][c][sel], res["py"][2][c][sel]]))
    row["same_describe"] = res["R"][3].equals(res["py"][3].set_axis(res["R"][3].index))
    descs[t] = (res["R"][3], res["py"][3])
    if t == EXAMPLE:                                              # UMAP of the cells, for display only
        umap_frac["before"] = np.where(pure, o0 / (o0 + s0), np.nan)
        umap_species = np.where(pd.isna(sp_of), "mixed", sp_of)
        ad = scanpy.AnnData(sp.csr_matrix(toc.T).astype(np.float32))
        scanpy.pp.normalize_total(ad, target_sum=1e4)
        scanpy.pp.log1p(ad)
        scanpy.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
        scanpy.pp.pca(ad, n_comps=30, random_state=0)
        scanpy.pp.neighbors(ad, n_neighbors=15, random_state=0)
        scanpy.tl.umap(ad, random_state=0)
        xy = ad.obsm["X_umap"]
    rows.append(row)
tab = pd.DataFrame(rows)
prof_pairs = np.vstack(prof_pairs)
for c in CFG:                                                     # as % of all counts in single-species cells
    tab[f"remc_{c}"] = tab[f"removed_{c}_R"] * tab.other_share * 100
    tab[f"lostc_{c}"] = tab[f"lost_{c}_R"] * (1 - tab.other_share) * 100
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95], hspace=0.55)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def crop(img):
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.55)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pyfastcar_logo.png", "omicverse_logo.png"]]
x = np.arange(len(tab))

axa = fig.add_subplot(top[1])
w = 0.26
axa.bar(x - w, tab.other_share * 100, w * 0.9, color="#c9c8c3", label="contamination present")
axa.bar(x, tab.remc_recommended, w * 0.9, color="#2f7f6f", label="contamination removed")
axa.bar(x + w, tab.lostc_recommended, w * 0.9, color="#b5a642", label="own RNA removed")
axa.set_xticks(x, tab.label, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axa.set_ylim(0, max(tab.other_share.max() * 100, tab.lostc_recommended.max()) * 1.45)
axa.set_ylabel("% of counts in cells")
axa.set_title("Recommended cutoff")
axa.grid(axis="y", color=GRID, lw=0.5)
axa.set_axisbelow(True)
axa.legend(fontsize=5, loc="upper left", handlelength=1, handletextpad=0.3, borderaxespad=0.1)
panel(axa, "a")

axb = fig.add_subplot(top[2])
dR, dP = descs[EXAMPLE]
axb.plot(dR.index, dR.genesContaminating, color=C_R, lw=2.2, label="FastCAR (R)")
axb.plot(dP.index, dP.genesContaminating, color=C_PY, lw=0.8, label="pyfastcar")
ex = tab.set_index("dataset").loc[EXAMPLE]
axb.axvline(ex.cutoff_R, color=INK2, lw=0.5, ls=":")
axb.text(ex.cutoff_R, axb.get_ylim()[1] * 0.5, f" recommended\n cutoff {ex.cutoff_R}", fontsize=5, color=INK2,
         ha="right", va="center")
axb.set_xlabel("empty-droplet cutoff (UMI)")
axb.set_ylabel("genes to correct")
axb.set_title(f"Cutoff scan ({LABEL[EXAMPLE].split(' (')[0]})")
axb.grid(color=GRID, lw=0.5)
axb.set_axisbelow(True)
axb.legend(fontsize=5, loc="upper left", handletextpad=0.3, borderaxespad=0.1)
panel(axb, "b")

axc = fig.add_subplot(top[3])
lim = prof_pairs.max() * 1.5
axc.plot([0.8, lim], [0.8, lim], color=GRID, lw=0.6, zorder=0)
axc.scatter(prof_pairs[:, 0], prof_pairs[:, 1], s=14, color=C_R, edgecolor="white", linewidth=0.3,
            label="FastCAR (R)")
axc.scatter(prof_pairs[:, 0], prof_pairs[:, 1], s=3, color=C_PY, lw=0, label="pyfastcar")
axc.set_xscale("log")
axc.set_yscale("log")
axc.set_xlim(0.8, lim)
axc.set_ylim(0.8, lim)
axc.set_aspect("equal")
axc.set_xlabel("FastCAR (R)")
axc.set_ylabel("pyfastcar")
axc.set_title("Counts subtracted per gene")
axc.text(0.96, 0.04, f"{len(prof_pairs)} genes x cutoffs\nidentical", transform=axc.transAxes, ha="right",
         va="bottom", fontsize=5, color=INK2)
panel(axc, "c")

bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3)
axd = fig.add_subplot(bot[0])
for c, mk, name in [("fixed", "o", "cutoff 100"), ("recommended", "D", "recommended cutoff")]:
    axd.scatter(tab.droplets, tab[f"sec_{c}_R"], s=12, marker=mk, color=C_R, edgecolor="white", linewidth=0.3,
                label=f"FastCAR, {name}")
    axd.scatter(tab.droplets, tab[f"sec_{c}_py"], s=12, marker=mk, color=C_PY, edgecolor="white", linewidth=0.3,
                label=f"pyfastcar, {name}")
    for _, r in tab.iterrows():
        axd.plot([r.droplets] * 2, [r[f"sec_{c}_py"], r[f"sec_{c}_R"]], color=GRID, lw=0.8, zorder=0)
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xlabel("droplets in the raw matrix")
axd.set_yticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"])
axd.minorticks_off()
axd.set_xticks([4e5, 7e5, 1e6, 2e6], ["0.4M", "0.7M", "1M", "2M"])
axd.set_ylabel("wall time, 1 thread (s)")
axd.grid(color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.set_ylim(0.04, 200)
axd.legend(fontsize=5, loc="lower right", handletextpad=0.1, ncol=2, columnspacing=0.6)
axd.set_title("Run time, profile to corrected matrix")
panel(axd, "d", dx=-0.2)

axe = fig.add_subplot(bot[1])
before, after_R = cellwise["R"]
_, after_py = cellwise["py"]
axe.scatter(before * 100, after_R * 100, s=1.5, color=C_R, lw=0, rasterized=True, label="FastCAR (R)")
axe.scatter(before * 100, after_py * 100, s=0.4, color=C_PY, lw=0, rasterized=True, label="pyfastcar")
top_ = np.nanpercentile(before * 100, 99.5)
axe.plot([0, top_], [0, top_], color=INK2, lw=0.6, zorder=0)
axe.set_xlim(0, top_)
axe.set_ylim(0, top_)
axe.set_xlabel("other-species counts before correction (% of cell)")
axe.set_ylabel("after correction (%)")
axe.set_title(f"Per cell, {LABEL[EXAMPLE]}, recommended cutoff")
axe.grid(color=GRID, lw=0.5)
axe.set_axisbelow(True)
leg = axe.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
for hh in leg.legend_handles:
    hh.set_sizes([10])
panel(axe, "e", dx=-0.2)

row3 = gs[2].subgridspec(1, 4, wspace=0.06)


def umap_ax(k, title, letter):
    ax = fig.add_subplot(row3[k])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")
    panel(ax, letter, dx=-0.04)
    return ax


ax = umap_ax(0, f"{LABEL[EXAMPLE]}: species", "f")
for name, col in [("human", "#2f7f6f"), ("mouse", "#b5a642"), ("mixed", "#2a2a2a")]:
    m = umap_species == name
    ax.scatter(xy[m, 0], xy[m, 1], s=0.6 if name != "mixed" else 1.2, c=col, lw=0, rasterized=True,
               label=f"{name} ({m.sum():,})")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=5.5, handletextpad=0.1,
                columnspacing=0.6)
for hh in leg.legend_handles:
    hh.set_sizes([10])
vmax = np.nanpercentile(umap_frac["before"] * 100, 99)
for k, (key, title, letter) in enumerate([("before", "Before correction", "g"), ("R", "After, FastCAR (R)", "h"),
                                          ("py", "After, pyfastcar", "i")]):
    ax = umap_ax(k + 1, title, letter)
    v = umap_frac[key] * 100
    nan = np.isnan(v)
    ax.scatter(xy[nan, 0], xy[nan, 1], s=0.6, c="#d9d8d3", lw=0, rasterized=True)
    o = np.flatnonzero(~nan)[np.argsort(v[~nan], kind="stable")]
    im = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=0.6, cmap="magma_r", vmin=0, vmax=vmax, lw=0, rasterized=True)
cax = fig.axes[-2].inset_axes([0.05, -0.14, 0.9, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
cax.set_title("other-species counts (% of cell; mixed cells grey)", fontsize=5.5, color=INK2, pad=2)

pa, pd_ = axa.get_position(), axd.get_position()
x0, x1 = pd_.x0 - 0.2 * pd_.width, pa.x0 - 0.075
y0, y1 = pa.y0 - 0.02, pa.y1 + 0.035
h_ov = (x1 - x0) * 7.2 / 6.5 * logos[1].shape[0] / logos[1].shape[1]
for img, rect, anchor in [(logos[0], [x0, y0 + h_ov + 0.012, x1 - x0, y1 - y0 - h_ov - 0.012], "N"),
                          (logos[1], [x0, y0, x1 - x0, h_ov], "S")]:
    axl = fig.add_axes(rect)
    axl.imshow(img)
    axl.set_anchor(anchor)
    axl.set_axis_off()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
sf = tab.sec_fixed_R / tab.sec_fixed_py
sr = tab.sec_recommended_R / tab.sec_recommended_py
big = tab.iloc[tab.droplets.idxmax()]
macros = {
    "nDatasets": len(tab), "nCellsTotal": f"{int(tab.cells.sum()):,}",
    "nCellsMin": f"{int(tab.cells.min()):,}", "nCellsMax": f"{int(tab.cells.max()):,}",
    "nDropMin": f"{tab.droplets.min() / 1e6:.2f}", "nDropMax": f"{tab.droplets.max() / 1e6:.2f}",
    "nSameDescribe": int(tab.same_describe.sum()), "nSameCutoff": int((tab.cutoff_R == tab.cutoff_py).sum()),
    "nSameProfile": int((tab.same_profile_fixed & tab.same_profile_recommended).sum()),
    "maxDiff": f"{max(tab.maxdiff_fixed.max(), tab.maxdiff_recommended.max()):g}",
    "cutoffMin": int(tab.cutoff_R.min()), "cutoffMax": int(tab.cutoff_R.max()),
    "genesFixedMin": int(tab.genes_fixed.min()), "genesFixedMax": int(tab.genes_fixed.max()),
    "genesRecMin": int(tab.genes_recommended.min()), "genesRecMax": int(tab.genes_recommended.max()),
    "otherMin": f"{tab.other_share.min() * 100:.1f}", "otherMax": f"{tab.other_share.max() * 100:.1f}",
    "remFixedMax": f"{tab.removed_fixed_R.max() * 100:.1f}",
    "remRecMin": f"{tab.removed_recommended_R.min() * 100:.1f}",
    "remRecMax": f"{tab.removed_recommended_R.max() * 100:.1f}",
    "lostRecMax": f"{tab.lost_recommended_R.max() * 100:.2f}",
    "sFixMin": f"{sf.min():.1f}", "sFixMax": f"{sf.max():.1f}",
    "sRecMin": f"{sr.min():.1f}", "sRecMax": f"{sr.max():.1f}",
    "tBigRecR": f"{big.sec_recommended_R:.0f}", "tBigRecPy": f"{big.sec_recommended_py:.1f}", "bigName": big.label,
    "exampleName": LABEL[EXAMPLE], "exGenes": int(ex.genes_recommended), "nProfile": len(prof_pairs),
    "remcMax": f"{tab.remc_recommended.max():.2f}", "lostcMax": f"{tab.lostc_recommended.max():.1f}",
    "exRemc": f"{ex.remc_recommended:.2f}", "exLostc": f"{ex.lostc_recommended:.1f}",
    "exOther": f"{ex.other_share * 100:.1f}",
    "exCutoff": int(ex.cutoff_R),
}
for key, name in [("OneK", "hgmm_1k_v3"), ("FiveK", "hgmm_5k_v3"), ("SixK", "hgmm_6k"), ("TenK", "hgmm_10k_v3"),
                  ("TwelveK", "hgmm_12k")]:
    r_ = tab.set_index("dataset").loc[name]
    macros[f"rem{key}"] = f"{r_.removed_recommended_R * 100:.1f}"
    macros[f"cutoff{key}"] = int(r_.cutoff_R)
    macros[f"lostc{key}"] = f"{r_.lostc_recommended:.2f}"
    macros[f"remc{key}"] = f"{r_.remc_recommended:.2f}"
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
print(tab.drop(columns=["label"]).round(5).T.to_string())
