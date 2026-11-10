#!/usr/bin/env python3
"""PHASE-WATER Phase 4A: reproducible full-footprint cartographic figures.

Reads existing Phase 3G GeoTIFFs only. No classification, area estimation,
calibration, or reference validation is performed by this script.

Usage (from D:\\Hackathon):
  .\\.venv\\Scripts\\python.exe phase4a_final_nisar_map_figures.py --mode preflight
  .\\.venv\\Scripts\\python.exe phase4a_final_nisar_map_figures.py --mode run

Requires: numpy, rasterio, matplotlib. Outputs 350-dpi PNG, PDF and markdown/CSV
under outputs/phase4a_high_quality.  Spatial data remain in the original Phase 3G GeoTIFFs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

BASE = Path("outputs/phase3g")
DEST = Path("outputs/phase4a_high_quality")
SOURCES = {
    "L_aug": "phase3g_aug_L_only_water.tif",
    "S_aug": "phase3g_aug_S_only_water.tif",
    "F_aug": "phase3g_aug_LS_equalweight_water.tif",
    "L_sep": "phase3g_sep_L_only_water.tif",
    "S_sep": "phase3g_sep_S_only_water.tif",
    "F_sep": "phase3g_sep_LS_equalweight_water.tif",
    "L_change": "phase3g_L_only_change_4class.tif",
    "S_change": "phase3g_S_only_change_4class.tif",
    "F_change": "phase3g_LS_equalweight_change_4class.tif",
    "disagreement": "phase3g_L_vs_S_change_disagreement.tif",
    "joint": "phase3g_four_scene_joint_valid.tif",
}
WATER_COLORS = ["#e9e5dd", "#176aaf", "#ffffff"]
CHANGE_COLORS = ["#d9d6cf", "#126caa", "#13a381", "#e7943c", "#ffffff"]
DISAGREE_COLORS = ["#d9d6cf", "#af3a73", "#ffffff"]
WATER_LEGEND = [Patch(facecolor=WATER_COLORS[0], label="Classified non-water"),
                Patch(facecolor=WATER_COLORS[1], label="Classified water"),
                Patch(facecolor="white", edgecolor="#666", label="Invalid / no joint coverage")]
CHANGE_LEGEND = [Patch(facecolor=CHANGE_COLORS[0], label="Stable non-water"),
                 Patch(facecolor=CHANGE_COLORS[1], label="Stable water"),
                 Patch(facecolor=CHANGE_COLORS[2], label="Newly classified water"),
                 Patch(facecolor=CHANGE_COLORS[3], label="Classified water loss"),
                 Patch(facecolor="white", edgecolor="#666", label="Invalid")]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    ap.add_argument("--input-dir", type=Path, default=BASE)
    ap.add_argument("--output-dir", type=Path, default=DEST)
    ap.add_argument("--max-dim", type=int, default=3000,
                    help="Maximum width/height of preview per panel (native TIFs untouched)")
    ap.add_argument("--dpi", type=int, default=350,
                    help="High-resolution output DPI, default 350 (maximum 500)")
    return ap.parse_args()


def audit(srcdir):
    reportpath = srcdir / "phase3g_report.json"
    if not reportpath.is_file():
        raise FileNotFoundError(f"Missing {reportpath}")
    report = json.loads(reportpath.read_text(encoding="utf-8-sig"))
    res = report.get("full_scene_results", report)
    total = int(res["four_scene_joint_valid_pixels"])
    if total <= 0:
        raise ValueError("The Phase 3G reported joint valid total is zero")
    classes = ("stable_nonwater", "stable_water", "water_gain", "water_loss")
    stats = {}
    for key in ("L", "S", "F"):
        d = res["models"][key]
        counts = d["classes_pixels"]
        if sum(int(counts[c]) for c in classes) != total:
            raise ValueError(f"{key} native class pixel counts do not sum to four-scene valid total")
        stats[key] = d
    table = res.get("L_vs_S_change_class_contingency_pixels")
    if table is not None and sum(int(x) for x in table.values()) != total:
        raise ValueError("L/S change contingency total differs from four-scene valid total")
    metadata = {}
    for k, filename in SOURCES.items():
        path = srcdir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 3G output: {path}")
        with rasterio.open(path) as ds:
            if ds.count < 1:
                raise ValueError(f"No data bands in {path}")
            metadata[k] = {
                "crs": ds.crs.to_string() if ds.crs else None,
                "shape": (ds.height, ds.width),
                "transform": tuple(ds.transform),
                "bounds": tuple(ds.bounds),
                "dtype": ds.dtypes[0],
                "nodata": ds.nodata,
            }
    ref = metadata["L_change"]
    for k, m in metadata.items():
        if m["crs"] != ref["crs"] or m["shape"] != ref["shape"] or not np.allclose(m["transform"], ref["transform"], atol=1.e-8, rtol=0):
            raise ValueError(f"Grid mismatch: {k}")
    if ref["crs"] != "EPSG:32645":
        raise ValueError("Unexpected CRS: expected the Phase 3G UTM 45N products")
    if not np.isclose(ref["transform"][0], 10.0) or not np.isclose(ref["transform"][4], -10.0):
        raise ValueError("Expected 10 m pixel grid")
    return report, res, stats, metadata


def preview(srcdir, key, metadata, max_dim):
    height, width = metadata[key]["shape"]
    factor = min(1., max_dim / max(height, width))
    h = max(1, int(round(height * factor)))
    w = max(1, int(round(width * factor)))
    with rasterio.open(srcdir / SOURCES[key]) as ds:
        a = ds.read(1, out_shape=(h, w), resampling=Resampling.nearest, masked=False)
        nodata = ds.nodata
        bounds = ds.bounds
    if nodata is not None and np.isfinite(nodata):
        a = np.where(a == nodata, 255, a)
    return a, (bounds.left/1000, bounds.right/1000, bounds.bottom/1000, bounds.top/1000)


def colored(a, kind):
    """Map fixed class codes into discrete palette indices, invalid -> white."""
    b = np.full(a.shape, 255, dtype=np.uint8)
    if kind == "water":
        for v in (0, 1):
            b[a == v] = v
        b[b == 255] = 2
        return b, ListedColormap(WATER_COLORS), BoundaryNorm([-0.5, .5, 1.5, 2.5], 3)
    if kind == "change":
        for v in range(4):
            b[a == v] = v
        b[b == 255] = 4
        return b, ListedColormap(CHANGE_COLORS), BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5)
    if kind == "disagreement":
        for v in (0, 1):
            b[a == v] = v
        b[b == 255] = 2
        return b, ListedColormap(DISAGREE_COLORS), BoundaryNorm([-.5, .5, 1.5, 2.5], 3)
    raise ValueError(kind)


def draw_panel(ax, a, extent, kind, title, font=10):
    p, cmap, norm = colored(a, kind)
    ax.imshow(p, extent=extent, origin="upper", interpolation="nearest", cmap=cmap,
              norm=norm, rasterized=True)
    ax.set_title(title, fontsize=font, weight="bold", pad=7)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8, direction="out", width=.6, length=3)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(.75)
        spine.set_edgecolor("#59616c")
    return ax


def add_north(ax):
    # Explicitly GRID north; true north differs slightly due to UTM convergence.
    ax.annotate("", xy=(.90, .94), xytext=(.90, .81), xycoords="axes fraction",
                arrowprops={"arrowstyle":"-|>", "color":"#25384b", "lw":1.7})
    ax.text(.90, .95, "Grid N", transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="center", va="bottom", color="#25384b",
            bbox={"boxstyle":"round,pad=.18", "facecolor":"white", "alpha":.86, "edgecolor":"none"})


def add_scale(ax, extent, length_km=50):
    x1, x2, y1, y2 = extent
    length_km = min(float(length_km), (x2-x1)*.33)
    xx, yy = x1 + .06*(x2-x1), y1 + .07*(y2-y1)
    ax.plot([xx, xx+length_km], [yy, yy], color="white", lw=5, solid_capstyle="butt", zorder=10)
    ax.plot([xx, xx+length_km], [yy, yy], color="#142c3d", lw=2.7, solid_capstyle="butt", zorder=11)
    tick = .008 * (y2-y1)
    ax.plot([xx,xx],[yy-tick,yy+tick], color="#142c3d",lw=2,zorder=12)
    ax.plot([xx+length_km,xx+length_km],[yy-tick,yy+tick], color="#142c3d",lw=2,zorder=12)
    ax.text(xx+length_km/2, yy+.022*(y2-y1), f"{length_km:g} km (UTM)", fontsize=8,
            color="#142c3d", weight="bold", ha="center", va="bottom", zorder=13,
            bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1})


def save(fig, path, dpi):
    """High-resolution lossless PNG and scalable-type print PDF.

    Source map data are still categorical nearest-neighbour previews, never reclassified.
    The PDF keeps labels/vector elements crisp for print and presentation use.
    """
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=.18)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    print("Saved high-quality PNG:", path)
    print("Saved print-ready PDF:", pdf_path)


def make_panel_grid(srcdir, dest, metadata, max_dim, dpi):
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.titlesize":10, "axes.labelsize":9})
    fig, axs = plt.subplots(3, 3, figsize=(17.2, 15.9), constrained_layout=False)
    fig.subplots_adjust(left=.075, right=.975, top=.925, bottom=.135, hspace=.21, wspace=.12)
    fig.suptitle("NISAR L-band / S-band / equal-weight fusion | Bihar Ganga floodplain",
                 fontsize=17, weight="bold", y=.985)
    fig.text(.5, .954, "19 August 2026 vs 24 September 2026 | Full S-band scene rectangle | EPSG:32645 | 10 m native grid", 
             fontsize=10, ha="center", color="#485261")
    labels = {"L":"L only", "S":"S only", "F":"L+S (50:50 exploratory)"}
    ext = None
    for row, method in enumerate(("L","S","F")):
        for col, (suf, day, kind) in enumerate((("aug","19 August","water"),("sep","24 September","water"),("change","Aug–Sep change","change"))):
            key = f"{method}_{suf}"
            a, ext = preview(srcdir, key, metadata, max_dim)
            draw_panel(axs[row,col], a, ext, kind, f"{labels[method]} | {day}", 10.5)
            if row == 2:
                axs[row,col].set_xlabel("Easting (km, UTM 45N)", fontsize=9)
            else:
                axs[row,col].tick_params(labelbottom=False)
            if col == 0:
                axs[row,col].set_ylabel("Northing (km, UTM 45N)", fontsize=9)
            else:
                axs[row,col].tick_params(labelleft=False)
            del a
    add_north(axs[0,0]); add_scale(axs[0,0], ext)
    h1 = WATER_LEGEND[:2] + CHANGE_LEGEND[2:4] + [WATER_LEGEND[2]]
    fig.legend(handles=h1, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(.5,.072), fontsize=10)
    save(fig, dest/"phase4a_nisar_3x3_comparison.png", dpi)


def make_changes(srcdir, dest, metadata, max_dim, dpi):
    fig, axes = plt.subplots(1,3,figsize=(17.5,7.5))
    fig.subplots_adjust(left=.07,right=.985,top=.86,bottom=.20,wspace=.15)
    fig.suptitle("Provisional water change | 19 August to 24 September 2026",
                 fontsize=17,weight="bold", y=.98)
    fig.text(.5,.91,"NISAR full S-band footprint | Four-scene joint valid support: 57,660.12 km² | UTM zone 45N",ha="center",fontsize=10)
    ext=None
    for ax, key, label in zip(axes,("L_change","S_change","F_change"),
                              ("L-band only","S-band only","L+S equal-weight fusion")):
        a,ext=preview(srcdir,key,metadata,max_dim)
        draw_panel(ax,a,ext,"change",label,11.5)
        ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    add_north(axes[0]); add_scale(axes[0],ext)
    fig.legend(handles=CHANGE_LEGEND,loc="lower center",bbox_to_anchor=(.5,.09),
               frameon=False,ncol=5,fontsize=10)
    save(fig,dest/"phase4a_three_change_maps.png",dpi)


def make_disagreement(srcdir, dest, metadata, max_dim, dpi, res):
    a,ext = preview(srcdir,"disagreement",metadata,max_dim)
    fig,ax=plt.subplots(figsize=(9,9.3))
    fig.subplots_adjust(left=.13,right=.96,top=.88,bottom=.13)
    fig.suptitle("L-band versus S-band | Change-class disagreement",fontsize=16,weight="bold",y=.98)
    fig.text(.5,.925,"Four-class change assignments differ over 40.36% of jointly valid pixels",ha="center",fontsize=10)
    draw_panel(ax,a,ext,"disagreement","Four-scene comparable extent",11)
    add_north(ax); add_scale(ax,ext)
    ax.set_xlabel("Easting (km, UTM 45N)");ax.set_ylabel("Northing (km, UTM 45N)")
    handles=[Patch(facecolor=DISAGREE_COLORS[0],label="Same change class"),
             Patch(facecolor=DISAGREE_COLORS[1],label="Different change class"),
             Patch(facecolor="white",edgecolor="#666",label="Invalid")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.062),ncol=3,frameon=False,fontsize=9)
    save(fig,dest/"phase4a_change_disagreement.png",dpi)


def write_stats(dest, res):
    rows=[]
    for code, name in (("L","L only"),("S","S only"),("F","L+S 50:50")):
        d=res["models"][code]
        for k,v in d["classes_km2"].items():
            rows.append((name,k,round(float(v),4)))
        rows.append((name,"net_water_change_km2",round(float(d["net_water_change_km2"]),4)))
    p=dest/"phase4a_native_resolution_areas.csv"
    with p.open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f);wr.writerow(("method","measure","area_km2_from_phase3g_report"));wr.writerows(rows)
    notes = "# PHASE-WATER Phase 4A map captions / reporting safeguards\n\n"
    notes += "**Figure 1.** Full-footprint August/September provisional water classifications and four-class change for NISAR L-band, S-band and equal-weight L+S fusion. Data were processed on a shared 10 m UTM 45N grid and restricted to 576,601,215 jointly valid four-scene pixels (57,660.1215 km²). The method uses August-fixed, frequency-specific, exploratory HH/HV thresholds; these are not calibrated water probabilities. Invalid pixels are displayed in white.\n\n"
    notes += "**Figure 2.** Four-class water-change comparison for L-only, S-only and equal-weight dual-frequency models. Class 2 denotes newly *classified* water and class 3 denotes loss of *classified* water; neither is independent evidence of flooding or recession. The S-band September scene uses urgent-response (UR) rather than the August precise/reprocessed (PR) product, and its cross-pol HV lower tail remains unresolved.\n\n"
    notes += "**Figure 3.** Spatial distribution of L/S four-class change-assignment disagreement. The fraction of jointly valid pixels assigned different change classes is 40.363812%; this measures classification disagreement, not error rate. Spatial sample/validation design is required to determine which predictions are correct.\n\n"
    notes += "**Figure-design note:** Red caution footnotes have deliberately been removed from the three PNG/PDF maps for a clean layout. Scientific limitations are preserved in this accompanying captions document and should remain disclosed in slides/reports.\n\n"
    notes += "**Important notes:** Cartography is downsampled using nearest-neighbour solely for display. Reported area counts come only from native-resolution Phase 3G totals, NOT from rendered PNGs. UTM north arrows indicate grid north. Figures span the full selected S-band GCOV rectangle, *not* the full larger L-band rectangle; L-only full original extent is available separately in Phase 2A. No independent ground-truth water labels, accuracy estimates, calibrated fusion probabilities or verified flood extents are implied.\n"
    (dest/"phase4a_figure_captions_and_limitations.md").write_text(notes,encoding="utf-8")
    print("Saved:",p)
    print("Saved:",dest/"phase4a_figure_captions_and_limitations.md")


def main():
    a=parse_args()
    if a.max_dim < 400 or a.max_dim > 8000: raise ValueError("--max-dim must be 400..8000")
    if a.dpi < 60 or a.dpi > 500: raise ValueError("--dpi must be 60..500")
    report,res,stats,meta=audit(a.input_dir)
    print("PHASE-WATER 4A | Phase 3G inputs verified: 11 compatible 10 m GeoTIFFs")
    print("Native pixels:",res["four_scene_joint_valid_pixels"],"| area km²:",res["four_scene_joint_valid_km2"])
    print("Four-class L/S disagreement %:",round(100*res["L_vs_S_change_disagreement_fraction"],3))
    print("Data not modified; classification not rerun.")
    if a.mode=="preflight":
        print("PREFLIGHT PASS | run --mode run to export PNG figures and captions.")
        return
    a.output_dir.mkdir(parents=True,exist_ok=True)
    make_panel_grid(a.input_dir,a.output_dir,meta,a.max_dim,a.dpi)
    make_changes(a.input_dir,a.output_dir,meta,a.max_dim,a.dpi)
    make_disagreement(a.input_dir,a.output_dir,meta,a.max_dim,a.dpi,res)
    write_stats(a.output_dir,res)
    print("PHASE-WATER 4A CLEAN FIGURES COMPLETE | no red warning footers; see captions for caveats.")


if __name__ == "__main__":
    main()
