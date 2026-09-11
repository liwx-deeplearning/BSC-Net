"""Generate the retained structural and dynamic panel-C plots from numeric data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "quantitative_analysis"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "figures"
DEFAULT_MORPHOLOGY_DIR = RESULTS_ROOT / "morphology"
DEFAULT_HEMODYNAMICS_DIR = RESULTS_ROOT / "hemodynamics"

CASES = ("v09", "v11")
CASE_DISPLAY = {"v09": "V09", "v11": "V11"}
ROI_KEYS = ("proximal", "mid", "distal")
ROI_LABELS = {"proximal": "Proximal ROI", "mid": "Mid ROI", "distal": "Distal ROI"}
ROI_COLORS = {"proximal": "#D9A7B8", "mid": "#7FA7C9", "distal": "#79B8B1"}
ROI_MARKERS = {"proximal": "o", "mid": "s", "distal": "^"}
ROI_LINESTYLES = {"proximal": "-", "mid": "--", "distal": "-."}
DIAMETER_COLOR = "#7FA7C9"
REFERENCE_COLOR = "#79B8B1"
LESION_COLOR = "#D9A7B8"
MLD_COLOR = "#A5655D"
WASH_IN_COLOR = "#CFE8BF"

logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


def validate_structural_results(results: dict[str, dict[str, Any]]) -> None:
    for case, result in results.items():
        position = np.asarray(result["centerline_position"])
        diameter = np.asarray(result["diameter_smooth"])
        reference = np.asarray(result["d_ref_profile_px"])
        if position.ndim != 1 or diameter.shape != position.shape or reference.shape != position.shape:
            raise ValueError(f"{case}: structural profile arrays must be one-dimensional and equal length")
        if not np.isfinite(position).all() or not np.isfinite(diameter).all() or not np.isfinite(reference).all():
            raise ValueError(f"{case}: structural profile contains non-finite values")


def validate_dynamic_results(results: dict[str, dict[str, Any]]) -> None:
    for case, result in results.items():
        frames = np.asarray(result["frame_ids"])
        if frames.ndim != 1 or frames.size == 0:
            raise ValueError(f"{case}: frame_ids must be a non-empty one-dimensional array")
        for roi in ROI_KEYS:
            metric = result["tic"][roi]
            curve = np.asarray(metric["normalized"])
            if curve.shape != frames.shape or not np.isfinite(curve).all():
                raise ValueError(f"{case}/{roi}: TIC must be finite and match frame_ids")
            if int(metric["ttp"]) < int(metric["at"]):
                raise ValueError(f"{case}/{roi}: TTP must be at or after AT")


def set_paper_style(dpi: int = 600) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": False,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
        }
    )
    return font


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_structural_results(
    cases: list[str],
    morphology_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load precomputed QCA morphology profiles without recalculating vessel geometry."""
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = case.lower()
        case_dir = morphology_dir / key
        metrics = _read_json(case_dir / "metrics.json")
        with (case_dir / "diameter_profile.csv").open(
            newline="", encoding="utf-8-sig"
        ) as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"{key}: diameter_profile.csv is empty")
        lesions = list(metrics.get("lesions", []))
        primary = max(
            lesions,
            key=lambda item: float(item["maximum_diameter_stenosis_percent"]),
            default=None,
        )
        results[key] = {
            "case": CASE_DISPLAY.get(key, key.upper()),
            "centerline_position": np.asarray(
                [row["normalized_centerline_position"] for row in rows], dtype=np.float64
            ),
            "diameter_smooth": np.asarray(
                [row["apparent_diameter_pixel"] for row in rows], dtype=np.float32
            ),
            "d_ref_profile_px": np.asarray(
                [row["reference_diameter_pixel"] for row in rows], dtype=np.float64
            ),
            "lesions": lesions,
            "primary_lesion": primary,
        }
    validate_structural_results(results)
    return results


def load_dynamic_results(
    cases: list[str],
    hemodynamics_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load computed TICs and event parameters without reading rendered images."""
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = case.lower()
        metrics = _read_json(hemodynamics_dir / key / "metrics.json")
        tic: dict[str, dict[str, Any]] = {}
        for roi in ROI_KEYS:
            roi_metrics = metrics["metrics"][roi]
            tic[roi] = {
                "at": int(roi_metrics["AT"]),
                "ttp": int(roi_metrics["TTP"]),
                "normalized": np.asarray(metrics["curves"][f"{roi}_norm"], dtype=np.float32),
            }
        results[key] = {
            "case": CASE_DISPLAY.get(key, key.upper()),
            "frame_ids": np.asarray(metrics["frame_ids"], dtype=np.int32),
            "tic": tic,
        }
    validate_dynamic_results(results)
    return results


def _style_axis(ax: Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.85)
    ax.tick_params(width=0.85, length=3.0, color="black", labelcolor="black")


def plot_diameter_profile(ax: Axes, result: dict[str, Any], show_legend: bool) -> None:
    """Plot one structural panel directly from centerline-profile arrays."""
    # x/profile/reference shapes: (N_centerline_points,)
    x = np.asarray(result["centerline_position"], dtype=np.float64)
    profile = np.asarray(result["diameter_smooth"], dtype=np.float64)
    reference = np.asarray(result["d_ref_profile_px"], dtype=np.float64)
    ax.plot(x, profile, color=DIAMETER_COLOR, linewidth=1.7, label="Apparent lumen diameter")
    ax.plot(x, reference, color=REFERENCE_COLOR, linestyle="--", linewidth=1.3, label=r"Reference taper $D_{ref}(s)$")
    for lesion_index, lesion in enumerate(result["lesions"]):
        ax.axvspan(
            x[int(lesion["start_index"])],
            x[int(lesion["end_index"])],
            color=LESION_COLOR,
            alpha=0.20,
            linewidth=0,
            label="Lesion interval" if lesion_index == 0 else None,
        )
    primary = result["primary_lesion"]
    if primary is not None:
        ax.scatter(
            [x[int(primary["peak_index"])]],
            [float(primary["minimum_lumen_diameter_mld_pixel"])],
            color=MLD_COLOR,
            edgecolors="white",
            linewidths=0.5,
            s=32,
            zorder=5,
            label="MLD",
        )
    ax.set_title(f"{result['case']} lumen-diameter profile")
    ax.set_xlabel("Normalized centerline position")
    ax.set_ylabel("Apparent lumen diameter (px)")
    if show_legend:
        ax.legend(loc="upper right", frameon=False)
    _style_axis(ax)


def _event_index(frames: np.ndarray, event_frame: int) -> int:
    matches = np.flatnonzero(frames == int(event_frame))
    if matches.size == 0:
        raise ValueError(f"Event frame {event_frame} is absent from frame_ids")
    return int(matches[0])


def plot_tic_panel(ax: Axes, result: dict[str, Any], show_legend: bool) -> None:
    """Plot one dynamic panel directly from normalized TIC arrays and event values."""
    # frames/each normalized TIC shape: (N_frames,)
    frames = np.asarray(result["frame_ids"], dtype=np.int32)
    distal = result["tic"]["distal"]
    ax.axvspan(int(distal["at"]), int(distal["ttp"]), color=WASH_IN_COLOR, alpha=0.58, linewidth=0, zorder=0)
    for roi in ROI_KEYS:
        metric = result["tic"][roi]
        curve = np.asarray(metric["normalized"], dtype=np.float64)
        ax.plot(
            frames,
            curve,
            color=ROI_COLORS[roi],
            linestyle=ROI_LINESTYLES[roi],
            marker=ROI_MARKERS[roi],
            markevery=5,
            linewidth=1.5,
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=ROI_LABELS[roi],
        )
        at_index = _event_index(frames, int(metric["at"]))
        ttp_index = _event_index(frames, int(metric["ttp"]))
        ax.scatter([frames[at_index]], [curve[at_index]], s=28, marker="D", color="black", zorder=6)
        ax.scatter([frames[ttp_index]], [curve[ttp_index]], s=60, marker="*", color="black", zorder=7)
    ax.set_title(f"{result['case']} TICs")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized enhancement")
    ax.set_ylim(-0.04, 1.08)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(
            [
                Patch(facecolor=WASH_IN_COLOR, edgecolor="none", alpha=0.65),
                Line2D([0], [0], marker="D", color="black", linewidth=0, markersize=5),
                Line2D([0], [0], marker="*", color="black", linewidth=0, markersize=8),
            ]
        )
        labels.extend(["Distal wash-in interval", "AT", "TTP"])
        ax.legend(handles, labels, loc="lower right", frameon=False, ncol=2)
    _style_axis(ax)


def build_combined_panel_c(
    structural_results: dict[str, dict[str, Any]],
    dynamic_results: dict[str, dict[str, Any]],
    dpi: int = 600,
) -> Figure:
    """Build retained panels as two rows: structural above, TIC below."""
    validate_structural_results(structural_results)
    validate_dynamic_results(dynamic_results)
    cases = list(structural_results.keys())
    if cases != list(dynamic_results.keys()):
        raise ValueError("Structural and dynamic cases must have identical order")
    if len(cases) != 2:
        raise ValueError("The retained paper layout requires exactly two cases")

    set_paper_style(dpi)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), dpi=dpi, constrained_layout=False)
    for index, case in enumerate(cases):
        plot_diameter_profile(axes[0, index], structural_results[case], show_legend=index == 0)
        plot_tic_panel(axes[1, index], dynamic_results[case], show_legend=index == 0)
    fig.text(0.055, 0.975, "Fig. 4(c) Centerline-Based Apparent Lumen-Diameter Profiles", ha="left", va="top", fontsize=11, fontstyle="italic")
    fig.text(0.055, 0.505, "Fig. 5(c) Time-Intensity Curve Analysis", ha="left", va="top", fontsize=11, fontstyle="italic")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.08, wspace=0.24, hspace=0.55)
    return fig


def export_figure(fig: Figure, output_dir: Path, dpi: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Fig_panel_c_combined"
    paths = {suffix: output_dir / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png")}
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight")
    return {key: value.name for key, value in paths.items()}


def write_source_tables(
    output_dir: Path,
    structural_results: dict[str, dict[str, Any]],
    dynamic_results: dict[str, dict[str, Any]],
) -> dict[str, str]:
    structural_path = output_dir / "panel_c_structural_data.csv"
    with structural_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "normalized_centerline_position", "smoothed_diameter_px", "reference_diameter_px"])
        for case, result in structural_results.items():
            rows = zip(result["centerline_position"], result["diameter_smooth"], result["d_ref_profile_px"])
            writer.writerows((CASE_DISPLAY[case], float(s), float(d), float(ref)) for s, d, ref in rows)

    dynamic_path = output_dir / "panel_c_dynamic_data.csv"
    with dynamic_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "frame", "roi", "normalized_enhancement", "AT", "TTP"])
        for case, result in dynamic_results.items():
            for roi in ROI_KEYS:
                metric = result["tic"][roi]
                rows = zip(result["frame_ids"], metric["normalized"])
                writer.writerows(
                    (CASE_DISPLAY[case], int(frame), roi, float(value), int(metric["at"]), int(metric["ttp"]))
                    for frame, value in rows
                )
    return {
        "structural_csv": structural_path.name,
        "dynamic_csv": dynamic_path.name,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the retained structural and dynamic panel-C plots.")
    parser.add_argument("--fig", choices=["panel-c"], default="panel-c", help=argparse.SUPPRESS)
    parser.add_argument(
        "--morphology-dir",
        type=Path,
        default=DEFAULT_MORPHOLOGY_DIR,
        help="Directory produced by morphology/structural_analysis.py.",
    )
    parser.add_argument(
        "--hemodynamics-dir",
        type=Path,
        default=DEFAULT_HEMODYNAMICS_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    structural_results = load_structural_results(
        cases=list(CASES),
        morphology_dir=args.morphology_dir,
    )
    dynamic_results = load_dynamic_results(
        cases=list(CASES),
        hemodynamics_dir=args.hemodynamics_dir,
    )
    output_dir = args.output_dir.resolve()
    fig = build_combined_panel_c(structural_results, dynamic_results, dpi=args.dpi)
    outputs = export_figure(fig, output_dir, dpi=args.dpi)
    plt.close(fig)
    outputs.update(write_source_tables(output_dir, structural_results, dynamic_results))
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "numeric centerline-diameter profiles and normalized time-intensity curves",
                "layout": "top row: structural diameter profiles; bottom row: dynamic TICs",
                "cases": [CASE_DISPLAY[case] for case in CASES],
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path.name
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
