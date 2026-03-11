#!/usr/bin/env python3


import argparse
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("plot_results")

FIGSIZE = (8, 5)
DPI = 150
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_params(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json_atomically(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".tmp_summary_",
        suffix=".json",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, output_path)
        log.info("Summary written atomically: %s", output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_figure_atomically(fig, output_path: Path, dpi: int = DPI) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".tmp_fig_",
        suffix=".png",
    )
    os.close(tmp_fd)
    try:
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
        os.replace(tmp_path, output_path)
        log.info("Figure saved atomically: %s", output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise



def collect_metrics(params: dict) -> dict:

    metrics_dir = Path(params["data"]["metrics_dir"])
    languages = params["languages"]
    snr_levels = params["noise"]["snr_levels"]

    data = {}  # lang -> list of (snr_db, per_mean, per_std)

    for lang in languages:
        lang_data = []

        # Clean condition
        clean_file = metrics_dir / lang / "clean.json"
        if clean_file.exists():
            m = load_json(clean_file)
            lang_data.append((None, m["per_mean"], m.get("per_std", 0.0)))
        else:
            log.warning("Missing clean metrics for lang=%s", lang)

        # Noisy conditions
        for snr_db in sorted(snr_levels):
            noisy_file = metrics_dir / lang / f"snr_{snr_db}.json"
            if noisy_file.exists():
                m = load_json(noisy_file)
                lang_data.append((snr_db, m["per_mean"], m.get("per_std", 0.0)))
            else:
                log.warning("Missing metrics for lang=%s snr=%s", lang, snr_db)

        data[lang] = lang_data

    return data


def plot_language(lang: str, lang_data: list, output_path: Path) -> None:

    # Separating clean from noisy
    noisy = [(snr, per, std) for snr, per, std in lang_data if snr is not None]
    clean = [(snr, per, std) for snr, per, std in lang_data if snr is None]

    snrs = [x[0] for x in noisy]
    pers = [x[1] for x in noisy]
    stds = [x[2] for x in noisy]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(snrs, pers, marker="o", color=COLORS[0], linewidth=2, label=f"{lang} PER")
    ax.fill_between(
        snrs,
        [p - s for p, s in zip(pers, stds)],
        [p + s for p, s in zip(pers, stds)],
        alpha=0.2,
        color=COLORS[0],
    )

    # Adding clean baseline as a horizontal dashed line
    if clean:
        clean_per = clean[0][1]
        ax.axhline(
            clean_per,
            color="grey",
            linestyle="--",
            linewidth=1.5,
            label=f"Clean baseline ({clean_per:.3f})",
        )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("PER", fontsize=12)
    ax.set_title(f"Phoneme Error Rate vs Noise Level – {lang.upper()}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    save_figure_atomically(fig, output_path)
    plt.close(fig)


def plot_cross_language(data: dict, output_path: Path) -> None:
    """
    Plot per-language curves + cross-language mean PER vs SNR.
    Only SNR levels present in all languages are included in the mean.
    """
    # Finding common SNR levels (excluding clean)
    snr_sets = []
    for lang, lang_data in data.items():
        snr_sets.append(set(snr for snr, _, _ in lang_data if snr is not None))
    if not snr_sets:
        log.warning("No data to plot for cross-language figure.")
        return
    common_snrs = sorted(snr_sets[0].intersection(*snr_sets[1:]) if len(snr_sets) > 1 else snr_sets[0])

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Per-language curves
    for i, (lang, lang_data) in enumerate(data.items()):
        snr_to_per = {snr: per for snr, per, _ in lang_data if snr is not None}
        xs = [snr for snr in common_snrs if snr in snr_to_per]
        ys = [snr_to_per[snr] for snr in xs]
        ax.plot(xs, ys, marker="s", linewidth=1.5, alpha=0.7,
                color=COLORS[i % len(COLORS)], label=lang)

    # Cross-language mean
    mean_pers = []
    for snr in common_snrs:
        vals = []
        for lang_data in data.values():
            snr_to_per = {s: p for s, p, _ in lang_data if s is not None}
            if snr in snr_to_per:
                vals.append(snr_to_per[snr])
        mean_pers.append(sum(vals) / len(vals) if vals else float("nan"))

    ax.plot(
        common_snrs, mean_pers,
        marker="D", linewidth=2.5, linestyle="-",
        color="black", label="Cross-language mean",
    )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("PER", fontsize=12)
    ax.set_title("PER vs Noise Level – All Languages", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    save_figure_atomically(fig, output_path)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="Plot PER vs SNR results.")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    languages = params["languages"]
    figures_dir = Path(params["data"]["figures_dir"])
    metrics_dir = Path(params["data"]["metrics_dir"])

    # Collecting all metrics
    data = collect_metrics(params)

    if not any(data.values()):
        log.error("No metrics found. Run evaluate stage first.")
        sys.exit(1)

    # Per-language plots
    for lang in languages:
        if lang not in data or not data[lang]:
            log.warning("No data for language %s, skipping plot.", lang)
            continue
        plot_language(lang, data[lang], figures_dir / lang / "per_vs_snr.png")

    # Cross-language mean plot
    active_data = {lang: d for lang, d in data.items() if d}
    if len(active_data) >= 1:
        plot_cross_language(active_data, figures_dir / "cross_language_mean.png")

    # Summary metrics (DVC-trackable)
    summary = {}
    for lang, lang_data in data.items():
        summary[lang] = [
            {"snr_db": snr, "per_mean": per, "per_std": std}
            for snr, per, std in lang_data
        ]

    write_json_atomically(summary, metrics_dir / "summary.json")
    log.info("Plotting complete.")


if __name__ == "__main__":
    main()
