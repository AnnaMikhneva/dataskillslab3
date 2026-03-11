#!/usr/bin/env python3


import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml


def load_params(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dvc_stages(params: dict) -> dict:
    """
    Build the complete DVC stages dict for all languages and SNR levels.
    """
    languages = params["languages"]
    snr_levels = params["noise"]["snr_levels"]
    stages = {}

    for lang in languages:

        # 1: prepare_manifests_{lang}

        stages[f"prepare_manifests_{lang}"] = {
            "cmd": f"python3 src/prepare_manifests.py --lang {lang} --params params.yaml",
            "deps": [
                "src/prepare_manifests.py",
                f"data/raw/{lang}/wav",
                f"data/raw/{lang}/text.tsv",
            ],
            "params": [
                {"params.yaml": ["languages", "data"]},
            ],
            "outs": [
                f"data/manifests/{lang}/clean.jsonl",
            ],
        }

        # 2: add_noise_{lang}

        noisy_manifests = [
            f"data/manifests/{lang}/noisy_snr_{snr_db}.jsonl"
            for snr_db in snr_levels
        ]
        noisy_audio_dirs = [
            f"data/noisy/{lang}/snr_{snr_db}/wav"
            for snr_db in snr_levels
        ]

        stages[f"add_noise_{lang}"] = {
            "cmd": f"python3 src/add_noise.py --lang {lang} --params params.yaml",
            "deps": [
                "src/add_noise.py",
                f"data/manifests/{lang}/clean.jsonl",
            ],
            "params": [
                {"params.yaml": ["noise", "data"]},
            ],
            "outs": noisy_manifests + noisy_audio_dirs,
        }

        # 3a: predict_clean_{lang}

        stages[f"predict_clean_{lang}"] = {
            "cmd": (
                f"python3 src/predict_phonemes.py "
                f"--manifest data/manifests/{lang}/clean.jsonl "
                f"--output data/predictions/{lang}/clean_pred.jsonl "
                f"--params params.yaml"
            ),
            "deps": [
                "src/predict_phonemes.py",
                f"data/manifests/{lang}/clean.jsonl",
            ],
            "params": [
                {"params.yaml": ["model"]},
            ],
            "outs": [
                f"data/predictions/{lang}/clean_pred.jsonl",
            ],
        }


        # 3b: predict_noisy_{lang}_{snr_db}  (one per SNR level)

        for snr_db in snr_levels:
            stages[f"predict_noisy_{lang}_snr{snr_db}"] = {
                "cmd": (
                    f"python3 src/predict_phonemes.py "
                    f"--manifest data/manifests/{lang}/noisy_snr_{snr_db}.jsonl "
                    f"--output data/predictions/{lang}/noisy_snr_{snr_db}_pred.jsonl "
                    f"--params params.yaml"
                ),
                "deps": [
                    "src/predict_phonemes.py",
                    f"data/manifests/{lang}/noisy_snr_{snr_db}.jsonl",
                ],
                "params": [
                    {"params.yaml": ["model"]},
                ],
                "outs": [
                    f"data/predictions/{lang}/noisy_snr_{snr_db}_pred.jsonl",
                ],
            }

        # 4a: evaluate_clean_{lang}

        stages[f"evaluate_clean_{lang}"] = {
            "cmd": (
                f"python3 src/evaluate.py "
                f"--manifest data/predictions/{lang}/clean_pred.jsonl "
                f"--output metrics/{lang}/clean.json "
                f"--snr null "
                f"--lang {lang} "
                f"--params params.yaml"
            ),
            "deps": [
                "src/evaluate.py",
                f"data/predictions/{lang}/clean_pred.jsonl",
            ],
            "params": [
                {"params.yaml": ["data"]},
            ],
            "metrics": [
                {f"metrics/{lang}/clean.json": {"cache": False}},
            ],
        }

        # 4b: evaluate_noisy_{lang}_{snr_db}

        for snr_db in snr_levels:
            stages[f"evaluate_noisy_{lang}_snr{snr_db}"] = {
                "cmd": (
                    f"python3 src/evaluate.py "
                    f"--manifest data/predictions/{lang}/noisy_snr_{snr_db}_pred.jsonl "
                    f"--output metrics/{lang}/snr_{snr_db}.json "
                    f"--snr {snr_db} "
                    f"--lang {lang} "
                    f"--params params.yaml"
                ),
                "deps": [
                    "src/evaluate.py",
                    f"data/predictions/{lang}/noisy_snr_{snr_db}_pred.jsonl",
                ],
                "params": [
                    {"params.yaml": ["data"]},
                ],
                "metrics": [
                    {f"metrics/{lang}/snr_{snr_db}.json": {"cache": False}},
                ],
            }

    # 5: plot_results (all languages)

    all_clean_metrics = [f"metrics/{lang}/clean.json" for lang in languages]
    all_noisy_metrics = [
        f"metrics/{lang}/snr_{snr_db}.json"
        for lang in languages
        for snr_db in snr_levels
    ]

    stages["plot_results"] = {
        "cmd": "python3 src/plot_results.py --params params.yaml",
        "deps": [
            "src/plot_results.py",
        ] + all_clean_metrics + all_noisy_metrics,
        "params": [
            {"params.yaml": ["languages", "noise", "data"]},
        ],
        "metrics": [
            {"metrics/summary.json": {"cache": False}},
        ],
        "plots": [
            {"figures/cross_language_mean.png": {"cache": False}},
        ]
        + [
            {f"figures/{lang}/per_vs_snr.png": {"cache": False}}
            for lang in languages
        ],
    }

    return stages


def write_dvc_yaml_atomically(stages: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".tmp_dvc_",
        suffix=".yaml",
    )
    header = (
        "# ============================================================\n"
        "# dvc.yaml – AUTO-GENERATED by generate_dvc_yaml.py\n"
        "# DO NOT EDIT MANUALLY. Regenerate with:\n"
        "#   python3 generate_dvc_yaml.py --params params.yaml\n"
        "# ============================================================\n\n"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(
                {"stages": stages},
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        os.replace(tmp_path, output_path)
        print(f"dvc.yaml written atomically to {output_path}")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate dvc.yaml from params.yaml.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="dvc.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    stages = build_dvc_stages(params)
    write_dvc_yaml_atomically(stages, Path(args.output))

    n_stages = len(stages)
    print(f"Generated {n_stages} DVC stages for languages: {params['languages']}")


if __name__ == "__main__":
    main()
