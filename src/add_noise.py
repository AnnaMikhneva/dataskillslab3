#!/usr/bin/env python3
"""
Stage 2 – add_noise
====================
Reads a clean manifest, generates noisy variants of every utterance at the
SNR levels specified in params.yaml, writes the noisy WAV files, and emits
one manifest per SNR level – all atomically.

Usage:
    python src/add_noise.py --lang en --params params.yaml

Inputs:
    - data/manifests/{lang}/clean.jsonl
    - params.yaml

Outputs:
    - data/noisy/{lang}/snr_{snr_db}/wav/*.wav
    - data/manifests/{lang}/noisy_snr_{snr_db}.jsonl   (written atomically)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("add_noise")


# ---------------------------------------------------------------------------
# Noise addition (provided by lab)
# ---------------------------------------------------------------------------

def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def add_noise_to_file(
    input_wav: str,
    output_wav: str,
    snr_db: float,
    seed: int | None = None,
) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_atomically(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".tmp_manifest_",
        suffix=".jsonl",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_path)
        log.info("Manifest written atomically: %s (%d records)", output_path, len(records))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add noise to audio at multiple SNR levels.")
    parser.add_argument("--lang", required=True, help="Language code (e.g. en, fr)")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    lang = args.lang
    snr_levels = params["noise"]["snr_levels"]
    global_seed = params["noise"]["seed"]
    manifest_dir = Path(params["data"]["manifest_dir"]) / lang
    noisy_dir = Path(params["data"]["noisy_dir"]) / lang

    clean_manifest = manifest_dir / "clean.jsonl"
    if not clean_manifest.exists():
        log.error("Clean manifest not found: %s", clean_manifest)
        sys.exit(1)

    records = load_manifest(clean_manifest)
    log.info("Loaded %d utterances from %s", len(records), clean_manifest)

    for snr_db in snr_levels:
        snr_tag = f"snr_{snr_db:+.0f}".replace("+", "").replace("-", "neg")
        # Use a tag like snr_5, snr_neg5 for directories
        snr_wav_dir = noisy_dir / f"snr_{snr_db}" / "wav"
        snr_wav_dir.mkdir(parents=True, exist_ok=True)

        noisy_records = []
        errors = 0

        for rec in records:
            utt_id = rec["utt_id"]
            input_wav = rec["wav_path"]
            stem = Path(input_wav).stem

            output_wav = str(snr_wav_dir / f"{stem}.wav")

            # Deterministic per-utterance seed: combine global seed + hash of utt_id
            utt_seed = global_seed ^ (hash(utt_id) & 0xFFFFFFFF)

            try:
                add_noise_to_file(input_wav, output_wav, snr_db, seed=utt_seed)
            except Exception as e:
                log.error("Failed to add noise to %s at SNR %s dB: %s", utt_id, snr_db, e)
                errors += 1
                continue

            # Compute checksum of noisy file
            audio_md5 = md5_file(output_wav)

            noisy_rec = dict(rec)  # copy all fields from clean
            noisy_rec["wav_path"] = output_wav
            noisy_rec["snr_db"] = snr_db
            noisy_rec["audio_md5"] = audio_md5
            # utt_id stays the same (invariant across variants)
            noisy_records.append(noisy_rec)

        if errors > 0:
            log.warning("SNR=%s dB: %d errors encountered", snr_db, errors)

        # Write manifest atomically only after all files are produced
        manifest_out = manifest_dir / f"noisy_snr_{snr_db}.jsonl"
        write_manifest_atomically(noisy_records, manifest_out)

    log.info("Done. Noisy manifests written for %d SNR levels.", len(snr_levels))


def load_params(params_path: str) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
