#!/usr/bin/env python3
"""
setup_data.py
=============
Downloads a small subset of the Mozilla Common Voice dataset for one or more
languages and organises it into the directory layout expected by the pipeline:

    data/raw/{lang}/wav/   ← mono 16 kHz WAV files
    data/raw/{lang}/text.tsv   ← tab-separated: stem<TAB>ref_text

Usage:
    python setup_data.py --langs en fr --n-utterances 50 --params params.yaml

Dependencies:
    pip install datasets soundfile numpy scipy

The script uses HuggingFace datasets to access Common Voice (mozilla-foundation/common_voice_11_0).
You may need to accept the terms on the HuggingFace Hub and be logged in:
    huggingface-cli login

Audio is converted to mono 16 kHz WAV using soundfile + scipy.

This script is NOT a DVC stage – it is a one-time data preparation helper.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

# HuggingFace language codes for Common Voice (subset)
CV_LANG_MAP = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "ru": "ru",
    "zh-CN": "zh-CN",
}

TARGET_SR = 16_000


def resample_if_needed(signal: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return signal
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(TARGET_SR, orig_sr)
    up, down = TARGET_SR // g, orig_sr // g
    return resample_poly(signal, up, down).astype(np.float32)


def to_mono(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 2:
        return signal.mean(axis=1).astype(np.float32)
    return signal.astype(np.float32)


def download_and_prepare(lang: str, n_utterances: int, raw_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not found. Install with: pip install datasets")
        sys.exit(1)

    cv_lang = CV_LANG_MAP.get(lang, lang)
    print(f"Downloading Common Voice ({cv_lang}, up to {n_utterances} utterances)...")

    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        cv_lang,
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    wav_dir = raw_dir / lang / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    text_tsv = raw_dir / lang / "text.tsv"

    count = 0
    texts = []

    for item in dataset:
        if count >= n_utterances:
            break

        sentence = item.get("sentence", "").strip()
        if not sentence:
            continue

        audio = item["audio"]
        signal = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        signal = to_mono(signal)
        signal = resample_if_needed(signal, sr)

        # Create a deterministic stem from the content hash
        content_hash = hashlib.md5(sentence.encode("utf-8")).hexdigest()[:8]
        stem = f"cv_{lang}_{count:06d}_{content_hash}"
        wav_path = wav_dir / f"{stem}.wav"

        sf.write(str(wav_path), signal, TARGET_SR)
        texts.append(f"{stem}\t{sentence}")
        count += 1

        if count % 10 == 0:
            print(f"  {count}/{n_utterances} utterances processed...")

    # Write text.tsv
    with open(text_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(texts) + "\n")

    print(f"Done: {count} utterances for lang={lang} → {raw_dir / lang}")


def main():
    parser = argparse.ArgumentParser(description="Download Common Voice data for the pipeline.")
    parser.add_argument("--langs", nargs="+", default=["en"], help="Language codes")
    parser.add_argument("--n-utterances", type=int, default=50)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    raw_dir = Path(params["data"]["raw_dir"])

    for lang in args.langs:
        download_and_prepare(lang, args.n_utterances, raw_dir)


if __name__ == "__main__":
    main()
