#!/usr/bin/env python3
"""
Stage 3 – predict_phonemes
===========================
Loads the pre-trained phoneme recognition model
(facebook/wav2vec2-lv-60-espeak-cv-ft), runs inference on every utterance
described in an input manifest, and writes a prediction manifest.

The model requires 16 kHz mono audio. Audio that does not already satisfy
these constraints is resampled/converted before inference.

Usage:
    python src/predict_phonemes.py \\
        --manifest data/manifests/en/clean.jsonl \\
        --output   data/predictions/en/clean_pred.jsonl \\
        --params   params.yaml

Inputs:
    - Any manifest (clean or noisy JSONL)
    - params.yaml

Outputs:
    - Prediction manifest (input manifest + hyp_phon field) – written atomically
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from transformers import AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("predict_phonemes")

TARGET_SR = 16_000  # wav2vec2 requires 16 kHz


# ---------------------------------------------------------------------------
# Audio loading with resampling
# ---------------------------------------------------------------------------

def load_audio_16khz(wav_path: str) -> np.ndarray:
    """
    Load a WAV file and return a 16 kHz mono float32 array.
    Resamples if necessary using scipy (pure-Python fallback).
    """
    signal, sr = sf.read(wav_path, dtype="float32", always_2d=False)

    # Downmix to mono if needed
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != TARGET_SR:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(TARGET_SR, sr)
            up, down = TARGET_SR // g, sr // g
            signal = resample_poly(signal, up, down).astype(np.float32)
            log.debug("Resampled %s from %d Hz to %d Hz", wav_path, sr, TARGET_SR)
        except ImportError:
            raise RuntimeError(
                f"scipy is required for resampling. Install with: pip install scipy"
            )

    return signal


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


def write_manifest_atomically(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".tmp_pred_manifest_",
        suffix=".jsonl",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_path)
        log.info("Prediction manifest written atomically: %s (%d records)", output_path, len(records))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model_and_processor(model_name: str):
    """Load the wav2vec2 CTC model and its processor."""
    log.info("Loading model and processor: %s", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    log.info("Model loaded on device: %s", device)
    return model, processor, device


def predict_batch(
    signals: list,
    model,
    processor,
    device: str,
) -> list:
    """
    Run CTC decoding on a batch of 16 kHz mono signals.
    Returns a list of phoneme strings (one per signal).
    """
    inputs = processor(
        signals,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    # Greedy CTC decoding
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    return transcriptions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run phoneme recognition on a manifest.")
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, help="Output prediction JSONL manifest")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    model_name = params["model"]["name"]
    batch_size = params["model"]["batch_size"]

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    records = load_manifest(manifest_path)
    log.info("Loaded %d utterances from %s", len(records), manifest_path)

    model, processor, device = load_model_and_processor(model_name)

    pred_records = []
    errors = 0

    # Process in batches
    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start: batch_start + batch_size]
        batch_signals = []
        valid_records = []

        for rec in batch_records:
            try:
                signal = load_audio_16khz(rec["wav_path"])
                batch_signals.append(signal)
                valid_records.append(rec)
            except Exception as e:
                log.error("Failed to load audio %s: %s", rec.get("utt_id", "?"), e)
                errors += 1

        if not batch_signals:
            continue

        try:
            transcriptions = predict_batch(batch_signals, model, processor, device)
        except Exception as e:
            log.error("Batch inference failed: %s", e)
            for rec in valid_records:
                errors += 1
            continue

        for rec, hyp_phon in zip(valid_records, transcriptions):
            pred_rec = dict(rec)
            pred_rec["hyp_phon"] = hyp_phon.strip()
            pred_records.append(pred_rec)

        log.info(
            "Processed %d/%d utterances",
            min(batch_start + batch_size, len(records)),
            len(records),
        )

    if errors > 0:
        log.warning("%d utterances failed inference", errors)

    if not pred_records:
        log.error("No predictions produced. Aborting.")
        sys.exit(1)

    write_manifest_atomically(pred_records, output_path)
    log.info("Done. %d predictions written.", len(pred_records))


if __name__ == "__main__":
    main()
