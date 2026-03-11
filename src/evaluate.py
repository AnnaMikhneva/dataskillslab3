#!/usr/bin/env python3
"""
Stage 4 – evaluate
===================
Computes the Phoneme Error Rate (PER) for every utterance in a prediction
manifest and writes a JSON metrics file.

PER = (S + D + I) / N  where N = number of reference phonemes.

Since each phoneme is a single token (space-separated), this is exactly
equivalent to the Word Error Rate computed token-by-token, or the CER when
phonemes are single characters. We use the Levenshtein (edit) distance on
the phoneme token sequence.

Usage:
    python src/evaluate.py \\
        --manifest  data/predictions/en/clean_pred.jsonl \\
        --output    metrics/en/clean.json \\
        --snr       null \\
        --lang      en \\
        --params    params.yaml

Inputs:
    - Prediction manifest (JSONL with ref_phon + hyp_phon fields)

Outputs:
    - JSON metrics file: {lang, snr_db, per_mean, per_std, n_utts, ...}
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# PER computation
# ---------------------------------------------------------------------------

def edit_distance(ref_tokens: list, hyp_tokens: list) -> int:
    """
    Standard dynamic-programming Levenshtein distance between two token sequences.
    Each phoneme is treated as a single token (space-separated in the manifest).
    This is strictly equivalent to CER when phonemes are single characters.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[m]


def compute_per(ref_phon: str, hyp_phon: str) -> float:
    """
    Compute PER for a single utterance.
    Phoneme sequences are whitespace-tokenised.
    Returns 0.0 if ref is empty (avoid ZeroDivisionError).
    """
    ref_tokens = ref_phon.split()
    hyp_tokens = hyp_phon.split()
    if not ref_tokens:
        return 0.0
    dist = edit_distance(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


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


def write_json_atomically(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".tmp_metrics_",
        suffix=".json",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, output_path)
        log.info("Metrics written atomically: %s", output_path)
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
    parser = argparse.ArgumentParser(description="Evaluate phoneme predictions (PER).")
    parser.add_argument("--manifest", required=True, help="Prediction JSONL manifest")
    parser.add_argument("--output", required=True, help="Output JSON metrics file")
    parser.add_argument("--snr", default="null", help="SNR level in dB (or 'null' for clean)")
    parser.add_argument("--lang", required=True, help="Language code")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    snr_db = None if args.snr in ("null", "None", "") else float(args.snr)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    records = load_manifest(manifest_path)
    log.info("Loaded %d records from %s", len(records), manifest_path)

    per_values = []
    utt_metrics = []
    errors = 0

    for rec in records:
        utt_id = rec.get("utt_id", "?")
        ref_phon = rec.get("ref_phon", "")
        hyp_phon = rec.get("hyp_phon", "")

        if not ref_phon:
            log.warning("Empty ref_phon for %s, skipping", utt_id)
            errors += 1
            continue
        if hyp_phon is None:
            log.warning("Missing hyp_phon for %s, skipping", utt_id)
            errors += 1
            continue

        per = compute_per(ref_phon, str(hyp_phon))
        per_values.append(per)
        utt_metrics.append({"utt_id": utt_id, "per": round(per, 6)})

    if not per_values:
        log.error("No valid utterances to evaluate.")
        sys.exit(1)

    import statistics
    mean_per = statistics.mean(per_values)
    std_per = statistics.stdev(per_values) if len(per_values) > 1 else 0.0

    metrics = {
        "lang": args.lang,
        "snr_db": snr_db,
        "per_mean": round(mean_per, 6),
        "per_std": round(std_per, 6),
        "n_utts": len(per_values),
        "n_errors": errors,
        "utterances": utt_metrics,
    }

    write_json_atomically(metrics, output_path)
    log.info(
        "PER: mean=%.4f  std=%.4f  n=%d  (SNR=%s dB, lang=%s)",
        mean_per, std_per, len(per_values), snr_db, args.lang,
    )


if __name__ == "__main__":
    main()
