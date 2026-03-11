#!/usr/bin/env python3


import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("prepare_manifests")


# Helpers

def load_params(params_path: str) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def md5_file(path: str) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def text_to_phonemes(text: str, lang: str) -> str:
    """
    Convert text to IPA phoneme string using espeak-ng.

    espeak-ng flags:
      -v <lang>  : language/voice
      --ipa      : output IPA transcription
      -q         : quiet (no audio)
      --stdin    : read text from stdin

    Returns the phoneme string with leading/trailing whitespace stripped.
    Raises RuntimeError if espeak-ng is unavailable or fails.
    """
    cmd = ["espeak-ng", "-v", lang, "--ipa", "-q", "--stdin"]
    try:
        result = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        # espeak-ng prefixes each line with spaces
        phonemes = " ".join(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )
        return phonemes
    except FileNotFoundError:
        raise RuntimeError(
            "espeak-ng not found. Install with: sudo apt-get install espeak-ng"
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"espeak-ng failed for text '{text[:40]}': {e.stderr}"
        ) from e


def load_text_references(text_tsv: Path) -> dict:

    refs = {}
    with open(text_tsv) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                log.warning("Skipping malformed line %d in %s", lineno, text_tsv)
                continue
            stem, ref_text = parts
            refs[stem.strip()] = ref_text.strip()
    return refs


def write_manifest_atomically(records: list, output_path: Path) -> None:
    """
    Write JSONL manifest atomically using a temp file + rename.
    The rename is atomic on POSIX; on Windows we use os.replace which is
    also atomic (replaces the destination if it already exists).
    """
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
        # Atomic rename (works on Linux/macOS; os.replace on Windows)
        os.replace(tmp_path, output_path)
        log.info("Manifest written atomically to %s (%d records)", output_path, len(records))
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Prepare clean manifest with phoneme references.")
    parser.add_argument("--lang", required=True, help="Language code (e.g. en, fr)")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    lang = args.lang
    raw_dir = Path(params["data"]["raw_dir"]) / lang
    manifest_dir = Path(params["data"]["manifest_dir"]) / lang
    max_utt = params["data"].get("max_utterances")

    wav_dir = raw_dir / "wav"
    text_tsv = raw_dir / "text.tsv"

    if not wav_dir.exists():
        log.error("WAV directory not found: %s", wav_dir)
        sys.exit(1)
    if not text_tsv.exists():
        log.error("Text reference file not found: %s", text_tsv)
        sys.exit(1)

    # Loading references
    refs = load_text_references(text_tsv)
    log.info("Loaded %d text references for language '%s'", len(refs), lang)

    # Discovering WAV files
    wav_files = sorted(wav_dir.glob("*.wav"))
    if max_utt is not None:
        wav_files = wav_files[:max_utt]
    log.info("Found %d WAV files", len(wav_files))

    records = []
    errors = 0

    for wav_path in wav_files:
        stem = wav_path.stem

        if stem not in refs:
            log.warning("No reference text for stem '%s', skipping", stem)
            errors += 1
            continue

        ref_text = refs[stem]

        # Stable utt_id: {lang}_{stem}
        utt_id = f"{lang}_{stem}"

        # Relative path from project root
        rel_wav = str(wav_path)
        # If path starts with ./ strip it for cleanliness
        if rel_wav.startswith("./"):
            rel_wav = rel_wav[2:]

        # Audio metadata
        try:
            info = sf.info(str(wav_path))
            sr = info.samplerate
            duration_s = info.duration
            n_channels = info.channels
        except Exception as e:
            log.warning("Could not read audio info for %s: %s", wav_path, e)
            errors += 1
            continue

        if n_channels != 1:
            log.warning(
                "Audio %s has %d channels (expected mono). Skipping.", wav_path, n_channels
            )
            errors += 1
            continue

        # Checksum
        audio_md5 = md5_file(str(wav_path))

        # Phoneme transcription
        try:
            ref_phon = text_to_phonemes(ref_text, lang)
        except RuntimeError as e:
            log.error("Phoneme conversion failed for %s: %s", utt_id, e)
            errors += 1
            continue

        record = {
            "utt_id": utt_id,
            "lang": lang,
            "wav_path": rel_wav,
            "ref_text": ref_text,
            "ref_phon": ref_phon,
            "sr": sr,
            "duration_s": round(duration_s, 4),
            "snr_db": None,
            "audio_md5": audio_md5,
        }
        records.append(record)
        log.debug("Processed %s: phonemes='%s'", utt_id, ref_phon[:60])

    if not records:
        log.error("No valid records produced. Aborting.")
        sys.exit(1)

    if errors > 0:
        log.warning("%d utterances skipped due to errors", errors)

    # Writing manifest atomically
    output_path = manifest_dir / "clean.jsonl"
    write_manifest_atomically(records, output_path)
    log.info("Done. %d records written, %d skipped.", len(records), errors)


if __name__ == "__main__":
    main()
