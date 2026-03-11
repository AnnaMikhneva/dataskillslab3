# save as setup_data_librispeech.py
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import numpy as np
import yaml
from math import gcd
from scipy.signal import resample_poly

def to_mono_16k(signal, sr):
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)
    if sr != 16000:
        g = gcd(16000, sr)
        signal = resample_poly(signal, 16000 // g, sr // g).astype(np.float32)
    return signal

with open("params.yaml") as f:
    params = yaml.safe_load(f)

raw_dir = Path(params["data"]["raw_dir"])

# For French — replace the ds line above with:
ds = load_dataset("facebook/multilingual_librispeech", "french", split="test", streaming=True, trust_remote_code=True)

for lang in params["languages"]:
    wav_dir = raw_dir / lang / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    count = 0
    n = params["data"].get("max_utterances") or 50

    for item in ds:
        if count >= n:
            break
        signal = np.array(item["audio"]["array"], dtype=np.float32)
        sr     = item["audio"]["sampling_rate"]
        signal = to_mono_16k(signal, sr)

        stem = f"ls_{lang}_{count:06d}"
        sf.write(str(wav_dir / f"{stem}.wav"), signal, 16000)
# Try both field names — librispeech_asr uses 'text', multilingual_librispeech uses 'sentence'
        transcript = item.get("text") or item.get("sentence") or item.get("transcript", "")
        texts.append(f"{stem}\t{transcript.lower()}")
        count += 1
        if count % 10 == 0:
            print(f"  [{lang}] {count}/{n}")

    with open(raw_dir / lang / "text.tsv", "w") as f:
        f.write("\n".join(texts) + "\n")
    print(f"Done: {count} utterances for lang={lang}")