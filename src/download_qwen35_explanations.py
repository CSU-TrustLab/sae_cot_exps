"""
Download Neuronpedia explanation JSONL files for Qwen3.5-27B SAEs.

Currently only layer 31 (31-qwenscope-res-80k) is available on S3.
Files are saved to data/qwen3.5-27b/{sae_name}/explanations/.

Usage:
    python src/download_qwen35_explanations.py
"""

import os
import sys
import urllib.request

S3_BASE    = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
MODEL_KEY  = "qwen3.5-27b"
# Only layer 31 has explanations uploaded as of 2026-05
SAE_NAMES  = ["31-qwenscope-res-80k"]
N_BATCHES  = 304   # batch-0 … batch-303

_src_dir   = os.path.dirname(os.path.abspath(__file__))
_data_root = os.path.normpath(os.path.join(_src_dir, "..", "data", MODEL_KEY))


def _download(url, dest):
    tmp = dest + ".part"
    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            sys.stdout.write(f"\r  {pct:5.1f}%")
            sys.stdout.flush()
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
        sys.stdout.write("\n")
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


for sae_name in SAE_NAMES:
    out_dir = os.path.join(_data_root, sae_name, "explanations")
    os.makedirs(out_dir, exist_ok=True)

    downloaded = skipped = 0
    for i in range(N_BATCHES):
        fname = f"batch-{i}.jsonl.gz"
        dest  = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            skipped += 1
            continue
        url = f"{S3_BASE}/v1/{MODEL_KEY}/{sae_name}/explanations/{fname}"
        print(f"[{sae_name}] {fname} ({i+1}/{N_BATCHES})")
        _download(url, dest)
        downloaded += 1

    print(f"\n{sae_name}: {downloaded} downloaded, {skipped} already present "
          f"(total {N_BATCHES} files)\n")

print("All done!")
