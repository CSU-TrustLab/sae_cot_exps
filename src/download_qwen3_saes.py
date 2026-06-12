"""
Download SAE weights for Qwen3-1.7B (BatchTopK, 65k features) from
huggingface.co/adamkarvonen/qwen3-1.7b-saes.

Saves to data/qwen3-1.7b/{layer}-resid-batchtopk-65k__l0-{k}/trainer_{n}/
Skips files that already exist.

Usage:
    python src/download_qwen3_saes.py
    python src/download_qwen3_saes.py --layers 7 21   # specific layers only
"""

import argparse
import os
import sys
import urllib.request

HF_REPO    = "adamkarvonen/qwen3-1.7b-saes"
HF_SUBDIR  = "saes_Qwen_Qwen3-1.7B_batch_top_k"
# Layers available in the repo
ALL_LAYERS = [7, 14, 21]
TRAINERS   = [0, 1]   # trainer_0 = k80, trainer_1 = k160 (lowest/highest sparsity)
FILES      = ["ae.pt", "config.json", "eval_results.json"]

_src_dir   = os.path.dirname(os.path.abspath(__file__))
_data_root = os.path.normpath(os.path.join(_src_dir, "..", "data", "qwen3-1.7b"))

parser = argparse.ArgumentParser()
parser.add_argument("--layers", nargs="*", type=int, default=None,
                    help="Which layers to download (default: all missing)")
args = parser.parse_args()

layers = args.layers if args.layers else ALL_LAYERS


def local_dir(layer, trainer):
    # Folder convention: {layer}-resid-batchtopk-65k__l0-80/trainer_{n}
    return os.path.join(_data_root, f"{layer}-resid-batchtopk-65k__l0-80",
                        f"trainer_{trainer}")


def hf_url(layer, trainer, filename):
    return (f"https://huggingface.co/{HF_REPO}/resolve/main/"
            f"{HF_SUBDIR}/resid_post_layer_{layer}/trainer_{trainer}/{filename}")


def download_with_progress(url, dest_path):
    tmp = dest_path + ".part"

    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size * 100 / total_size, 100)
        mb  = count * block_size / 1_048_576
        tot = total_size / 1_048_576
        sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f} / {tot:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
        sys.stdout.write("\n")
        os.replace(tmp, dest_path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


total_files = 0
skipped     = 0
downloaded  = 0

for layer in layers:
    for trainer in TRAINERS:
        d = local_dir(layer, trainer)
        os.makedirs(d, exist_ok=True)
        for fname in FILES:
            dest = os.path.join(d, fname)
            if os.path.exists(dest):
                skipped += 1
                continue
            url = hf_url(layer, trainer, fname)
            print(f"Downloading L{layer}/trainer_{trainer}/{fname}")
            download_with_progress(url, dest)
            downloaded += 1
        total_files += len(FILES)

print(f"\nDone. {downloaded} downloaded, {skipped} already present "
      f"(total {total_files} files across {len(layers)} layers).")
