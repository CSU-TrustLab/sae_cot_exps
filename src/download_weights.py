"""
Run once to download model + SAE weights into data/<model_name>/.
After this, experiment001.py will load from disk without hitting the network.

Usage (run from the sae_exps root):
    python src/download_weights.py
    python src/download_weights.py --model gemma-2b --sae-release gemma-2b-res-jb --sae-id blocks.12.hook_resid_post
"""

import argparse
import os

parser = argparse.ArgumentParser(description="Download model + SAE weights to data/<model>/.")
parser.add_argument("--model",       default="gemma-2b-it",               help="TransformerLens model name")
parser.add_argument("--sae-release", default="gemma-2b-it-res-jb",        help="sae_lens SAE release name")
parser.add_argument("--sae-id",      default="blocks.12.hook_resid_post",  help="sae_lens SAE id")
args = parser.parse_args()

_src_dir   = os.path.dirname(os.path.abspath(__file__))
_data_root = os.path.normpath(os.path.join(_src_dir, "..", "data"))
_model_dir = os.path.join(_data_root, args.model)
os.makedirs(_model_dir, exist_ok=True)

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _model_dir
    print(f"HF_HOME set to: data/{args.model}/")
else:
    print(f"HF_HOME already set to: {os.environ['HF_HOME']}")

import torch
from sae_lens import SAE, HookedSAETransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"Downloading transformer model '{args.model}'...")
HookedSAETransformer.from_pretrained(args.model, device=device)
print("  done.")

print(f"Downloading SAE '{args.sae_release} / {args.sae_id}'...")
SAE.from_pretrained(args.sae_release, args.sae_id, device=device)
print("  done.")

print(f"All weights cached to data/{args.model}/")
