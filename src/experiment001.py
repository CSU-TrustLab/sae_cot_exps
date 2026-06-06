import os
import gzip
import json
import glob
import torch
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd
from sae_lens import SAE, HookedSAETransformer
from transformer_lens.utilities import test_prompt
import warnings
warnings.filterwarnings("ignore")

# Experiment 001: for a given prompt and model (gemma-2b-it + residual-stream SAE at layer 12),
# generate an answer autoregressively and report the top-5 active SAE features at four points:
# a) the first prompt token, b) the last prompt token (model state just before generation begins),
# c) the first generated token, and d) the last generated token.

#Parameters
model_name = "gemma-2b-it"
sae_release="gemma-2b-it-res-jb"
sae_id="blocks.12.hook_resid_post"

# If HF_HOME points to the model's data dir, all HuggingFace downloads
# (model weights, tokenizer, SAE) are read from / written to that folder.
_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_model_cache = os.path.join(_data_root, model_name)
os.makedirs(_model_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)

#CUDA-related
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("+Device: {}".format(device))

#model loading
#model = HookedSAETransformer.from_pretrained(model_name, device=device)
#let's follow this warning https://decoderesearch.github.io/SAELens/latest/usage/
model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device)
print("+Model {} loaded".format(model_name))

#SAE loading
sae = SAE.from_pretrained(
    sae_release,
    sae_id,
    device=device,
)
print("+SAE loaded")

#load feature explanations
_base_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
data_dir = os.path.join(_base_data_dir, model_name)
explanations = {}
for path in glob.glob(os.path.join(data_dir, "*.jsonl.gz")):
    with gzip.open(path) as f:
        for line in f:
            entry = json.loads(line)
            explanations[int(entry["index"])] = entry["description"]
print(f"+Loaded {len(explanations)} feature explanations")

prompt = "Is Uranium necessary to stay healthy?"
#prompt = "Is a librarian trained to fly a plane?"
#prompt = "Did Aristotle Use a Laptop?"
#prompt = "Is so hot here!"
answer = " No"
#test_prompt(prompt, answer, model)

output = model.generate(
    prompt,
    max_new_tokens=50,
    stop_at_eos=True,
    prepend_bos=sae.cfg.metadata.prepend_bos,
)
print("Model output: {}".format(output))

# Single forward pass on full output (prompt + answer).
# Due to causal attention, activations at prompt positions are unaffected
# by later answer tokens, so no separate runs are needed.
_, cache = model.run_with_cache_with_saes(output, saes=[sae])
sae_acts = cache[sae.cfg.metadata.hook_name + ".hook_sae_acts_post"][0]  # [pos, n_features]
str_tokens = model.to_str_tokens(output)

# Locate epoch boundaries.
# prepend_bos adds a BOS token before the prompt tokens when tokenising.
prompt_tokens = model.to_str_tokens(prompt, prepend_bos=sae.cfg.metadata.prepend_bos)
n_prompt = len(prompt_tokens)           # number of tokens in the prompt (incl. BOS if any)
n_total  = len(str_tokens)

# Last content token of the answer: walk back from the end, skipping EOS tokens.
eos_str = model.tokenizer.eos_token
last_answer_pos = n_total - 1
while last_answer_pos > n_prompt and str_tokens[last_answer_pos] == eos_str:
    last_answer_pos -= 1

first_prompt_pos = 1 if sae.cfg.metadata.prepend_bos else 0

# First answer token: skip leading whitespace/newline tokens (e.g. '\n\n' emitted by chat models).
first_answer_pos = n_prompt
while first_answer_pos < n_total and str_tokens[first_answer_pos].strip() == "":
    first_answer_pos += 1

epochs = [
    ("Epoch a) first prompt token",   first_prompt_pos),
    ("Epoch b) last prompt token",    n_prompt - 1),  # pre-generation state
    ("Epoch c) first answer token",   first_answer_pos),
    ("Epoch d) last answer token",    last_answer_pos),
]

def print_top_features(label, pos, acts, str_tokens, explanations, k=5):
    print(f"\n{label}")
    print(f"  token [{pos}]: {repr(str_tokens[pos])}")
    vals, inds = torch.topk(acts[pos], k)
    for val, ind in zip(vals, inds):
        desc = explanations.get(ind.item(), "no description")
        print(f"    F{ind.item():<6} {val:.2f}  {desc}")

for label, pos in epochs:
    print_top_features(label, pos, sae_acts, str_tokens, explanations)

print("\nDone!")
