import os
import gzip
import json
import glob
import torch
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Experiment 002: same structure as experiment001 but using qwen3-1.7b + a residual-stream
# BatchTopK SAE at layer 14 (65k features, L0=80, adamkarvonen/qwen3-1.7b-saes trainer_2).
# Thinking mode is forced on by appending <think> to the prompt, which the model treats as the
# start of a reasoning trace it must continue. Reports top-5 active SAE features at six epochs:
# a) first prompt token, b) last prompt token (<think>), c) first thinking token,
# d) last thinking token, e) first answer token, f) last answer token.

#Parameters
model_name  = "qwen3-1.7b"
sae_release = "adamkarvonen/qwen3-1.7b-saes"   # HF repo_id (not in sae_lens catalog)
sae_id      = "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_14/trainer_2"
sae_name    = "14-resid-batchtopk-65k__l0-80"  # folder name used for explanations

# HF_HOME at model level so all models for qwen3-1.7b share the same weight cache.
_data_root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_model_cache = os.path.join(_data_root, model_name)
os.makedirs(_model_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)

from sae_lens import SAE, HookedSAETransformer                                     # noqa: E402
from sae_lens.loading.pretrained_sae_loaders import (                              # noqa: E402
    dictionary_learning_sae_huggingface_loader_1,
)
from transformer_lens.utilities import test_prompt                                 # noqa: E402

#CUDA-related
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("+Device: {}".format(device))

#model loading
model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device)
print("+Model {} loaded".format(model_name))

#SAE loading — uses dictionary_learning_1 converter because this SAE is not in sae_lens catalog
sae = SAE.from_pretrained(
    sae_release,
    sae_id,
    device=device,
    converter=dictionary_learning_sae_huggingface_loader_1,
)
print("+SAE loaded")

#load feature explanations
data_dir = os.path.join(_data_root, model_name, sae_name, "explanations")
explanations = {}
for path in glob.glob(os.path.join(data_dir, "*.jsonl.gz")):
    with gzip.open(path) as f:
        for line in f:
            entry = json.loads(line)
            explanations[int(entry["index"])] = entry["description"]
print(f"+Loaded {len(explanations)} feature explanations")

# <think> appended to force the model to start a reasoning trace.
# It becomes the last prompt token; the model generates thinking content from there.
prompt = "Is Uranium necessary to stay healthy? <think> It"
#prompt = "Is a librarian trained to fly a plane? <think> It"
#prompt = "Did Aristotle Use a Laptop? <think>"
answer = " No"
#test_prompt(prompt, answer, model)

output = model.generate(
    prompt,
    max_new_tokens=550,  # thinking traces can be long
    stop_at_eos=True,
    prepend_bos=sae.cfg.metadata.prepend_bos,
)
print("Model output: {}".format(output))

_, cache = model.run_with_cache_with_saes(output, saes=[sae])
sae_acts = cache[sae.cfg.metadata.hook_name + ".hook_sae_acts_post"][0]  # [pos, n_features]
str_tokens = model.to_str_tokens(output)

prompt_tokens = model.to_str_tokens(prompt, prepend_bos=sae.cfg.metadata.prepend_bos)
n_prompt = len(prompt_tokens)
n_total  = len(str_tokens)

eos_str = model.tokenizer.eos_token
last_answer_pos = n_total - 1
while last_answer_pos > n_prompt and str_tokens[last_answer_pos] == eos_str:
    last_answer_pos -= 1

bos_str = model.tokenizer.bos_token
first_prompt_pos = 1 if (len(str_tokens) > 0 and str_tokens[0] == bos_str) else 0

# Last question token: '?' marks the end of the question, before any <think> forcing tokens.
last_question_pos = n_prompt - 1
while last_question_pos >= first_prompt_pos and str_tokens[last_question_pos] != "?":
    last_question_pos -= 1
if last_question_pos < first_prompt_pos:
    print("WARNING: '?' not found in prompt — falling back to last prompt token")
    last_question_pos = n_prompt - 1

# Find </think> to split thinking trace from answer.
end_think_pos = None
for i in range(n_prompt, n_total):
    if str_tokens[i] == "</think>":
        end_think_pos = i
        break
if end_think_pos is None:
    print("WARNING: </think> not found — thinking epochs may be inaccurate")
    end_think_pos = n_prompt

# First thinking token: skip leading whitespace/newlines immediately after the prompt.
first_think_pos = n_prompt - 2
while first_think_pos < end_think_pos and str_tokens[first_think_pos].strip() == "":
    first_think_pos += 1

# Last thinking token: walk back from </think> skipping whitespace.
last_think_pos = end_think_pos - 1
while last_think_pos > n_prompt and str_tokens[last_think_pos].strip() == "":
    last_think_pos -= 1

# First answer token: skip whitespace after </think>.
first_answer_pos = end_think_pos + 1
while first_answer_pos < n_total and str_tokens[first_answer_pos].strip() == "":
    first_answer_pos += 1

epochs = [
    ("Epoch a) first prompt token",   first_prompt_pos),
    ("Epoch b) last question token",  last_question_pos),  # '?' — question state before thinking
    ("Epoch c) first thinking token", first_think_pos),
    ("Epoch d) last thinking token",  last_think_pos),
    ("Epoch e) first answer token",   first_answer_pos),
    ("Epoch f) last answer token",    last_answer_pos),
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
