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

# Experiment 003: DeepSeek-R1-Distill-Llama-8B + residual-stream SAE at layer 20 (32k features,
# llama_scope_r1_distill / fnlp/Llama-Scope-R1-Distill). Unlike experiments 001-002, this model
# natively generates <think> traces without any prompt forcing. Reports top-5 active SAE features
# at six epochs: a) first prompt token, b) last question token (?), c) first thinking token,
# d) last thinking token, e) first answer token, f) last answer token.

#Parameters
model_name   = "deepseek-r1-distill-llama-8b"
hf_model_id  = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tl_model_ref = "meta-llama/Llama-3.1-8B"   # same architecture; TransformerLens registry name
sae_release  = "llama_scope_r1_distill"
sae_id       = "l20r_800m_slimpajama"
sae_name     = "20-llamascope-slimpj-openr1-res-32k"

_data_root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_model_cache = os.path.join(_data_root, model_name)
os.makedirs(_model_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)

from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402
from sae_lens import SAE, HookedSAETransformer                 # noqa: E402
from transformer_lens.utilities import test_prompt              # noqa: E402


def _build_byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

_byte_decoder = _build_byte_decoder()

def _decode_str_token(tok):
    """Invert GPT-2/tiktoken BPE byte encoding: Ġ→space, Ċ→newline, âĢĻ→' etc."""
    try:
        return bytes([_byte_decoder[c] for c in tok]).decode("utf-8", errors="replace")
    except KeyError:
        return tok  # special token (e.g. <think>); not BPE-encoded, return as-is


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("+Device: {}".format(device))

# TransformerLens does not have DeepSeek-R1-Distill in its registry.
# Load weights + tokenizer from HF directly, then hand them to TransformerLens
# with the Llama-3.1-8B config (same architecture, same vocab size).
hf_model_raw = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    cache_dir=_model_cache,
    torch_dtype=torch.float32,
)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_model_cache)
model = HookedSAETransformer.from_pretrained_no_processing(
    tl_model_ref,
    hf_model=hf_model_raw,
    tokenizer=hf_tokenizer,
    device=device,
)
del hf_model_raw   # free the raw model; TransformerLens holds its own copy
print("+Model {} loaded".format(hf_model_id))

sae = SAE.from_pretrained(sae_release, sae_id, device=device)
print("+SAE loaded")

data_dir = os.path.join(_data_root, model_name, sae_name, "explanations")
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
answer = " No"
#test_prompt(prompt, answer, model)

# The DeepSeek tokenizer's normalizer strips ASCII spaces (0x20). Replacing spaces with
# Ġ (U+0120 — the BPE word-boundary marker) bypasses this and produces correct token IDs.
# BOS is prepended manually since add_special_tokens is unreliable for the Ġ-encoded form.
_prompt_bpe = prompt.replace(" ", "Ġ")
_prompt_ids = hf_tokenizer(_prompt_bpe, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
input_ids = torch.cat([torch.tensor([[hf_tokenizer.bos_token_id]], device=device), _prompt_ids], dim=1)
n_prompt = input_ids.shape[1]
print("DEBUG str_tokens(input)[:15]:", model.to_str_tokens(input_ids[0])[:15])

output_tokens = model.generate(
    input_ids,
    max_new_tokens=512,
    stop_at_eos=True,
    prepend_bos=False,  # BOS already in input_ids
    return_type="tokens",
)
str_tokens = model.to_str_tokens(output_tokens[0])
decoded_tokens = [_decode_str_token(t) for t in str_tokens]
print("Model output: {}".format("".join(decoded_tokens)))

_, cache = model.run_with_cache_with_saes(output_tokens, saes=[sae])
sae_acts = cache[sae.cfg.metadata.hook_name + ".hook_sae_acts_post"][0]  # [pos, n_features]

n_total = len(str_tokens)

eos_str = model.tokenizer.eos_token
last_answer_pos = n_total - 1
while last_answer_pos > n_prompt and decoded_tokens[last_answer_pos] == eos_str:
    last_answer_pos -= 1

bos_id = model.tokenizer.bos_token_id
first_prompt_pos = 1 if (output_tokens[0][0].item() == bos_id) else 0

# Last question token: '?' marks the end of the question.
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

# First thinking token: skip the opening <think> tag and leading whitespace.
first_think_pos = n_prompt
while first_think_pos < end_think_pos and (
    str_tokens[first_think_pos] == "<think>" or
    decoded_tokens[first_think_pos].strip() == ""
):
    first_think_pos += 1

# Last thinking token: walk back from </think> skipping whitespace.
last_think_pos = end_think_pos - 1
while last_think_pos > n_prompt and decoded_tokens[last_think_pos].strip() == "":
    last_think_pos -= 1

# First answer token: skip whitespace after </think>.
first_answer_pos = end_think_pos + 1
while first_answer_pos < n_total and decoded_tokens[first_answer_pos].strip() == "":
    first_answer_pos += 1

epochs = [
    ("Epoch a) first prompt token",   first_prompt_pos),
    ("Epoch b) last question token",  last_question_pos),  # '?' — question state before thinking
    ("Epoch c) first thinking token", first_think_pos),
    ("Epoch d) last thinking token",  last_think_pos),
    ("Epoch e) first answer token",   first_answer_pos),
    ("Epoch f) last answer token",    last_answer_pos),
]

def print_top_features(label, pos, acts, decoded_tokens, explanations, k=5):
    print(f"\n{label}")
    print(f"  token [{pos}]: {repr(decoded_tokens[pos])}")
    vals, inds = torch.topk(acts[pos], k)
    for val, ind in zip(vals, inds):
        desc = explanations.get(ind.item(), "no description")
        print(f"    F{ind.item():<6} {val:.2f}  {desc}")

for label, pos in epochs:
    print_top_features(label, pos, sae_acts, decoded_tokens, explanations)

print("\nDone!")
