import os
import gzip
import json
import glob
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Experiment 006: DeepSeek-R1-Distill-Llama-8B. Runs the model once to cache the clean
# residual stream at every layer, then applies each layer's SAE (0–31) independently
# to find, for each of the 6 tokens of interest, the top 3 features by average activation
# strength across all 32 layers. Saves one line plot per token (x = layer, y = strength).

# Parameters
model_name    = "deepseek-r1-distill-llama-8b"
hf_model_id   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tl_model_ref  = "meta-llama/Llama-3.1-8B"   # same architecture; TransformerLens registry name
sae_release   = "llama_scope_r1_distill"
N_LAYERS      = 32
TOP_FEATURES  = 3

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment006")
_model_cache  = os.path.join(_data_root, model_name)
os.makedirs(_model_cache, exist_ok=True)
os.makedirs(_results_root, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)

from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402
from sae_lens import SAE, HookedSAETransformer                 # noqa: E402


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
del hf_model_raw
print("+Model {} loaded".format(hf_model_id))

#prompt = "Is Uranium necessary to stay healthy?"
prompt = "Is a librarian trained to fly a plane?"
#prompt = "Did Aristotle Use a Laptop?"
answer = " No"

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
    ("Epoch b) last question token",  last_question_pos),
    ("Epoch c) first thinking token", first_think_pos),
    ("Epoch d) last thinking token",  last_think_pos),
    ("Epoch e) first answer token",   first_answer_pos),
    ("Epoch f) last answer token",    last_answer_pos),
]
epoch_short = [
    "a_first_prompt", "b_last_question", "c_first_think",
    "d_last_think",   "e_first_answer",  "f_last_answer",
]

# Cache clean residual stream at every layer (no SAEs hooked)
print("+Building residual stream cache...")
_, resid_cache = model.run_with_cache(
    output_tokens,
    names_filter=lambda name: name.endswith("hook_resid_post"),
    return_type=None,
)
print("+Cache built ({} layers)".format(len(resid_cache)))

# Apply each SAE to the cached residual stream; keep only the 6 epoch positions (CPU)
# acts_by_layer[layer][epoch_label] = tensor[n_features]
acts_by_layer = {}
for layer in range(N_LAYERS):
    sae_id = f"l{layer}r_800m_slimpajama"
    sae = SAE.from_pretrained(sae_release, sae_id, device=device)
    resid = resid_cache[sae.cfg.metadata.hook_name][0]   # [pos, d_model]
    feature_acts = sae.encode(resid)             # [pos, n_features]
    acts_by_layer[layer] = {
        label: feature_acts[pos].cpu()
        for label, pos in epochs
    }
    del sae
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"+Layer {layer:2d} processed")

del resid_cache
if device == "cuda":
    torch.cuda.empty_cache()

# Load feature descriptions from available explanation files
# explanations[layer][feat_idx] = description string (may be absent for many layers)
explanations = {}
for layer in range(N_LAYERS):
    sae_name = f"{layer}-llamascope-slimpj-openr1-res-32k"
    exps = {}
    top_ex = {}
    data_dir = os.path.join(_data_root, model_name, sae_name, "explanations")
    for path in glob.glob(os.path.join(data_dir, "*.jsonl.gz")):
        with gzip.open(path) as f:
            for line in f:
                entry = json.loads(line)
                idx = int(entry["index"])
                if "description" in entry:
                    exps[idx] = entry["description"]
                else:
                    mv = entry.get("maxValue", 0)
                    if idx not in top_ex or mv > top_ex[idx]["maxValue"]:
                        top_ex[idx] = entry
    for idx, ex in top_ex.items():
        if idx not in exps:
            tokens = ex["tokens"]
            pos = ex["maxValueTokenIndex"]
            window = tokens[max(0, pos - 4): pos + 6]
            exps[idx] = "[top act] " + "".join(window).replace("\n", " ").strip()
    explanations[layer] = exps
    if exps:
        print(f"+Layer {layer:2d}: {len(exps)} descriptions loaded")

# For each epoch: find top-3 features by mean activation across layers, save line plot
layers_x = list(range(N_LAYERS))

for (label, pos), short in zip(epochs, epoch_short):
    # stacked: [N_LAYERS, n_features]
    stacked = torch.stack([acts_by_layer[layer][label] for layer in layers_x])
    mean_acts = stacked.mean(dim=0)
    top_vals, top_inds = torch.topk(mean_acts, TOP_FEATURES)

    print(f"\n{label}  token={repr(decoded_tokens[pos])}")

    fig, ax = plt.subplots(figsize=(12, 5))
    for rank, (val, ind) in enumerate(zip(top_vals, top_inds), 1):
        feat_idx = ind.item()
        y = [acts_by_layer[layer][label][feat_idx].item() for layer in layers_x]
        peak_layer = int(torch.tensor(y).argmax())
        desc = explanations.get(peak_layer, {}).get(feat_idx, "")
        legend_str = f"F{feat_idx}" + (f": {desc}" if desc else "")
        print(f"  #{rank} F{feat_idx:<6} mean={val:.4f}  peak=L{peak_layer}  {desc}")
        ax.plot(layers_x, y, marker="o", markersize=4, label=legend_str)

    ax.set_title(f"Experiment 006 — {label}  [token: {repr(decoded_tokens[pos])}]")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature activation strength")
    ax.set_xticks(layers_x)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(_results_root, f"exp006_{short}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

print("\nDone!")
