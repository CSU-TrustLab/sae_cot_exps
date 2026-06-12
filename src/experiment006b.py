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

# Experiment 006b: DeepSeek-R1-Distill-Llama-8B, LlamaScope SAE at layer 31 only.
# Top-3 features selected by mean activation across all token positions at layer 31.
# Single plot: x = token position, y = feature activation strength, with epoch markers.

model_name    = "deepseek-r1-distill-llama-8b"
hf_model_id   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tl_model_ref  = "meta-llama/Llama-3.1-8B"
sae_release   = "llama_scope_r1_distill"
LAST_LAYER    = 31
TOP_FEATURES  = 3

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment006b")
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
    """Invert GPT-2/tiktoken BPE byte encoding: Ġ→space, Ċ→newline, etc."""
    try:
        return bytes([_byte_decoder[c] for c in tok]).decode("utf-8", errors="replace")
    except KeyError:
        return tok


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"+Device: {device}")

hf_model_raw = AutoModelForCausalLM.from_pretrained(
    hf_model_id, cache_dir=_model_cache, torch_dtype=torch.float32,
)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_model_cache)
model = HookedSAETransformer.from_pretrained_no_processing(
    tl_model_ref, hf_model=hf_model_raw, tokenizer=hf_tokenizer, device=device,
)
del hf_model_raw
print(f"+Model {hf_model_id} loaded")

#prompt = "Is a librarian trained to fly a plane?"
prompt = "Is Uranium necessary to stay healthy?"

# The DeepSeek tokenizer's normalizer strips ASCII spaces; replace with Ġ to
# get correct token IDs, then prepend BOS manually.
_prompt_bpe = prompt.replace(" ", "Ġ")
_prompt_ids = hf_tokenizer(_prompt_bpe, add_special_tokens=False,
                            return_tensors="pt").input_ids.to(device)
input_ids = torch.cat(
    [torch.tensor([[hf_tokenizer.bos_token_id]], device=device), _prompt_ids], dim=1
)
n_prompt = input_ids.shape[1]
print("DEBUG str_tokens(input)[:15]:", model.to_str_tokens(input_ids[0])[:15])

output_tokens = model.generate(
    input_ids, max_new_tokens=512, stop_at_eos=True,
    prepend_bos=False, return_type="tokens",
)
str_tokens    = model.to_str_tokens(output_tokens[0])
decoded_tokens = [_decode_str_token(t) for t in str_tokens]
print("Model output: {}".format("".join(decoded_tokens)))
n_total = len(str_tokens)

eos_str = model.tokenizer.eos_token
last_answer_pos = n_total - 1
while last_answer_pos > n_prompt and decoded_tokens[last_answer_pos] == eos_str:
    last_answer_pos -= 1

bos_id = model.tokenizer.bos_token_id
first_prompt_pos = 1 if (output_tokens[0][0].item() == bos_id) else 0

last_question_pos = n_prompt - 1
while last_question_pos >= first_prompt_pos and str_tokens[last_question_pos] != "?":
    last_question_pos -= 1
if last_question_pos < first_prompt_pos:
    print("WARNING: '?' not found in prompt — falling back to last prompt token")
    last_question_pos = n_prompt - 1

end_think_pos = None
for i in range(n_prompt, n_total):
    if str_tokens[i] == "</think>":
        end_think_pos = i
        break
if end_think_pos is None:
    print("WARNING: </think> not found — thinking epochs may be inaccurate")
    end_think_pos = n_prompt

first_think_pos = n_prompt
while first_think_pos < end_think_pos and (
    str_tokens[first_think_pos] == "<think>" or
    decoded_tokens[first_think_pos].strip() == ""
):
    first_think_pos += 1

last_think_pos = end_think_pos - 1
while last_think_pos > n_prompt and decoded_tokens[last_think_pos].strip() == "":
    last_think_pos -= 1

first_answer_pos = end_think_pos + 1
while first_answer_pos < n_total and decoded_tokens[first_answer_pos].strip() == "":
    first_answer_pos += 1

epochs = [
    ("a) first prompt",   first_prompt_pos),
    ("b) last question",  last_question_pos),
    ("c) first thinking", first_think_pos),
    ("d) last thinking",  last_think_pos),
    ("e) first answer",   first_answer_pos),
    ("f) last answer",    last_answer_pos),
]

# Cache residual stream (only hook_resid_post needed)
print("+Building residual stream cache...")
_, resid_cache = model.run_with_cache(
    output_tokens,
    names_filter=lambda name: name.endswith("hook_resid_post"),
    return_type=None,
)
print("+Cache built")

# Apply SAE at layer 31; keep full sequence activations for the token-axis plot
sae_id = f"l{LAST_LAYER}r_800m_slimpajama"
sae    = SAE.from_pretrained(sae_release, sae_id, device=device)
resid  = resid_cache[sae.cfg.metadata.hook_name][0]   # [n_total, d_model]
full_acts = sae.encode(resid).cpu()                    # [n_total, n_features]
del sae, resid_cache
if device == "cuda":
    torch.cuda.empty_cache()
print(f"+Layer {LAST_LAYER} processed  shape={list(full_acts.shape)}")

# Top-3 features by mean activation across all token positions
mean_acts = full_acts.mean(dim=0)
top_vals, top_inds = torch.topk(mean_acts, TOP_FEATURES)

# Load descriptions for layer 31
sae_name = f"{LAST_LAYER}-llamascope-slimpj-openr1-res-32k"
exps   = {}
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
        pos    = ex["maxValueTokenIndex"]
        window = tokens[max(0, pos - 4): pos + 6]
        exps[idx] = "[top act] " + "".join(window).replace("\n", " ").strip()
print(f"+Layer {LAST_LAYER}: {len(exps)} descriptions loaded")


def get_description(feat_idx):
    desc = exps.get(feat_idx, "")
    if desc:
        return desc
    peak_pos = int(full_acts[:, feat_idx].argmax())
    window   = str_tokens[max(0, peak_pos - 4): peak_pos + 6]
    return "[top act] " + "".join(window).replace("\n", " ").strip()


# Single plot: x = token position, y = activation strength, epoch boundaries marked
xs = list(range(n_total))
epoch_colors = ["#e41a1c", "#984ea3", "#4daf4a", "#a65628", "#ff7f00", "#377eb8"]

fig, ax = plt.subplots(figsize=(14, 5))

for rank, (val, ind) in enumerate(zip(top_vals, top_inds), 1):
    feat_idx   = ind.item()
    y          = full_acts[:, feat_idx].tolist()
    desc       = get_description(feat_idx)
    legend_str = f"F{feat_idx}" + (f": {desc}" if desc else "")
    print(f"  #{rank} F{feat_idx:<6} mean={val:.4f}  {desc}")
    ax.plot(xs, y, linewidth=1, alpha=0.85, label=legend_str)

for (label, pos), color in zip(epochs, epoch_colors):
    ax.axvline(x=pos, color=color, linestyle="--", linewidth=1, alpha=0.8, label=label)

ax.set_xticks([pos for _, pos in epochs])
ax.set_xticklabels(
    [f"{pos}\n{repr(decoded_tokens[pos])}" for _, pos in epochs],
    fontsize=7, rotation=20, ha="right",
)

ax.set_title(f"Experiment 006b — Top-3 features at layer {LAST_LAYER}, all tokens")
ax.set_xlabel("Token position")
ax.set_ylabel("Feature activation strength")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(_results_root, "exp006b_top3_layer31_tokens.png"), dpi=150)
plt.close(fig)

print("\nDone!")
