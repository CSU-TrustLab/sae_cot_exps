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

# Experiment 007b: Qwen3-1.7B, BatchTopK SAE at layer 14 only (65k features, k=80,
# adamkarvonen/qwen3-1.7b-saes trainer_0). Top-3 features selected by mean activation
# across all token positions at layer 14 (the only layer with NL descriptions).
# Single plot: x = token position, y = feature activation strength, with epoch markers.

model_name  = "qwen3-1.7b"
sae_release = "adamkarvonen/qwen3-1.7b-saes"
LAYER       = 14
SAE_ID      = "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_14/trainer_0"
SAE_NAME    = "14-resid-batchtopk-65k__l0-80"
TOP_FEATURES = 3

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment007b")
_model_cache  = os.path.join(_data_root, model_name)
os.makedirs(_model_cache, exist_ok=True)
os.makedirs(_results_root, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)

from sae_lens import SAE, HookedSAETransformer                                    # noqa: E402
from sae_lens.loading.pretrained_sae_loaders import (                             # noqa: E402
    dictionary_learning_sae_huggingface_loader_1,
)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"+Device: {device}")

model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device)
print(f"+Model {model_name} loaded")

sae, _, _ = SAE.from_pretrained(
    sae_release, SAE_ID, device=device,
    converter=dictionary_learning_sae_huggingface_loader_1,
)
prepend_bos = sae.cfg.metadata.prepend_bos
print(f"+SAE layer {LAYER} loaded  hook={sae.cfg.metadata.hook_name}")

# Load Neuronpedia descriptions for layer 14
exps = {}
data_dir = os.path.join(_data_root, model_name, SAE_NAME, "explanations")
for path in glob.glob(os.path.join(data_dir, "*.jsonl.gz")):
    with gzip.open(path) as f:
        for line in f:
            entry = json.loads(line)
            if "description" in entry:
                exps[int(entry["index"])] = entry["description"]
print(f"+Layer {LAYER}: {len(exps)} descriptions loaded")

prompt = "Is Uranium necessary to stay healthy? <think>"
prompt = "Is a librarian trained to fly a plane? <think>"

output_tokens = model.generate(
    prompt,
    max_new_tokens=550,
    stop_at_eos=True,
    prepend_bos=prepend_bos,
    return_type="tokens",
)   # [1, n_total]

str_tokens = model.to_str_tokens(output_tokens[0])
print("Model output: {}".format("".join(str_tokens)))
n_total  = len(str_tokens)
n_prompt = model.to_tokens(prompt, prepend_bos=prepend_bos).shape[1]

_stop_tokens = {model.tokenizer.eos_token, "<|im_end|>"}
last_answer_pos = n_total - 1
while last_answer_pos > n_prompt and str_tokens[last_answer_pos] in _stop_tokens:
    last_answer_pos -= 1

bos_str = model.tokenizer.bos_token
first_prompt_pos = 1 if (n_total > 0 and str_tokens[0] == bos_str) else 0

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
while first_think_pos < end_think_pos and str_tokens[first_think_pos].strip() == "":
    first_think_pos += 1

last_think_pos = end_think_pos - 1
while last_think_pos > n_prompt and str_tokens[last_think_pos].strip() == "":
    last_think_pos -= 1

first_answer_pos = end_think_pos + 1
while first_answer_pos < n_total and str_tokens[first_answer_pos].strip() == "":
    first_answer_pos += 1

epochs = [
    ("a) first prompt",   first_prompt_pos),
    ("b) last question",  last_question_pos),
    ("c) first thinking", first_think_pos),
    ("d) last thinking",  last_think_pos),
    ("e) first answer",   first_answer_pos),
    ("f) last answer",    last_answer_pos),
]

# Cache residual stream then apply SAE at layer 14; keep full sequence activations
print("+Building residual stream cache...")
_, resid_cache = model.run_with_cache(
    output_tokens,
    names_filter=lambda name: name.endswith("hook_resid_post"),
    return_type=None,
)
print("+Cache built")

resid     = resid_cache[sae.cfg.metadata.hook_name][0]   # [n_total, d_model]
full_acts = sae.encode(resid).cpu()                       # [n_total, n_features]
del sae, resid_cache
if device == "cuda":
    torch.cuda.empty_cache()
print(f"+Layer {LAYER} processed  shape={list(full_acts.shape)}")

# Top-3 features by mean activation, excluding outlier token positions.
# Positions where the peak feature value is ≥10× the median peak are skipped
# (this removes BOS and other tokens whose activations dwarf the rest).
token_peaks  = full_acts.max(dim=1).values          # [n_total] max act per position
median_peak  = token_peaks.median()
normal_mask  = token_peaks < 10.0 * median_peak     # True for non-outlier positions
n_excluded   = int((~normal_mask).sum())
print(f"+Outlier filter: excluding {n_excluded} position(s) with peak ≥10× median ({median_peak:.4f})")
mean_acts = full_acts[normal_mask].mean(dim=0)
top_vals, top_inds = torch.topk(mean_acts, TOP_FEATURES)


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
    ax.plot(xs[1:], y[1:], linewidth=1, alpha=0.85, label=legend_str)

for (label, pos), color in zip(epochs, epoch_colors):
    ax.axvline(x=pos, color=color, linestyle="--", linewidth=1, alpha=0.8, label=label)

ax.set_xticks([pos for _, pos in epochs])
ax.set_xticklabels(
    [f"{pos}\n{repr(str_tokens[pos])}" for _, pos in epochs],
    fontsize=7, rotation=20, ha="right",
)

ax.set_title(f"Experiment 007b — Top-3 features at layer {LAYER}, all tokens")
ax.set_xlabel("Token position")
ax.set_ylabel("Feature activation strength")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(_results_root, f"exp007b_top3_layer{LAYER}_tokens.png"), dpi=150)
plt.close(fig)

print("\nDone!")
