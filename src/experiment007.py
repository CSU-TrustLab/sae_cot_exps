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

# Experiment 007: Qwen3-1.7B with BatchTopK SAEs at layers 7, 14, 21 (65k features, k=80,
# adamkarvonen/qwen3-1.7b-saes trainer_0). Runs the model once to cache the clean residual
# stream, then applies each SAE independently. For each of the 6 tokens of interest, finds
# the top 3 features by mean activation across the 3 layers and saves a line plot.
# Thinking mode forced by appending <think> to the prompt.
# Descriptions: layer 14 from Neuronpedia JSONL; layers 7/21 from peak-token snippets.

model_name  = "qwen3-1.7b"
sae_release = "adamkarvonen/qwen3-1.7b-saes"
TOP_FEATURES = 3

SAE_CFGS = [
    {"layer": 7,  "sae_id": "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_7/trainer_0",
                  "sae_name": "7-resid-batchtopk-65k__l0-80"},
    {"layer": 14, "sae_id": "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_14/trainer_0",
                  "sae_name": "14-resid-batchtopk-65k__l0-80"},
    {"layer": 21, "sae_id": "saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_21/trainer_0",
                  "sae_name": "21-resid-batchtopk-65k__l0-80"},
]

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment007")
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
print("+Device: {}".format(device))

model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device)
print("+Model {} loaded".format(model_name))

# Load all three SAEs up front; grab prepend_bos from the first
saes = {}
prepend_bos = None
for cfg in SAE_CFGS:
    sae = SAE.from_pretrained(
        sae_release, cfg["sae_id"], device=device,
        converter=dictionary_learning_sae_huggingface_loader_1,
    )
    saes[cfg["layer"]] = sae
    if prepend_bos is None:
        prepend_bos = sae.cfg.metadata.prepend_bos
    print(f"+SAE layer {cfg['layer']} loaded  hook={sae.cfg.metadata.hook_name}")

# Load Neuronpedia descriptions where available (layer 14 only)
explanations = {}   # layer -> {feat_idx: description}
for cfg in SAE_CFGS:
    exps = {}
    data_dir = os.path.join(_data_root, model_name, cfg["sae_name"], "explanations")
    for path in glob.glob(os.path.join(data_dir, "*.jsonl.gz")):
        with gzip.open(path) as f:
            for line in f:
                entry = json.loads(line)
                if "description" in entry:
                    exps[int(entry["index"])] = entry["description"]
    explanations[cfg["layer"]] = exps
    if exps:
        print(f"+Layer {cfg['layer']}: {len(exps)} descriptions loaded")

# <think> appended to force the model into thinking mode
#prompt = "Is Uranium necessary to stay healthy? <think>"
prompt = "Is a librarian trained to fly a plane? <think>"
answer = " No"

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

# Last question token: '?'
last_question_pos = n_prompt - 1
while last_question_pos >= first_prompt_pos and str_tokens[last_question_pos] != "?":
    last_question_pos -= 1
if last_question_pos < first_prompt_pos:
    print("WARNING: '?' not found in prompt — falling back to last prompt token")
    last_question_pos = n_prompt - 1

# Find </think>
end_think_pos = None
for i in range(n_prompt, n_total):
    if str_tokens[i] == "</think>":
        end_think_pos = i
        break
if end_think_pos is None:
    print("WARNING: </think> not found — thinking epochs may be inaccurate")
    end_think_pos = n_prompt

# First thinking token: skip leading whitespace after prompt
first_think_pos = n_prompt
while first_think_pos < end_think_pos and str_tokens[first_think_pos].strip() == "":
    first_think_pos += 1

# Last thinking token
last_think_pos = end_think_pos - 1
while last_think_pos > n_prompt and str_tokens[last_think_pos].strip() == "":
    last_think_pos -= 1

# First answer token
first_answer_pos = end_think_pos + 1
while first_answer_pos < n_total and str_tokens[first_answer_pos].strip() == "":
    first_answer_pos += 1

epochs = [
    ("Epoch a) first prompt token",   first_prompt_pos),
    ("Epoch b) last question token",  last_question_pos),
    ("Epoch c) first thinking token", first_think_pos),
    ("Epoch d) last thinking token",  last_think_pos),
    ("Epoch e) first answer token",   first_answer_pos),
    ("Epoch f) last answer token",    last_answer_pos),
]
epoch_short = ["a_first_prompt", "b_last_question", "c_first_think",
               "d_last_think",   "e_first_answer",  "f_last_answer"]

# Cache clean residual stream (no SAEs hooked)
print("+Building residual stream cache...")
_, resid_cache = model.run_with_cache(
    output_tokens,
    names_filter=lambda name: name.endswith("hook_resid_post"),
    return_type=None,
)
print("+Cache built")

# Apply each SAE to the cached residuals
# For layers without pre-loaded descriptions, keep full-sequence acts for snippet fallback
acts_by_layer      = {}   # layer -> {epoch_label: tensor[n_features]}
full_acts_by_layer = {}   # layer -> tensor[n_total, n_features]  (snippet-only layers)

for cfg in SAE_CFGS:
    layer = cfg["layer"]
    sae   = saes[layer]
    resid = resid_cache[sae.cfg.metadata.hook_name][0]   # [pos, d_model]
    feature_acts = sae.encode(resid)                      # [pos, n_features]
    acts_by_layer[layer] = {label: feature_acts[pos].cpu() for label, pos in epochs}
    if not explanations[layer]:
        full_acts_by_layer[layer] = feature_acts.cpu()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"+Layer {layer} processed")

del resid_cache, saes
if device == "cuda":
    torch.cuda.empty_cache()


def get_description(layer, feat_idx):
    """Return Neuronpedia description if available, else a peak-token snippet."""
    desc = explanations.get(layer, {}).get(feat_idx, "")
    if desc:
        return desc
    if layer in full_acts_by_layer:
        peak_pos = int(full_acts_by_layer[layer][:, feat_idx].argmax())
        window   = str_tokens[max(0, peak_pos - 4): peak_pos + 6]
        return "[top act] " + "".join(window).replace("\n", " ").strip()
    return ""


# For each epoch: top-3 features by mean activation, line plot over layers
layers_x = [cfg["layer"] for cfg in SAE_CFGS]

for (label, pos), short in zip(epochs, epoch_short):
    # Rank by layer-14 strength so top features have Neuronpedia descriptions
    top_vals, top_inds = torch.topk(acts_by_layer[14][label], TOP_FEATURES)

    print(f"\n{label}  token={repr(str_tokens[pos])}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for rank, (val, ind) in enumerate(zip(top_vals, top_inds), 1):
        feat_idx   = ind.item()
        y          = [acts_by_layer[layer][label][feat_idx].item() for layer in layers_x]
        peak_layer = layers_x[int(torch.tensor(y).argmax())]
        desc       = get_description(peak_layer, feat_idx)
        legend_str = f"F{feat_idx}" + (f": {desc}" if desc else "")
        print(f"  #{rank} F{feat_idx:<6} L14={val:.4f}  peak=L{peak_layer}  {desc}")
        ax.plot(layers_x, y, marker="o", markersize=6, label=legend_str)

    ax.set_title(f"Experiment 007 — {label}  [token: {repr(str_tokens[pos])}]")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature activation strength")
    ax.set_xticks(layers_x)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(_results_root, f"exp007_{short}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

print("\nDone!")
