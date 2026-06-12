import os
import sys
import gzip
import json
import glob
import torch
import torch.nn.functional as F
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Experiment 008: Qwen3.5-27B with official Qwen-Scope SAEs (W80K-L0_50, TopK k=50)
# at layers 0, 15, 31, 47, 63. Same 6-epoch line-plot analysis as experiments 006/007.
# SAE weights loaded manually from HF (W_enc/b_enc), following the pattern in
# external/qwen_sae_prompt_code_3.5.py. Residuals captured via forward hooks.
# Neuronpedia descriptions loaded for layer 31 (the only layer available on S3).
# Token-snippet proxies used for layers 0, 15, 47, 63.
# Top-3 features ranked by layer-31 strength so descriptions are always available.

model_name   = "qwen3.5-27b"
hf_model_id  = "Qwen/Qwen3.5-27B"
sae_release  = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"
SAE_LAYERS   = [0, 15, 31, 47, 63]
TOP_FEATURES = 3
SAE_K        = 50   # L0_50 → top-50 active features per token

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment008")
_model_cache  = os.path.join(_data_root, model_name)
_sae_cache    = os.path.join(_data_root, model_name, "saes")
os.makedirs(_model_cache, exist_ok=True)
os.makedirs(_sae_cache, exist_ok=True)
os.makedirs(_results_root, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)


def _download(url, dest):
    tmp = dest + ".part"
    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            mb  = count * block_size / 1_048_576
            tot = total_size / 1_048_576
            sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f} / {tot:.1f} MB")
            sys.stdout.flush()
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
        sys.stdout.write("\n")
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


# Download SAE weights for the chosen layers
for layer in SAE_LAYERS:
    dest = os.path.join(_sae_cache, f"layer{layer}.sae.pt")
    if not os.path.exists(dest):
        url = f"https://huggingface.co/{sae_release}/resolve/main/layer{layer}.sae.pt"
        print(f"Downloading SAE layer {layer}...")
        _download(url, dest)

from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"+Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_model_cache)

model = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    cache_dir=_model_cache,
    torch_dtype=torch.bfloat16,
    device_map="auto" if device == "cuda" else "cpu",
)
model.eval()
print(f"+Model {hf_model_id} loaded")

# Load SAE encoder weights onto CPU (activations will also be moved to CPU via hooks)
saes = {}
for layer in SAE_LAYERS:
    d = torch.load(os.path.join(_sae_cache, f"layer{layer}.sae.pt"), map_location="cpu")
    saes[layer] = {
        "W_enc": d["W_enc"].to(dtype=torch.float32),
        "b_enc": d["b_enc"].to(dtype=torch.float32),
    }
    print(f"+SAE layer {layer} loaded  features={d['W_enc'].shape[0]}")


def sae_encode(layer, resid_cpu):
    """resid_cpu [pos, d_model] float32 → sparse feature_acts [pos, n_features]"""
    W = saes[layer]["W_enc"]
    b = saes[layer]["b_enc"]
    pre  = F.linear(resid_cpu, W, b)
    vals, inds = torch.topk(pre, SAE_K, dim=-1)
    vals = F.relu(vals)
    out  = torch.zeros_like(pre)
    out.scatter_(-1, inds, vals)
    return out


# Load Neuronpedia descriptions for layer 31 (the only layer with explanations on S3).
# Run src/download_qwen35_explanations.py first to populate this folder.
EXPLAINED_LAYER = 31
explanations    = {}   # feat_idx -> description string
_expl_dir = os.path.join(_data_root, model_name, "31-qwenscope-res-80k", "explanations")
for path in glob.glob(os.path.join(_expl_dir, "*.jsonl.gz")):
    with gzip.open(path) as f:
        for line in f:
            entry = json.loads(line)
            if "description" in entry:
                explanations[int(entry["index"])] = entry["description"]
if explanations:
    print(f"+Layer {EXPLAINED_LAYER}: {len(explanations)} descriptions loaded")
else:
    print(f"WARNING: no descriptions found in {_expl_dir} — run download_qwen35_explanations.py")


# Decode each token to a readable string (handles Qwen tiktoken BPE)
def _tok_str(token_id):
    return tokenizer.decode([token_id], skip_special_tokens=False)


# Use the chat template with enable_thinking=True so the template itself adds <think>
# at the end of the assistant prefix. Do NOT manually append <think> — that would
# produce a double <think> because the template already includes it.
_content = "Is Uranium necessary to stay healthy?"
messages = [{"role": "user", "content": _content}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)

input_device = next(model.parameters()).device
inputs    = tokenizer(prompt, return_tensors="pt").to(input_device)
n_prompt  = inputs["input_ids"].shape[1]

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

all_ids   = output_ids[0].cpu()   # [n_total]
n_total   = len(all_ids)
str_tokens = [_tok_str(i.item()) for i in all_ids]
print("Model output: {}".format("".join(str_tokens)))

# Epoch boundary detection
bos_id = tokenizer.bos_token_id
first_prompt_pos = 1 if (bos_id is not None and all_ids[0].item() == bos_id) else 0

stop_strs = {tokenizer.eos_token, "<|im_end|>"}
last_answer_pos = n_total - 1
while last_answer_pos > n_prompt and str_tokens[last_answer_pos] in stop_strs:
    last_answer_pos -= 1

last_question_pos = n_prompt - 1
while last_question_pos >= first_prompt_pos and "?" not in str_tokens[last_question_pos]:
    last_question_pos -= 1
if last_question_pos < first_prompt_pos:
    print("WARNING: '?' not found in prompt — falling back to last prompt token")
    last_question_pos = n_prompt - 1

end_think_pos = None
for i in range(n_prompt, n_total):
    if "</think>" in str_tokens[i]:
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
    ("Epoch a) first prompt token",   first_prompt_pos),
    ("Epoch b) last question token",  last_question_pos),
    ("Epoch c) first thinking token", first_think_pos),
    ("Epoch d) last thinking token",  last_think_pos),
    ("Epoch e) first answer token",   first_answer_pos),
    ("Epoch f) last answer token",    last_answer_pos),
]
epoch_short = ["a_first_prompt", "b_last_question", "c_first_think",
               "d_last_think",   "e_first_answer",  "f_last_answer"]

# Capture residual stream via forward hooks (one pass on the full generated sequence)
print("+Building residual stream cache via hooks...")

captured = {}
handles  = []

def _make_hook(layer_idx):
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[layer_idx] = h.detach().float().cpu()   # [1, pos, d_model]
    return _hook

for layer in SAE_LAYERS:
    handles.append(model.model.layers[layer].register_forward_hook(_make_hook(layer)))

with torch.no_grad():
    _ = model(all_ids.unsqueeze(0).to(input_device))

for h in handles:
    h.remove()

print("+Cache built")

# Apply SAEs and store per-epoch activations + full sequence for snippets
acts_by_layer      = {}
full_acts_by_layer = {}

for layer in SAE_LAYERS:
    resid        = captured[layer][0]       # [n_total, d_model]
    feature_acts = sae_encode(layer, resid) # [n_total, n_features]
    acts_by_layer[layer]      = {label: feature_acts[pos].clone() for label, pos in epochs}
    full_acts_by_layer[layer] = feature_acts
    print(f"+Layer {layer} processed")

del captured
if device == "cuda":
    torch.cuda.empty_cache()


def get_description(feat_idx):
    desc = explanations.get(feat_idx, "")
    if desc:
        return desc
    # Fallback: peak-token snippet from layer 31 activations
    peak_pos = int(full_acts_by_layer[EXPLAINED_LAYER][:, feat_idx].argmax())
    window   = str_tokens[max(0, peak_pos - 4): peak_pos + 6]
    snippet  = "".join(window).replace("\n", " ").strip()
    return "[top act] " + snippet


layers_x = SAE_LAYERS

for (label, pos), short in zip(epochs, epoch_short):
    # Rank by layer-31 strength so top features have Neuronpedia descriptions
    top_vals, top_inds = torch.topk(acts_by_layer[EXPLAINED_LAYER][label], TOP_FEATURES)

    print(f"\n{label}  token={repr(str_tokens[pos])}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for rank, (val, ind) in enumerate(zip(top_vals, top_inds), 1):
        feat_idx   = ind.item()
        y          = [acts_by_layer[l][label][feat_idx].item() for l in layers_x]
        peak_layer = layers_x[int(torch.tensor(y).argmax())]
        desc       = get_description(feat_idx)
        legend_str = f"F{feat_idx}" + (f": {desc}" if desc else "")
        print(f"  #{rank} F{feat_idx:<6} L31={val:.4f}  peak=L{peak_layer}  {desc}")
        ax.plot(layers_x, y, marker="o", markersize=6, label=legend_str)

    ax.set_title(f"Experiment 008 — {label}  [token: {repr(str_tokens[pos])}]")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature activation strength")
    ax.set_xticks(layers_x)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_results_root, f"exp008_{short}.png"), dpi=150)
    plt.close(fig)

print("\nDone!")
