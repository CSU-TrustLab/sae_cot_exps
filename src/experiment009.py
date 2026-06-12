import os
import gzip
import json
import glob
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Experiment 009: DeepSeek-R1-Distill-Llama-8B, LlamaScope SAE at layer 31.
# Same analysis as experiment006b (top-3 features by mean activation, single token-axis plot)
# but uses two GPUs via device_map="auto" and a fixed random seed for reproducibility.

SEED         = 3
model_name   = "deepseek-r1-distill-llama-8b"
hf_model_id  = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
sae_release  = "llama_scope_r1_distill"
LAST_LAYER   = 31
TOP_FEATURES = 3
SELECTED_TOKENS = [
    "Uranium", "heavy", "metal", "toxic", "essential",
    "elements", "biochemical", "process", "harmful", "radiation",
]

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment009")
_model_cache  = os.path.join(_data_root, model_name)
os.makedirs(_model_cache, exist_ok=True)
os.makedirs(_results_root, exist_ok=True)
os.environ.setdefault("HF_HOME", _model_cache)

from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402
from sae_lens import SAE                                         # noqa: E402

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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

tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir=_model_cache)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    cache_dir=_model_cache,
    torch_dtype=torch.float32,
    device_map="auto",
)
model.eval()
print(f"+Model {hf_model_id} loaded")

prompt = "Is Uranium necessary to stay healthy?"

# Ġ-trick: DeepSeek's normalizer strips ASCII spaces; prepend BOS manually
_prompt_bpe = prompt.replace(" ", "Ġ")
_prompt_ids = tokenizer(_prompt_bpe, add_special_tokens=False,
                         return_tensors="pt").input_ids
bos_id      = tokenizer.bos_token_id
input_ids   = torch.cat([torch.tensor([[bos_id]]), _prompt_ids], dim=1)
n_prompt    = input_ids.shape[1]
print("DEBUG str_tokens(input)[:15]:",
      [tokenizer.convert_ids_to_tokens(t.item()) for t in input_ids[0]][:15])

input_device = next(model.parameters()).device

# Re-fix seed immediately before generation so sampling is reproducible
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with torch.no_grad():
    output_ids = model.generate(
        input_ids.to(input_device),
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

all_ids        = output_ids[0].cpu()
n_total        = len(all_ids)
str_tokens     = [tokenizer.convert_ids_to_tokens(t.item()) for t in all_ids]
decoded_tokens = [_decode_str_token(t) for t in str_tokens]
print("Model output: {}".format("".join(decoded_tokens)))

# Find all positions in the generated sequence whose decoded token matches SELECTED_TOKENS
sel_lower = {t.lower() for t in SELECTED_TOKENS}
selected_positions = [
    i for i, tok in enumerate(decoded_tokens)
    if tok.strip().lower() in sel_lower
]
if not selected_positions:
    print("WARNING: none of the selected tokens were found in the output — check SELECTED_TOKENS")
else:
    print(f"+Found {len(selected_positions)} matching positions: "
          f"{[(i, repr(decoded_tokens[i].strip())) for i in selected_positions]}")

# Capture residual stream at layer 31 via forward hook (single pass on full sequence)
print("+Building residual stream cache...")
captured = {}

def _hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    captured[LAST_LAYER] = h.detach().float().cpu()   # [1, n_total, d_model]

handle = model.model.layers[LAST_LAYER].register_forward_hook(_hook)
with torch.no_grad():
    _ = model(all_ids.unsqueeze(0).to(input_device))
handle.remove()
print("+Cache built")

# Apply SAE at layer 31
sae_id    = f"l{LAST_LAYER}r_800m_slimpajama"
sae       = SAE.from_pretrained(sae_release, sae_id, device="cpu")
resid     = captured[LAST_LAYER][0]           # [n_total, d_model]
full_acts = sae.encode(resid).cpu()           # [n_total, n_features]
del sae, captured
if device == "cuda":
    torch.cuda.empty_cache()
print(f"+Layer {LAST_LAYER} processed  shape={list(full_acts.shape)}")

# Top-3 features by mean activation over selected token positions only
sel_acts  = full_acts[selected_positions] if selected_positions else full_acts
mean_acts = sel_acts.mean(dim=0)
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
    window   = decoded_tokens[max(0, peak_pos - 4): peak_pos + 6]
    return "[top act] " + "".join(window).replace("\n", " ").strip()


# Single plot: x = selected token (compact index), y = activation strength
xs_sel   = list(range(len(selected_positions)))
x_labels = [
    f"{repr(decoded_tokens[p].strip())}\n@{p}"
    for p in selected_positions
]

fig, ax = plt.subplots(figsize=(max(10, len(selected_positions) * 1.1), 5))

for rank, (val, ind) in enumerate(zip(top_vals, top_inds), 1):
    feat_idx   = ind.item()
    y          = full_acts[selected_positions, feat_idx].tolist()
    desc       = get_description(feat_idx)
    legend_str = f"F{feat_idx}" + (f": {desc}" if desc else "")
    print(f"  #{rank} F{feat_idx:<6} mean(sel)={val:.4f}  {desc}")
    ax.plot(xs_sel, y, marker="o", markersize=5, linewidth=1, alpha=0.85, label=legend_str)

ax.set_xticks(xs_sel)
ax.set_xticklabels(x_labels, fontsize=8, rotation=20, ha="right")
ax.set_title(
    f"Experiment 009 — Top-3 features at layer {LAST_LAYER}, "
    f"selected tokens (seed={SEED})"
)
ax.set_xlabel("Selected token (@ = sequence position)")
ax.set_ylabel("Feature activation strength")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(
    os.path.join(_results_root, f"exp009_top3_layer{LAST_LAYER}_selected.png"),
    dpi=150,
)
plt.close(fig)

print("\nDone!")
