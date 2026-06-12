import bisect
import os
import gzip
import json
import glob
import random
import re
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as _Rect
import warnings
warnings.filterwarnings("ignore")

# Experiment 009b: DeepSeek-R1-Distill-Llama-8B, LlamaScope SAE at layer 31.
# Identical to experiment009 but ranks features by mean × activation_rate over
# selected tokens, penalizing features that fire on only a subset of them.

SEED         = 3
model_name   = "deepseek-r1-distill-llama-8b"
hf_model_id  = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
sae_release  = "llama_scope_r1_distill"
LAST_LAYER   = 31
TOP_FEATURES = 5
SELECTED_TOKENS = [
    "Uranium", "heavy", "metal", "radioactive", "medications", "toxic", "ingested", "consume",
    "trace", "amounts", "vitamins", "supplements", "biochemical", "essential", "Iron", "Zinc",
    "Copper", "excrete", "eliminate", "required", "benefits", "radiation", "exposure", "harmful", "health"
]
SELECTED_TOKENS = [
    "Uranium", "heavy", "metal", "toxic", "essential",
    "elements", "biochemical", "process", "harmful", "radiation",
]

_data_root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "experiment009b")
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

# Match selected words against the full reconstructed text, then map character
# positions back to token indices via bisect. This handles words that span
# multiple BPE subword tokens (e.g. "Uranium" → "ĠUr" + "anium").
_full_text = "".join(decoded_tokens)
_offsets   = []
_pos = 0
for tok in decoded_tokens:
    _offsets.append(_pos)
    _pos += len(tok)

_sel_positions = {}   # tok_idx → matched surface form (e.g. "Uranium", "processes")
_canonical     = {}   # tok_idx → canonical word from SELECTED_TOKENS
for word in SELECTED_TOKENS:
    for m in re.finditer(r'\b' + re.escape(word) + r'(?:es|s)?\b', _full_text, re.IGNORECASE):
        tok_idx = bisect.bisect_right(_offsets, m.start()) - 1
        if tok_idx not in _sel_positions:
            _sel_positions[tok_idx] = m.group(0)
            _canonical[tok_idx]     = word
selected_positions = sorted(_sel_positions)

if not selected_positions:
    print("WARNING: none of the selected tokens were found in the output — check SELECTED_TOKENS")
else:
    print(f"+Found {len(selected_positions)} matching positions: "
          f"{[(i, repr(_sel_positions[i])) for i in selected_positions]}")

# Epoch boundary positions (in full sequence space)
first_q_pos = next(
    (i for i, tok in enumerate(decoded_tokens) if "?" in tok), None
)
end_think_pos = next(
    (i for i, tok in enumerate(decoded_tokens) if "</think>" in tok), None
)
if first_q_pos is None:
    print("WARNING: '?' not found — prompt boundary line will be skipped")
if end_think_pos is None:
    print("WARNING: </think> not found — CoT boundary line will be skipped")

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

# Rank features by mean × activation_rate over selected positions.
# activation_rate = fraction of selected tokens where the feature fires (> 0).
# This penalizes features that spike on only a few tokens of interest.
sel_acts        = full_acts[selected_positions] if selected_positions else full_acts
mean_acts       = sel_acts.mean(dim=0)
activation_rate = (sel_acts > 0).float().mean(dim=0)
scores          = mean_acts * activation_rate
top_vals, top_inds = torch.topk(scores, TOP_FEATURES)

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
    f"{repr(_sel_positions[p])}\n@{p}"
    for p in selected_positions
]

fig, ax = plt.subplots(figsize=(max(10, len(selected_positions) * 1.1), 5))

for rank, (val, ind) in enumerate(zip(top_vals, top_inds), 1):
    feat_idx   = ind.item()
    y          = full_acts[selected_positions, feat_idx].tolist()
    desc       = get_description(feat_idx)
    legend_str = f"F{feat_idx}" + (f": {desc}" if desc else "")
    rate       = activation_rate[feat_idx].item()
    print(f"  #{rank} F{feat_idx:<6} score={val:.4f}  rate={rate:.2f}  {desc}")
    ax.plot(xs_sel, y, marker="o", markersize=5, linewidth=1, alpha=0.85, label=legend_str)

# Thick boundary lines between prompt / CoT / answer, mapped to compact x-axis.
# bisect_left gives the index of the first selected position >= boundary_pos,
# so subtracting 0.5 places the line between the two surrounding selected tokens.
for boundary_pos, label, color in [
    (first_q_pos,    "end of prompt", "#e41a1c"),
    (end_think_pos,  "end of CoT",    "#377eb8"),
]:
    if boundary_pos is not None:
        x_vline = bisect.bisect_left(selected_positions, boundary_pos) - 0.5
        ax.axvline(x=x_vline, color=color, linewidth=2.5, linestyle="-",
                   alpha=0.8, label=label)

ax.set_xticks(xs_sel)
ax.set_xticklabels(x_labels, fontsize=8, rotation=20, ha="right")
ax.set_title(
    f"Experiment 009b — Top-{TOP_FEATURES} features at layer {LAST_LAYER}, "
    f"selected tokens, ranked by mean×rate (seed={SEED})"
)
ax.set_xlabel("Selected token (@ = sequence position)")
ax.set_ylabel("Feature activation strength")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(
    os.path.join(_results_root, f"exp009b_top{TOP_FEATURES}_layer{LAST_LAYER}_selected.png"),
    dpi=150,
)
plt.close(fig)

# ── N × K_raw raw activation matrix ─────────────────────────────────────────
# Rows = TOP_FEATURES selected features; cols = K_raw matched positions.
top_inds_list = top_inds.tolist()
# full_acts[selected_positions] → [K_raw, n_features]; index N columns → [K_raw, N]; transpose → [N, K_raw]
mat_raw = full_acts[selected_positions][:, torch.tensor(top_inds_list)].numpy().T

# ── N × K collapsed matrix (mean-pool per canonical token) ───────────────────
K     = len(SELECTED_TOKENS)
mat_k = np.full((TOP_FEATURES, K), np.nan)
for k, word in enumerate(SELECTED_TOKENS):
    cols = [j for j, p in enumerate(selected_positions)
            if _canonical.get(p, "").lower() == word.lower()]
    if cols:
        mat_k[:, k] = mat_raw[:, cols].mean(axis=1)

# ── Token → strongest-feature assignment ─────────────────────────────────────
token_to_feat = {}
for k, word in enumerate(SELECTED_TOKENS):
    col = mat_k[:, k]
    if not np.all(np.isnan(col)):
        best_n = int(np.nanargmax(col))
        token_to_feat[word] = top_inds_list[best_n]

print("\nFeature → Token(s) assignment:")
_feat_to_tokens = {}
for word, fi in token_to_feat.items():
    _feat_to_tokens.setdefault(fi, []).append(word)
for n in range(TOP_FEATURES):
    fi = top_inds_list[n]
    print(f"  F{fi:<6} → {', '.join(_feat_to_tokens.get(fi, [])) or '(none)'}")

# ── Heatmap: N × K matrix ────────────────────────────────────────────────────
_feat_labels = [f"F{top_inds_list[n]}" for n in range(TOP_FEATURES)]

fig2, ax2 = plt.subplots(figsize=(max(8, K * 1.0), max(3, TOP_FEATURES * 1.0 + 1.5)))
_cmap = plt.cm.Blues.copy()
_cmap.set_bad(color="#dddddd")
_masked = np.ma.masked_invalid(mat_k)
im = ax2.imshow(_masked, aspect="auto", cmap=_cmap, vmin=0)
plt.colorbar(im, ax=ax2, label="Mean activation")

ax2.set_xticks(range(K))
ax2.set_xticklabels(SELECTED_TOKENS, rotation=35, ha="right", fontsize=9)
ax2.set_yticks(range(TOP_FEATURES))
ax2.set_yticklabels(_feat_labels, fontsize=7)

for n in range(TOP_FEATURES):
    for k in range(K):
        val = mat_k[n, k]
        if np.isnan(val):
            ax2.text(k, n, "—", ha="center", va="center", fontsize=8, color="#999999")
        else:
            col      = mat_k[:, k]
            is_best  = (int(np.nanargmax(col)) == n)
            ax2.text(k, n, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, fontweight="bold" if is_best else "normal")
            if is_best:
                ax2.add_patch(_Rect((k - 0.5, n - 0.5), 1, 1,
                                    fill=False, edgecolor="#2166ac", linewidth=2.2))

ax2.set_title(
    f"Experiment 009b — Feature × Token heatmap  (N={TOP_FEATURES} features, K={K} tokens)\n"
    f"Cells = mean activation over matched positions; blue border = assigned feature"
)
ax2.set_xlabel("Canonical token")
ax2.set_ylabel("Feature")
fig2.tight_layout()
_heatmap_path = os.path.join(_results_root, f"exp009b_heatmap_top{TOP_FEATURES}_layer{LAST_LAYER}.png")
fig2.savefig(_heatmap_path, dpi=150)
plt.close(fig2)

print("\nDone!")
