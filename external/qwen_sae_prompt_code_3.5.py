import os
import csv
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIGURATION
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
MODEL_PATH = '/s/poppy/a/ezzedina/llm-testing/Qwen3.5-27B'
SAE_PATH = '/s/poppy/a/ezzedina/llm-testing/qwen3.5sae/layer30.sae.pt'
DATA_DIR = '/s/poppy/a/ezzedina/llm-testing/top_2000_codes_cleaned'

OUTPUT_CSV = 'qwen3.5_sae_features_1000_prompt_code.csv'

LAYER_ID = 30
TOP_K = 50
MAX_LENGTH = 2048


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# ============================================================
# LOAD SAE
# ============================================================

print(f"Loading SAE from: {SAE_PATH}")

sae_dict = torch.load(SAE_PATH, map_location=DEVICE)

# ------------------------------------------------------------
# IMPORTANT:
# Convert SAE weights to float16 to match model activations
# Model hidden states are float16
# ------------------------------------------------------------

W_enc = sae_dict["W_enc"].to(device=DEVICE, dtype=torch.float16)
b_enc = sae_dict["b_enc"].to(device=DEVICE, dtype=torch.float16)

print("W_enc shape:", W_enc.shape)
print("b_enc shape:", b_enc.shape)

print("W_enc dtype:", W_enc.dtype)
print("b_enc dtype:", b_enc.dtype)

# Detect dimensions automatically
SAE_WIDTH = W_enc.shape[0]
D_MODEL = W_enc.shape[1]

print(f"D_MODEL   : {D_MODEL}")
print(f"SAE_WIDTH : {SAE_WIDTH}")

# ============================================================
# HOOK SETUP
# ============================================================

captured_activation = {}

def hook_fn(module, input, output):

    if isinstance(output, tuple):
        captured_activation["res"] = output[0]
    else:
        captured_activation["res"] = output

handle = model.model.layers[LAYER_ID].register_forward_hook(hook_fn)

# ============================================================
# DATA FILES
# ============================================================

code_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".py")
])[:1000]

print(f"Found {len(code_files)} Python files")

# ============================================================
# CSV HEADER
# ============================================================

header = ["filename", "summary"] + [
    f"feat_{i}" for i in range(SAE_WIDTH)
]

# ============================================================
# PROCESSING LOOP
# ============================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(header)

    for filename in tqdm(code_files):

        try:

            # ------------------------------------------------
            # READ FILE
            # ------------------------------------------------

            file_path = os.path.join(DATA_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------
            prompt = (
                f"Summarize the following code:\n\n"
                f"{code_content}"
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)

            # ------------------------------------------------
            # FIND LAST TOKEN POSITION OF CODE
            # ------------------------------------------------

            code_prefix = f"Code:\n{code_content}"
            code_prefix_tokens = tokenizer(
                code_prefix,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).input_ids

            code_last_token_pos = code_prefix_tokens.shape[1] - 1

            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            with torch.no_grad():

                _ = model(**inputs)

                if "res" not in captured_activation:
                    raise RuntimeError(
                        "Hook failed to capture activations."
                    )

                hidden_states = captured_activation["res"]

                # hidden_states:
                # [batch, seq_len, hidden_dim]

                if hidden_states.ndim != 3:
                    raise RuntimeError(
                        f"Unexpected hidden state shape: "
                        f"{hidden_states.shape}"
                    )

                # ------------------------------------------------
                # LAST TOKEN OF CODE VECTOR
                # ------------------------------------------------

                #last_token_vector = hidden_states[0, code_last_token_pos, :]
                last_token_vector = hidden_states[0, -1, :]
                # ------------------------------------------------
                # FORCE DTYPE MATCH
                # ------------------------------------------------

                last_token_vector = last_token_vector.to(device=DEVICE, dtype=torch.float16)

                # ------------------------------------------------
                # SAE ENCODING
                # ------------------------------------------------

                latents = torch.relu(
                    torch.matmul(
                        last_token_vector.unsqueeze(0),
                        W_enc.T
                    ) + b_enc
                )

                # ------------------------------------------------
                # TOP-K FEATURES
                # ------------------------------------------------

                vals, inds = torch.topk(
                    latents,
                    k=TOP_K,
                    dim=-1
                )

            # ------------------------------------------------
            # GENERATE SUMMARY
            # ------------------------------------------------

            with torch.no_grad():

                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_tokens = output_tokens[
                0,
                inputs.input_ids.shape[1]:
            ]

            summary_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            summary_text = summary_text.replace("\n", " ")

            # ------------------------------------------------
            # BUILD SPARSE VECTOR
            # ------------------------------------------------

            row_activations = [0.0] * SAE_WIDTH

            vals_cpu = vals[0].float().cpu().tolist()
            inds_cpu = inds[0].cpu().tolist()

            for value, index in zip(vals_cpu, inds_cpu):
                row_activations[index] = round(float(value), 6)

            # ------------------------------------------------
            # WRITE ROW
            # ------------------------------------------------

            writer.writerow(
                [filename, summary_text] + row_activations
            )

        except Exception as e:

            print(f"\nError processing {filename}")
            print(e)

# ============================================================
# CLEANUP
# ============================================================

handle.remove()

print("\n================================================")
print("DONE")
print(f"Results saved to: {OUTPUT_CSV}")
print("================================================")