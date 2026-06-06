FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    sae-lens \
    transformer-lens \
    && pip install --no-cache-dir --force-reinstall \
    "torch==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# Fix transformer-lens 3.3.0 bug: Gemma-specific unsqueeze(1) corrupts token shape in to_str_tokens
RUN python3 - <<'EOF'
path = "/opt/conda/lib/python3.11/site-packages/transformer_lens/HookedTransformer.py"
with open(path) as f:
    src = f.read()
bad = (
    "                # Gemma tokenizer expects a batch dimension\n"
    "                if \"gemma\" in self.tokenizer.name_or_path and tokens.ndim == 1:\n"
    "                    tokens = tokens.unsqueeze(1)\n"
)
assert bad in src, "Patch target not found — check transformer-lens version"
src = src.replace(bad, "")
with open(path, "w") as f:
    f.write(src)
print("Patch applied successfully")
EOF

COPY src/ ./src/
