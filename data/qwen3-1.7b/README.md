# qwen3-1.7b

Model weights and feature explanation JSONL files for Qwen3-1.7B with its
residual-stream BatchTopK SAE at layer 14. Used in `src/experiment002.py`.

## Model weights

Downloaded via HuggingFace (`hub/`):

- **Qwen3-1.7B**: https://huggingface.co/Qwen/Qwen3-1.7B
- **BatchTopK SAEs**: https://huggingface.co/adamkarvonen/qwen3-1.7b-saes

## SAE subfolders

| Subfolder | Description |
|---|---|
| `14-resid-batchtopk-65k__l0-80/` | Layer-14 residual-stream SAE, 65k features, L0=80, trainer_2 |
