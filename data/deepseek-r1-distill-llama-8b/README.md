# deepseek-r1-distill-llama-8b

Model weights and feature explanation JSONL files for DeepSeek-R1-Distill-Llama-8B and its Llama-Scope SAEs.
Used in `src/experiment003.py` and `src/experiment004.py`.

## Model weights

Downloaded via HuggingFace (`hub/`):

- **DeepSeek-R1-Distill-Llama-8B**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- **Llama-Scope-R1-Distill SAEs**: https://huggingface.co/fnlp/Llama-Scope-R1-Distill

## SAE subfolders

| Subfolder | Description |
|---|---|
| `0-llamascope-slimpj-openr1-res-32k/` | Layer-0 residual-stream SAE, 32k features, trained on 800M SlimPajama tokens |
| `20-llamascope-slimpj-openr1-res-32k/` | Layer-20 residual-stream SAE, 32k features, trained on 800M SlimPajama tokens |
| `31-llamascope-slimpj-openr1-res-32k/` | Layer-31 residual-stream SAE, 32k features, trained on 800M SlimPajama tokens |
