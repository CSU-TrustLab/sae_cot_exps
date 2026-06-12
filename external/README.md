# External

`tutorial.py` is adapted from the SAELens tutorial notebook:

https://colab.research.google.com/github/decoderesearch/SAELens/blob/main/tutorials/tutorial_2_0.ipynb

---

`qwen_sae_prompt_code_3.5.py` — contributed by **Ezzedine Amari**.

Runs Qwen3.5-27B with a custom SAE at layer 30. For each of 1,000 Python files, it
builds a "Summarize the following code:" prompt, captures the residual stream at the
last token position via a forward hook, encodes it with the SAE encoder
(`W_enc`, `b_enc`), and records the top-50 active features. The model also generates
a short summary of the code. Results are written to a CSV with one row per file:
`filename`, `summary`, and one column per SAE feature.
