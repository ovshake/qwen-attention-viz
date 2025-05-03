# scripts/smoke_test.py
from pathlib import Path, PurePath
import torch, transformers, rwkv
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    # "mamba":  "models/mamba-790m",
    "rwkv":   "models/rwkv-5-world-1b5",
    # "pythia": "models/pythia-1b-deduped",   # new Transformer baseline
}

for name, path in MODELS.items():
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True)
    model.to("cuda:0")
    prompt = tok("The quick brown fox", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**prompt, max_new_tokens=5)
    print(name, tok.decode(out[0]))