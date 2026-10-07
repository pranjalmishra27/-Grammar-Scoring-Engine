import json
import os

p = r"e:\SHL\grammar_scoring_engine.ipynb"
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if "source" in cell and isinstance(cell["source"], list):
        for i, line in enumerate(cell["source"]):
            if "DATA_DIR   = " in line:
                cell["source"][i] = 'DATA_DIR   = "./data" if os.path.exists("./data") else "/kaggle/input/grammar-scoring"\n'
            if "WHISPER_SZ =" in line:
                cell["source"][i] = 'WHISPER_SZ = "base.en" if DEVICE == "cpu" else "small.en"\n'

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with local data directory fallback!")
