"""<nome> = base + ModelMemoryUsageFactorOverride(fator) [+ LTXVChunkFeedForward(chunks)] entre o Power Lora (853)
e os LTXReferenceConditioning (860/870) que alimentam os dois samplers."""
import json, sys
nome, fator, chunks = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
g = json.load(open("prompt_base_api.json"))
for n in g.values():
    if n["class_type"] == "VHS_VideoCombine":
        n["inputs"]["filename_prefix"] = n["inputs"]["filename_prefix"].replace("diag_base", "diag_" + nome)
g["951"] = {"class_type": "ModelMemoryUsageFactorOverride", "inputs": {"model": ["853", 0], "memory_usage_factor": fator}}
fim = ["951", 0]
if chunks > 1:
    g["952"] = {"class_type": "LTXVChunkFeedForward", "inputs": {"model": ["951", 0], "chunks": chunks, "dim_threshold": 4096}}
    fim = ["952", 0]
for k in ("860", "870"):
    assert g[k]["inputs"]["model"] == ["853", 0], g[k]["inputs"]["model"]
    g[k]["inputs"]["model"] = fim
json.dump(g, open(f"prompt_{nome}_api.json", "w"), indent=1)
print(nome, fator, chunks, "->", fim)
