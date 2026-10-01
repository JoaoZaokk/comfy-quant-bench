"""Sobe as variantes FP8/INT8 dos residuos do TE LTX nos dois repos do dono (pedido de 24/09).
O token de escrita e' lido pela propria huggingface_hub via HF_TOKEN_PATH; este script nao o le."""
import os, time
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi()
print("usuario:", api.whoami()["name"], flush=True)
TE = "ComfyUI/models/text_encoders"; CK = "P:/ComfyBench/checkpoints"; B = "bench/te_residuos"; S = r"C:/Users/joaoz/AppData/Local/Temp/claude/F--COMFY-PORTABLE/e5d98f7f-6e44-4b49-93be-731d2eed2d59/scratchpad"
pacotes = {
    "JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8": [
        (f"{TE}/gemma_3_12B_it_heretic_w4a8_embfp8.safetensors", "residual_fp8/gemma_3_12B_it_heretic_w4a8_embfp8.safetensors"),
        (f"{TE}/gemma_3_12B_it_heretic_w4a8_embfp8.quant.json", "residual_fp8/gemma_3_12B_it_heretic_w4a8_embfp8.quant.json"),
        (f"{TE}/gemma_3_12B_it_heretic_w4a8_embint8.safetensors", "residual_int8/gemma_3_12B_it_heretic_w4a8_embint8.safetensors"),
        (f"{TE}/gemma_3_12B_it_heretic_w4a8_embint8.quant.json", "residual_int8/gemma_3_12B_it_heretic_w4a8_embint8.quant.json"),
        (f"{B}/resumo.json", "residual_test/encode_vs_production.json"),
        (f"{B}/resumo_emb_int8_proj_int8_emb_fp8_proj_fp8.json", "residual_test/encode_one_piece_at_a_time.json"),
        (f"{B}/render_eros.json", "residual_test/i2v_render_numeric.json"),
        (f"{S}/README_gemma_novo.md", "README.md"),
    ],
    "JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot": [
        (f"{CK}/ltx-2.3_text_projection_fp8.safetensors", "text_projection/ltx-2.3_text_projection_fp8.safetensors"),
        (f"{CK}/ltx-2.3_text_projection_fp8.quant.json", "text_projection/ltx-2.3_text_projection_fp8.quant.json"),
        (f"{CK}/ltx-2.3_text_projection_int8.safetensors", "text_projection/ltx-2.3_text_projection_int8.safetensors"),
        (f"{CK}/ltx-2.3_text_projection_int8.quant.json", "text_projection/ltx-2.3_text_projection_int8.quant.json"),
        (f"{S}/README_ltx23_novo.md", "README.md"),
    ],
}
for repo, arqs in pacotes.items():
    for local, _ in arqs:
        assert os.path.isfile(local), local
    t = time.time()
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in arqs]
    c = api.create_commit(repo, ops, commit_message="Add FP8/INT8 re-encodings of the BF16 text-encoder residue, with measurements")
    print(repo, "commit", c.oid, f"{time.time() - t:.0f} s", flush=True)
    info = api.model_info(repo, files_metadata=True)
    tam = {s.rfilename: s.size for s in info.siblings}
    for src, dst in arqs:
        print("  ", dst, "OK" if tam.get(dst) == os.path.getsize(src) else f"TAMANHO DIFERENTE {tam.get(dst)} vs {os.path.getsize(src)}", flush=True)
print("FIM", flush=True)
