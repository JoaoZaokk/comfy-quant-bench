"""Acrescenta VoidLoadConditioningFull ao pacote rastreado custom_nodes/comfy-void-stage-tools
(o stub em ComfyUI/custom_nodes/ carrega este arquivo). Motivo medido em 2026-09-14: o
LTXVSaveConditioning/LTXVLoadConditioning do ComfyUI-LTXVideo perde a chave
`unprocessed_ltxav_embeds` que o LTX 2.3 exige, e o render sai ruido (MAE 75,9)."""
import io

p = 'custom_nodes/comfy-void-stage-tools/__init__.py'
s = io.open(p, encoding='utf-8').read()

NODE = '''
class VoidLoadConditioningFull:
    """Loads a conditioning file that carries ALL of the encoder's options, not only the tensor.

    Why this exists (measured 2026-09-14, bench/ltx23/cond_identity_ltxv_saver): ComfyUI-LTXVideo's
    LTXVSaveConditioning keeps only the tensor and an attention mask. The LTX 2.3 text encoder
    returns ``{"unprocessed_ltxav_embeds": True}`` beside the tensor (comfy/text_encoders/lt.py:201-204)
    and the model runs caption_projection + the embeddings connectors only when that key arrives
    (comfy/model_base.py:1185 -> comfy/ldm/lightricks/av_model.py:583). Loaded back through
    LTXVLoadConditioning the key is gone, the 6144-wide context is taken as already processed, and
    the render is noise: MAE 75.9 / SSIM 0.19 against the live encoder on the same W4A8 model.

    File layout, written by tools/ltx_encode_lowcommit.py: ``conditioning_data_{i}`` (float32 by
    default -- the dtype the live encoder hands the sampler), ``opt_{i}_{key}`` for tensor options,
    ``__metadata__["options_{i}"]`` as JSON for the non-tensor ones, ``num_conditionings``.
    A file without ``options_{i}`` is refused instead of guessed at.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("embeddings") or [""]
        return {
            "required": {
                "file_name": (sorted(files),),
                "device": (["cpu", "gpu"], {"default": "gpu"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "VOID/Conditioning"

    @classmethod
    def IS_CHANGED(cls, file_name, device):
        path = folder_paths.get_full_path_or_raise("embeddings", file_name)
        st = os.stat(path)
        return f"{st.st_size}:{st.st_mtime_ns}:{device}"

    def load(self, file_name, device):
        import safetensors

        path = folder_paths.get_full_path_or_raise("embeddings", file_name)
        target = model_management.get_torch_device() if device == "gpu" else torch.device("cpu")
        out = []
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
            n = int(meta.get("num_conditionings", "0"))
            if n <= 0:
                raise ValueError(f"VOID conditioning: no num_conditionings in {file_name}")
            for i in range(n):
                if f"options_{i}" not in meta:
                    raise ValueError(
                        f"VOID conditioning: {file_name} has no options_{i}; it was not written by "
                        "tools/ltx_encode_lowcommit.py and its encoder options are unknown -- refusing to guess"
                    )
                opts = json.loads(meta[f"options_{i}"])
                prefix = f"opt_{i}_"
                for k in keys:
                    if k.startswith(prefix):
                        opts[k[len(prefix):]] = f.get_tensor(k).to(target)
                tensor = f.get_tensor(f"conditioning_data_{i}").to(target)
                out.append([tensor, opts])
        return (out,)

'''
anchor = "NODE_CLASS_MAPPINGS = {\n"
assert s.count(anchor) == 1
s = s.replace(anchor, NODE + anchor)
s = s.replace('    "VoidRecomposeCroppedVideo": VoidRecomposeCroppedVideo,\n}',
              '    "VoidRecomposeCroppedVideo": VoidRecomposeCroppedVideo,\n    "VoidLoadConditioningFull": VoidLoadConditioningFull,\n}')
s = s.replace('    "VoidRecomposeCroppedVideo": "VOID Recompose Cropped Video (In Place)",\n}',
              '    "VoidRecomposeCroppedVideo": "VOID Recompose Cropped Video (In Place)",\n    "VoidLoadConditioningFull": "VOID Load Conditioning (all options)",\n}')
assert s.count("VoidLoadConditioningFull") == 4, s.count("VoidLoadConditioningFull")
io.open(p, 'w', encoding='utf-8').write(s)
import ast
ast.parse(s)
print("VoidLoadConditioningFull added; syntax ok")
