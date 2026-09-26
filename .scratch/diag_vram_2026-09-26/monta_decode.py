"""Grafos so de decode a partir dos latentes salvos pela r1 (copiados para ComfyUI/input):
d_base   = VAEDecodeTiled 512/64/64/8 -> VHS_VideoCombine (o caminho do workflow)
d_stream = StreamingTiledDecodeVideoCombine com os mesmos parametros
(d_fp16 = d_base com o servidor em --fp16-intermediates; o grafo e o mesmo.)
VAE de video avulso LTX23_video_vae_bf16 (byte a byte igual ao do checkpoint, compara_vae.py)."""
import json, sys
video, audio = sys.argv[1], sys.argv[2]
comum = {
    "1": {"class_type": "VAELoader", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},
    "2": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "LTX23_audio_vae_bf16.safetensors"}},
    "3": {"class_type": "LoadLatent", "inputs": {"latent": video}},
    "4": {"class_type": "LoadLatent", "inputs": {"latent": audio}},
    "5": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["4", 0], "audio_vae": ["2", 0]}},
    "6": {"class_type": "AudioAdjustVolume", "inputs": {"volume": 0, "audio": ["5", 0]}},
}
vhs = {"frame_rate": 24.0, "loop_count": 0, "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 24,
       "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True, "audio": ["6", 0]}
base = dict(comum)
base["7"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["3", 0], "vae": ["1", 0], "tile_size": 512,
             "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
for nome in ("d_base", "d_fp16"):
    g = dict(base)
    g["8"] = {"class_type": "VHS_VideoCombine", "inputs": {**vhs, "images": ["7", 0], "filename_prefix": "Eros/diag_" + nome}}
    json.dump(g, open(f"prompt_{nome}_api.json", "w"), indent=1)
s = dict(comum)
s["9"] = {"class_type": "StreamingTiledDecodeVideoCombine", "inputs": {
    "samples": ["3", 0], "vae": ["1", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
    "frame_rate": 24.0, "filename_prefix": "Eros/diag_d_stream", "pix_fmt": "yuv420p", "crf": 24,
    "save_metadata": True, "save_output": True, "audio": ["6", 0]}}
json.dump(s, open("prompt_d_stream_api.json", "w"), indent=1)
print("ok")
