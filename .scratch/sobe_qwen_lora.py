from huggingface_hub import HfApi
import urllib.request
api = HfApi(); repo = "JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot"
api.upload_file(path_or_fileobj="bench/hf/qwen-image-edit-2511-quant/images/lora_lightning_4steps.png", path_in_repo="images/lora_lightning_4steps.png", repo_id=repo, repo_type="model", commit_message="Evidence: Lightning 4-step LoRA, five arms with the no-LoRA control")
api.upload_file(path_or_fileobj="bench/hf/qwen-image-edit-2511-quant/README.md", path_in_repo="README.md", repo_id=repo, repo_type="model", commit_message="Card: does a LoRA work on this file -- buried at the weight, intact at the output, with the control that has to fail")
txt = urllib.request.urlopen(f"https://huggingface.co/{repo}/raw/main/README.md", timeout=30).read().decode()
files = set(api.list_repo_files(repo, repo_type="model"))
print("hub README has LoRA section:", "Does a LoRA work on this file" in txt, "| grid on hub:", "images/lora_lightning_4steps.png" in files)
