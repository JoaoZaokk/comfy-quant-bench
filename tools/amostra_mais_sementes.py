r"""Mais sementes sobre conditionings JA calculados, com uma unica carga do modelo de difusao.

POR QUE EXISTE
--------------
`probe_encoder_visual.py` fecha com duas sementes, e duas sementes nao sustentam uma afirmacao
num card publico -- esta bancada acabou de gastar um dia consertando um card cujas imagens
estavam todas fora do ponto de operacao. Mas repetir o probe inteiro recarregaria os tres
encoders e o modelo de difusao TRES vezes para amostrar alguns segundos.

O conditioning de cada braco ja esta em disco (`cond_<prompt>_<braco>.pt`). Ele nao depende da
semente. Entao mais sementes custam UMA carga do difusor e nada mais -- os encoders nem entram.

O CONTROLE
----------
Os conditionings carregados tem de diferir entre si. Se dois arquivos fossem iguais, os bracos
correspondentes gerariam a mesma imagem e a folha compararia um braco consigo mesmo. E impresso
antes de qualquer amostragem, porque um controle depois do trabalho ja nao evita o trabalho.

NAO COBERTO
-----------
Isto acrescenta sementes, nao prompts nem modelos. A comparacao continua LIVRE: cada braco leva o
amostrador por uma trajetoria propria, entao o que se le na folha e "isso ainda e uma boa imagem
do que foi pedido?", nunca "isso esta perto do bf16".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARGV_REAL = sys.argv[1:]          # guardado ANTES da linha abaixo, que apaga os argumentos
sys.path.insert(0, str(RAIZ / "ComfyUI"))
sys.argv = ["main.py"]

import comfy.options

comfy.options.enable_args_parsing()

import comfy.sample
import comfy.sd
import folder_paths
import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=RAIZ / "bench" / "encoder_visual")
    p.add_argument("--unet", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 99, 2026])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    a = p.parse_args(ARGV_REAL)

    conds = {}
    for f in sorted(a.dir.glob("cond_*.pt")):
        conds[f.stem[len("cond_"):]] = torch.load(f, map_location="cpu")
    if not conds:
        print(f"nenhum cond_*.pt em {a.dir}", file=sys.stderr)
        return 2

    nomes = sorted(conds)
    print(f"{len(nomes)} conditionings: {nomes}")
    print()
    print("CONTROLE -- os conditionings tem de diferir dois a dois:")
    iguais = []
    for i, x in enumerate(nomes):
        for y in nomes[i + 1:]:
            u, v = conds[x].float(), conds[y].float()
            if u.shape != v.shape:
                continue
            rel = ((v - u).norm() / u.norm()).item()
            if rel <= 1e-6:
                iguais.append((x, y))
                print(f"  {x} == {y}   IDENTICOS")
    if iguais:
        print("CONTROLE FALHOU: ha bracos identicos; amostrar mais sementes nao mede nada.")
        return 3
    print("  nenhum par identico. OK")

    model = comfy.sd.load_diffusion_model(
        folder_paths.get_full_path_or_raise("diffusion_models", a.unet))
    lf = model.model.latent_format
    side = a.size // 8
    print()
    for nome in nomes:
        c = conds[nome].cuda()
        positivo = [[c, {}]]
        negativo = [[torch.zeros_like(c[:, :1]), {}]]   # cfg 1,0 nao consulta o negativo
        for semente in a.seeds:
            destino = a.dir / f"latent_{nome}_s{semente}.pt"
            if destino.exists():
                print(f"  {destino.name} ja existe, pulando")
                continue
            latent = torch.zeros([1, lf.latent_channels, side, side], device="cpu")
            ruido = comfy.sample.prepare_noise(latent, semente, None)
            out = comfy.sample.sample(model, ruido, a.steps, a.cfg, "euler", "simple",
                                      positivo, negativo, latent, denoise=1.0,
                                      disable_noise=False, start_step=None, last_step=None,
                                      force_full_denoise=False, noise_mask=None, callback=None,
                                      disable_pbar=True, seed=semente)
            out = out.cpu().float()
            torch.save(out, destino)
            print(f"  {destino.name}  std {out.std():.4f}")
    print()
    print("NAO COBERTO: mais sementes, mesmos prompts e mesmo modelo. A comparacao continua")
    print("  LIVRE -- cada braco tem trajetoria propria, entao a folha responde 'ainda e uma boa")
    print("  imagem do que foi pedido?', nao 'esta perto do bf16'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
