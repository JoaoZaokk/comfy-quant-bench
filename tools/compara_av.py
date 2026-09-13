"""Compara bracos de VIDEO E AUDIO contra um braco de referencia, quadro a quadro e amostra a amostra.

POR QUE EXISTE
--------------
Ate 2026-09-13 esta bancada comparava LTX so pelos quadros. O modelo gera video E audio no mesmo
latente; medir metade e publicar como se fosse o todo foi apontado pelo dono como erro. Este
arquivo recebe os JSONs que `ltx_video.py` grava (lista de PNGs + FLAC por corrida) e mede os
dois ramos contra a referencia, com controles sinteticos para os numeros de audio terem escala.

O QUE MEDE, e o que cada numero NAO responde
--------------------------------------------
VIDEO (sobre os PNGs sem perda, nunca sobre o MP4, que e h264 com perda):
  MAE     erro medio por pixel em 0-255, media sobre TODOS os quadros, com min-max entre quadros.
  PSNR    o mesmo em dB. Nenhum dos dois e perceptual: um deslocamento de um pixel derruba os dois.
  SSIM    estrutura local, media sobre quadros. Menos cego que PSNR, ainda nao e julgamento humano.
  movimento  media de |quadro_t - quadro_{t-1}|, por braco. Nao compara com a referencia: diz se
          o braco tem movimento parecido, e denuncia video congelado ou tremido.

AUDIO (sobre o FLAC sem perda; os dois ramos saem do mesmo VAE de audio, na mesma taxa):
  MAE_onda   erro medio na forma de onda (amplitude em -1..1). Muito sensivel a fase: dois audios
             IGUAIS deslocados de 1 ms dao MAE alto. Por isso o lag entra ao lado.
  SNR        10*log10(energia_ref / energia_erro), dB. Mesma sensibilidade a fase.
  lag        deslocamento (ms) do pico de correlacao cruzada, limitado a +-200 ms. 0 = alinhado.
  log-mel L1 distancia media entre espectrogramas log-mel (64 bandas). Insensivel a fase fina;
             e o numero que mais se aproxima de "soa parecido".
  conv_esp   convergencia espectral, |S_ref - S| / |S_ref| sobre magnitude STFT.
  RMS        nivel do braco em dBFS, e fracao de janelas de 20 ms abaixo de -50 dBFS ("silencio").
             Nao compara: diz se o braco produziu som ou ficou mudo.

CONTROLES, para os numeros de audio terem escala:
  "controle: silencio"      a referencia contra zeros
  "controle: ruido branco"  a referencia contra ruido gaussiano de mesmo RMS
  Um braco que mede como o ruido branco nao "soa parecido", por menor que o numero pareca.

NAO COBERTO
-----------
Nenhuma metrica aqui foi validada contra julgamento humano nesta bancada -- sao padrao da
literatura, usadas como padrao da literatura. Tudo e distancia da referencia: "quao longe", nunca
"bom". Um braco pode estar longe por ser pior, ou por ter ido para outro lugar tambem bom -- a
trajetoria livre do sampler ja foi medida nesta bancada como caotica a 8 passos, e o audio nao
tem motivo para ser diferente. Nao mede sincronia audio-video (se o som bate com a imagem), so
cada ramo contra o seu par na referencia.

    python_embeded\\python.exe -s tools\\compara_av.py --ref REF.json --arm "rotulo=RUN.json" ... --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "ComfyUI" / "output"


def carrega_run(caminho: Path) -> dict:
    r = json.loads(Path(caminho).read_text(encoding="utf-8"))
    r["_quadros"] = [OUT_DIR / q for q in r.get("quadros", [])]
    r["_audio"] = [OUT_DIR / q for q in r.get("audio", [])]
    faltam = [str(p) for p in r["_quadros"] + r["_audio"] if not p.exists()]
    if faltam:
        raise SystemExit(f"{caminho}: {len(faltam)} arquivo(s) nao encontrado(s), ex {faltam[0]}")
    if not r["_quadros"]:
        raise SystemExit(f"{caminho}: sem quadros")
    if not r["_audio"]:
        raise SystemExit(f"{caminho}: sem audio -- este comparador exige os dois ramos")
    return r


def le_quadros(paths: list[Path]) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths])


def le_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.T.copy(), int(sr)  # (canais, amostras)


# ---------------------------------------------------------------- video
def metricas_video(ref: np.ndarray, arm: np.ndarray) -> dict:
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    n = min(len(ref), len(arm))
    if ref.shape[1:] != arm.shape[1:]:
        raise SystemExit(f"quadros de tamanho diferente: {ref.shape[1:]} vs {arm.shape[1:]}")
    r = ref[:n].astype(np.float32)
    a = arm[:n].astype(np.float32)
    dif = np.abs(r - a).reshape(n, -1)
    mae = dif.mean(axis=1)
    mse = (((r - a) ** 2).reshape(n, -1)).mean(axis=1)
    psnr = np.where(mse > 0, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12)), 99.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=255.0)
    vals = []
    with torch.no_grad():
        for i in range(0, n, 8):
            tr = torch.from_numpy(r[i:i + 8]).permute(0, 3, 1, 2)
            ta = torch.from_numpy(a[i:i + 8]).permute(0, 3, 1, 2)
            vals.append(float(ssim(ta, tr)))
            ssim.reset()
    return {"quadros_comparados": int(n), "quadros_ref": int(len(ref)), "quadros_braco": int(len(arm)),
            "mae": float(mae.mean()), "mae_min": float(mae.min()), "mae_max": float(mae.max()),
            "psnr": float(psnr.mean()), "psnr_min": float(psnr.min()),
            "ssim": float(np.mean(vals))}


def movimento(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    f = frames.astype(np.float32)
    return float(np.abs(f[1:] - f[:-1]).mean())


# ---------------------------------------------------------------- audio
def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return 20 * math.log10(rms)


def fracao_silencio(x: np.ndarray, sr: int, limiar_db: float = -50.0) -> float:
    mono = x.mean(axis=0)
    jan = max(int(sr * 0.02), 1)
    n = len(mono) // jan
    if n == 0:
        return 1.0
    blocos = mono[:n * jan].reshape(n, jan)
    rms = np.sqrt((blocos ** 2).mean(axis=1)) + 1e-12
    return float((20 * np.log10(rms) < limiar_db).mean())


def lag_ms(ref: np.ndarray, arm: np.ndarray, sr: int, max_ms: float = 200.0) -> float:
    r = ref.mean(axis=0); a = arm.mean(axis=0)
    r = r - r.mean(); a = a - a.mean()
    n = len(r) + len(a) - 1
    nf = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(r, nf) * np.conj(np.fft.rfft(a, nf)), nf)
    corr = np.concatenate([corr[-(len(a) - 1):], corr[:len(r)]])  # lags -(len(a)-1) .. len(r)-1
    lags = np.arange(-(len(a) - 1), len(r))
    maxl = int(sr * max_ms / 1000)
    m = np.abs(lags) <= maxl
    k = int(np.argmax(corr[m]))
    return float(lags[m][k] * 1000.0 / sr)


def logmel(x: np.ndarray, sr: int):
    import torch
    import torchaudio
    mono = torch.from_numpy(x.mean(axis=0)).float()
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=64)(mono)
    return torch.log10(mel + 1e-8).numpy()


def stft_mag(x: np.ndarray):
    import torch
    mono = torch.from_numpy(x.mean(axis=0)).float()
    s = torch.stft(mono, n_fft=2048, hop_length=512, window=torch.hann_window(2048), return_complex=True)
    return s.abs().numpy()


def metricas_audio(ref: np.ndarray, arm: np.ndarray, sr: int) -> dict:
    n = min(ref.shape[1], arm.shape[1])
    c = min(ref.shape[0], arm.shape[0])
    r = ref[:c, :n]; a = arm[:c, :n]
    err = r - a
    e_ref = float((r ** 2).sum()); e_err = float((err ** 2).sum())
    lm_r, lm_a = logmel(r, sr), logmel(a, sr)
    t = min(lm_r.shape[1], lm_a.shape[1])
    sm_r, sm_a = stft_mag(r), stft_mag(a)
    ts = min(sm_r.shape[1], sm_a.shape[1])
    return {"amostras_comparadas": int(n), "amostras_ref": int(ref.shape[1]), "amostras_braco": int(arm.shape[1]),
            "sr": sr, "mae_onda": float(np.abs(err).mean()),
            "snr_db": 10 * math.log10(e_ref / e_err) if e_err > 0 else 99.0,
            "lag_ms": lag_ms(r, a, sr),
            "logmel_l1": float(np.abs(lm_r[:, :t] - lm_a[:, :t]).mean()),
            "conv_espectral": float(np.linalg.norm(sm_r[:, :ts] - sm_a[:, :ts]) / (np.linalg.norm(sm_r[:, :ts]) + 1e-12)),
            "rms_dbfs": dbfs(a), "silencio_frac": fracao_silencio(a, sr)}


# ---------------------------------------------------------------- folha de contato
def folha(ref_nome: str, ref_fr: np.ndarray, ref_au: np.ndarray, sr: int,
          bracos: list[tuple[str, np.ndarray, np.ndarray, dict]], out: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    linhas = [(ref_nome, ref_fr, ref_au, None)] + bracos
    n = len(ref_fr)
    idx = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    ncol = len(idx) + 2
    fig, axes = plt.subplots(len(linhas), ncol, figsize=(2.3 * ncol, 2.4 * len(linhas)))
    axes = np.atleast_2d(axes)
    # Escala de cor do log-mel fixada nos percentis da REFERENCIA e compartilhada por todas as
    # linhas: com limites fixos o primeiro selftest saturou 80% da imagem em amarelo, e uma
    # escala por linha esconderia justamente a diferenca de nivel entre bracos.
    lm_ref = logmel(ref_au, sr)
    vmin, vmax = np.percentile(lm_ref, [2, 99.5])
    for li, (nome, fr, au, m) in enumerate(linhas):
        for ci, i in enumerate(idx):
            ax = axes[li, ci]
            ax.imshow(fr[min(i, len(fr) - 1)]); ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(f"quadro {i + 1}", fontsize=8)
        ax = axes[li, len(idx)]
        lm = logmel(au, sr)
        ax.imshow(lm, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if li == 0:
            ax.set_title("log-mel (64 bandas)", fontsize=8)
        ax = axes[li, len(idx) + 1]
        mono = au.mean(axis=0)
        tt = np.arange(len(mono)) / sr
        ax.plot(tt, mono, lw=0.3, color="k"); ax.set_ylim(-1, 1); ax.set_yticks([])
        ax.set_xlim(0, tt[-1] if len(tt) else 1); ax.tick_params(labelsize=6)
        if li == 0:
            ax.set_title("forma de onda (mono)", fontsize=8)
        rot = nome
        if m:
            rot += (f"\nMAE {m['video']['mae']:.2f}  PSNR {m['video']['psnr']:.1f} dB  SSIM {m['video']['ssim']:.3f}"
                    f"\naudio: log-mel L1 {m['audio']['logmel_l1']:.3f}  SNR {m['audio']['snr_db']:.1f} dB")
        else:
            rot += f"\n(referencia)\nRMS {dbfs(au):.1f} dBFS"
        axes[li, 0].set_ylabel(rot, fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, type=Path, help="JSON de ltx_video.py do braco de referencia")
    p.add_argument("--arm", action="append", required=True, metavar="ROTULO=RUN.json")
    p.add_argument("--out", required=True, type=Path, help="pasta de saida (json + folha de contato)")
    p.add_argument("--titulo", default="")
    a = p.parse_args()

    ref = carrega_run(a.ref)
    ref_fr = le_quadros(ref["_quadros"])
    ref_au, sr = le_audio(ref["_audio"][0])
    print(f"referencia: {ref.get('modelo')}  {len(ref_fr)} quadros {ref_fr.shape[1]}x{ref_fr.shape[2]}, "
          f"audio {ref_au.shape[0]} canais {ref_au.shape[1] / sr:.2f} s a {sr} Hz, RMS {dbfs(ref_au):.1f} dBFS, "
          f"silencio {fracao_silencio(ref_au, sr):.0%}", flush=True)

    rel: dict = {"referencia": {"json": str(a.ref), "modelo": ref.get("modelo"), "segundos": ref.get("segundos"),
                                "quadros": int(len(ref_fr)), "sr": sr, "audio_s": ref_au.shape[1] / sr,
                                "rms_dbfs": dbfs(ref_au), "silencio_frac": fracao_silencio(ref_au, sr),
                                "movimento": movimento(ref_fr)},
                 "controles": {}, "bracos": {}}
    rng = np.random.default_rng(0)
    rel["controles"]["silencio"] = metricas_audio(ref_au, np.zeros_like(ref_au), sr)
    ruido = rng.standard_normal(ref_au.shape).astype(np.float32) * float(np.sqrt(np.mean(ref_au ** 2)))
    rel["controles"]["ruido_branco_mesmo_rms"] = metricas_audio(ref_au, ruido, sr)

    linhas = []
    for spec in a.arm:
        rot, _, js = spec.partition("=")
        run = carrega_run(Path(js))
        fr = le_quadros(run["_quadros"])
        au, sr2 = le_audio(run["_audio"][0])
        if sr2 != sr:
            raise SystemExit(f"{rot}: taxa de amostragem {sr2} != {sr} da referencia; nao comparo sem reamostrar")
        mv = metricas_video(ref_fr, fr)
        ma = metricas_audio(ref_au, au, sr)
        m = {"json": js, "modelo": run.get("modelo"), "lora": run.get("lora"), "segundos": run.get("segundos"),
             "s_por_quadro": run.get("s_por_quadro"), "video": mv, "audio": ma, "movimento": movimento(fr)}
        rel["bracos"][rot] = m
        linhas.append((rot, fr, au, m))
        print(f"{rot:<28} MAE {mv['mae']:6.2f} [{mv['mae_min']:.2f}-{mv['mae_max']:.2f}]  PSNR {mv['psnr']:5.2f}  "
              f"SSIM {mv['ssim']:.4f}  mov {m['movimento']:.2f} | audio log-mel {ma['logmel_l1']:.3f}  "
              f"SNR {ma['snr_db']:6.2f} dB  lag {ma['lag_ms']:+.1f} ms  conv {ma['conv_espectral']:.3f}  "
              f"RMS {ma['rms_dbfs']:.1f} dBFS  silencio {ma['silencio_frac']:.0%}", flush=True)
    c = rel["controles"]
    print(f"{'controle: silencio':<28} audio log-mel {c['silencio']['logmel_l1']:.3f}  SNR {c['silencio']['snr_db']:6.2f} dB", flush=True)
    print(f"{'controle: ruido branco':<28} audio log-mel {c['ruido_branco_mesmo_rms']['logmel_l1']:.3f}  "
          f"SNR {c['ruido_branco_mesmo_rms']['snr_db']:6.2f} dB  conv {c['ruido_branco_mesmo_rms']['conv_espectral']:.3f}", flush=True)
    print(f"{'referencia (movimento)':<28} mov {rel['referencia']['movimento']:.2f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "comparacao_av.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    folha(f"{ref.get('modelo')}", ref_fr, ref_au, sr, linhas, a.out / "contato_av.png",
          a.titulo or f"{len(ref_fr)} quadros + audio, referencia {ref.get('modelo')}")
    print(f"\njson: {a.out / 'comparacao_av.json'}\nfolha: {a.out / 'contato_av.png'}", flush=True)
    print("\nNAO COBERTO: tudo aqui e distancia da referencia, nao qualidade; nenhuma metrica foi validada "
          "contra julgamento humano nesta bancada; sincronia audio-video nao e medida; MAE_onda e SNR "
          "sao sensiveis a fase (ver lag); MP4 nao entra em metrica nenhuma.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
