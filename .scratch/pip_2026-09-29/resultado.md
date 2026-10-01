# Upgrade pip do python_embeded (2026-09-29): resultado

Critério em `criterio.md`, escrito antes. Aplicado: 155 pacotes (lista `sobe.txt`, restrições `constraints.txt`).
Ficaram de fora torch/torchvision/triton/comfy-kitchen/frontend/templates/setuptools.
Principais: transformers 4.57.6 → 5.17.0, huggingface_hub 0.36.2 → 1.33.0, tokenizers 0.22 → 0.23.2,
numpy 2.4.6 → 2.5.3, opencv 4.13 → 5.0, diffusers 0.38 → 0.40, peft 0.19 → 0.21, accelerate 1.15, mediapipe 1.0.1,
insightface 2.0, librosa 1.0, SQLAlchemy 2.1, comfyui-manager 4.3.
Voltaram ao pino antigo: albucore 0.0.24 (albumentations 2.0.8) e supervision 0.6.0 (groundingdino-py).
fsspec ficou em 2026.6.0 (datasets).

Backup integral: `python_embeded_backup_20260929` (119.494 arquivos, 9,19 GB, 0 falhas no robocopy).
Rollback = trocar as pastas. Freezes: `freeze_antes.txt` / `freeze_depois.txt`.

## Medido
1. Importação (`--quick-test-for-ci`, 3090):
   - Antes: 98 pacotes de nós, 2 IMPORT FAILED, ambos preexistentes:
     - tinyterranodes: `config.ini` com uma linha `.1` solta;
     - VoxCPM: a guarda `"pytest" in sys.modules` dispara porque outro nó importa pytest.
   - Depois: 98 pacotes, 0 falhas (as duas corrigidas com patch local, originais `.orig_20260929`/`.quebrado_20260929`).
   - Diferença nas linhas de erro/aviso: só avisos benignos (schema do ultralytics, comfyregistry offline).
2. Tokenizadores do core (CPU, 5 prompts): 66/66 que funcionavam antes deram ids idênticos no transformers 5.
   Os outros 31 precisam do tokenizador de dentro do checkpoint e já falhavam igual antes.
3. Render pareado, mesmos grafos/seeds/flags da validação triton (on): 16/16 success e 16/16 idênticos pixel a pixel.
   Grafos: K1, K2, Z1, Z2, Q1, Q3, Q4 e Q5, cada um frio e quente.
   - Quente: tempo dentro de ±3% (ex.: Q1 25,6 → 26,0 s; Q5 11,3 → 11,5 s).
   - Frio dos Qwen: muito mais lento (Q1 146 → 454 s, Q5 81 → 632 s). O frio inclui a leitura do DiT pelo NAS e o quente
     não mudou. Hipótese não medida: rede/NAS lento na hora. Não atribuo ao upgrade sem repetir.
4. `pip check`: "No broken requirements found" (depois do ajuste de metadados do Nuvu abaixo).
5. Varredura estática: 120 símbolos `from transformers… import` dos nós testados no transformers 5.
   - Só `AutoModelForVision2Seq` sumiu; DaSiWa-Nodes e QwenVL-Mod já têm fallback para AutoModelForImageTextToText.
   - numpy 2.5 removeu `row_stack` (os outros quatro já não existiam no 2.4.6); nenhum nó usa.
   - mediapipe `solutions` já não existia no 0.10.35 (não é regressão).

## Achado: o Nuvu era o que prendia o transformers antigo
`comfyui-nuvu` rebaixava o ambiente de três formas:
- o `prestartup_script.py` força `huggingface_hub<1.0` com `--force-reinstall` a cada subida;
- o `pre_launch.py` exige `transformers==4.57.6`;
- o `requirements.txt` reinstala `comfyui-nuvu==1.0.90`, cujo METADATA pede `huggingface_hub<1.0`, e o resolvedor
  rebaixa hub e transformers juntos.

Isso aconteceu duas vezes durante o teste; na primeira deixou os arquivos do hub misturados e o ComfyUI não subiu.
Patch local nos três pontos mais o METADATA (original em `nuvu_METADATA.orig`). Um dry-run do requirements do Nuvu
agora não instala nada. Ao atualizar o Nuvu, refazer.

A mensagem do Manager "[SKIP] Downgrading pip package isn't allowed" não indica versão presa. Ela sai quando o
instalado é mais novo que o `>=` pedido (manager_core.py, `is_installed`), e continua igual no 4.3.

## Não medido
Nós que carregam modelos HF em runtime (QwenVL, DaSiWa LLM, RMBG, SUPIR, chatterbox, VoxCPM, WanVideoWrapper,
LTX2_SM) não foram exercitados com modelo real. Nenhum workflow salvo em `user/default/workflows` usa os nós de VLM.
LTX (vídeo+áudio) e Qwen Edit também não entraram no render.
