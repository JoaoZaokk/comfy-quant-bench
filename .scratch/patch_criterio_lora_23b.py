"""criterio_lora.md: a rodada 2.3 'sem gatilho' medida as 00:50 foi renderizada sobre condicionamento
QUEBRADO (LTXVSaveConditioning perde `unprocessed_ltxav_embeds`; MAE 75,9 contra o encoder vivo).
Marca a tabela como invalida e diz o que a substitui. As previsoes R7-R9 ficam como estao."""
import io

p = 'bench/criterio_lora.md'
s = io.open(p, encoding='utf-8').read()
old = "nos quatro braços (`bench/ltx23/lora/`, par em `bench/ltx23/lora_par/`), MEDIDO 2026-09-14 00:50:**"
assert s.count(old) == 1
new = ("nos quatro braços, MEDIDO 2026-09-14 00:50 — **INVÁLIDO, descoberto às 01:00** (ver a nota logo\n"
       "abaixo da tabela; movido para `bench/ltx23/lora_ltxv_saver/` e `lora_par_ltxv_saver/`):**")
s = s.replace(old, new)
old2 = "- No vídeo, aqui a distância com/sem LoRA (10,4–10,7) fica DENTRO da distância semente-a-semente"
assert s.count(old2) == 1
nota = ("- **NOTA (01:00): estes quatro braços NÃO são o farol.** O controle de identidade (W4A8, 249 quadros,\n"
        "  mesmo condicionamento salvo, contra o render com encoder vivo) deu **MAE 75,9 / SSIM 0,19 /\n"
        "  log-mel 1,05** — quadros de ruído marrom, áudio de ruído (`bench/ltx23/cond_identity_ltxv_saver/\n"
        "  contato_av.png`). Causa, lida no código depois de medir: o encoder do LTX 2.3 devolve\n"
        "  `extra = {\"unprocessed_ltxav_embeds\": True}` (`comfy/text_encoders/lt.py:201-204`) e o modelo só\n"
        "  aplica `caption_projection` + connectors com essa chave (`comfy/model_base.py:1185` →\n"
        "  `av_model.py:583`); o `LTXVSaveConditioning` grava só o tensor e o `LTXVLoadConditioning` devolve\n"
        "  sem a chave — o contexto de 6144 canais passa por já processado e entra cru na cross-attention.\n"
        "  Os números abaixo medem LoRA sobre ruído condicionado errado; ficam como registro do que a\n"
        "  ferramenta errada produz, e são substituídos pela rodada da fila j (condicionamento completo por\n"
        "  `tools/ltx_encode_lowcommit.py` + `VoidLoadConditioningFull`). O par fundido-vs-bypass de 2,40 também.\n")
s = s.replace(old2, nota + old2)
io.open(p, 'w', encoding='utf-8').write(s)
print('criterio_lora: rodada 00:50 marcada como invalida, causa registrada')
