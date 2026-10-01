# Avaliacao em lote, camada 1 (cabecalho, sidecar, analise)

Ordenado por olha-aqui-primeiro. **Nenhum checkpoint aqui esta aprovado**: esta camada
reprova, aponta e preve; ela nao aprova. Ver o cabecalho de `tools/avaliar.py` para o
motivo medido.

| veredito | arquivo | mediana erro | familia | achados |
|---|---|---|---|---|
| OLHAR | `10Eros_v1.5_bf16_w4a8.safetensors` | - | - | `backend_sem_registro` |
| OLHAR | `ltx-2.3-22b-distilled-1.1_w4a4.safetensors` | - | - | `backend_sem_registro` |
| SEM VEREDITO | `10Eros_v1.5_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `LTX23_audio_vae_bf16.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3-22b-distilled-1.1.safetensors` | - | - | - |
| SEM VEREDITO | `ltx-2.3_text_projection_bf16.safetensors` | - | - | - |

## 10Eros_v1.5_bf16_w4a8.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sidecar presente e SEM campo `backend` -- conversao nossa sem registro); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## ltx-2.3-22b-distilled-1.1_w4a4.safetensors -- OLHAR

- **OLHAR** `backend_sem_registro` -- nada registra que o preflight nativo passou (sidecar presente e SEM campo `backend` -- conversao nossa sem registro); o backend eager declara as mesmas capacidades e roda matematica dequantizada

## Nao coberto por esta camada

- sobra: nao le os bytes de sobra nem diz de onde vieram; so que estao la
- escala: nao confere o VALOR das escalas, so a presenca e a forma
- forma: nao pega escala torta em int8_tensorwise (ver FORMAS_DE_ESCALA) nem se os bits int4 decodificam certo
- resumo: nao conta forwards; o formato declarado por camada pode nao ser o executado
- backend: e o registro de uma conversao passada, nao uma execucao agora
- erro: preve o erro de predicao do modelo, nunca a imagem final; so uma renderizacao decide
- armadilha: so conhece as duas que ja custaram trabalho aqui; nao procura armadilhas novas
- tamanho: nao confere se a fonte no disco ainda e a mesma que foi quantizada
- dispatch: nada aqui carrega o modelo, entao ninguem contou forward quantizado nem
  chamada a `dequantize`. Um arquivo pode passar tudo acima e rodar dequantizado.
  Isto deixou de ser um buraco em 2026-09-01: `tools/avaliar_despacho.py` (camada 2, com
  GPU) carrega pelo caminho normal do ComfyUI e conta. Rode-a antes de acreditar no campo
  `backend` de qualquer sidecar -- ele registra a conversao, nao a execucao de hoje.
- imagem: nenhuma renderizacao. Nenhum corte medido nesta bancada separa usavel de
  inutilizavel -- 0,1837 correta contra 0,2147 destruida; 0,7173 boa contra 0,8255
  destruida. So alguem olhando decide.
- braco de referencia: o arquivo NAO quantizado nao e exercitado NESTA camada. Isto
  deixou de ser um buraco em 2026-09-01: `tools/avaliar_referencia.py` (camada 3, com
  GPU) mede se a referencia responde ao proprio condicionamento, e no par de verdade
  conhecida do Wan separou o destruido do bom por 3,19x. Rode-a antes de acreditar em
  qualquer numero desta tabela.
