# Resultado — refatoração QAT klein + nós (2026-09-29)

Critério escrito antes: `criterio_qat.md` (mesma pasta). Tudo medido nesta sessão em CPU com
`CUDA_VISIBLE_DEVICES=-1` (conferido `torch.cuda.is_available() == False` no início de cada rodada).
Nada de GPU, treino real, Colab, HF ou instalação.

## Veredito por item do critério

| item | resultado | evidência |
|---|---|---|
| A1 CLI | APROVADO | `depois_compara_cli.log`: 35 flags antigas presentes com mesmos defaults/choices/nargs; 3 novas opcionais (`--exporta`, `--retoma-config-diferente`, `--para-no-passo`); `--device` aceita também `cpu` (string; inteiros como antes) |
| A2 passo de treino | APROVADO | `test_passo_de_treino_e_codigos_ternarios`: perda finita, exportado com ≤1 módulo não nulo por grupo |
| A3 retomada determinística | APROVADO | `test_retomada_reproduz_a_trajetoria`: 6 passos contínuos ≡ parada no passo 3 (caminho SIGTERM via `--para-no-passo`) + retomada; perdas por passo iguais, mestre bit a bit, tensores e metadados dos exportados (desempacotado e lowbit) bit a bit, melhor.json igual. Bytes do arquivo NÃO: o safetensors grava `__metadata__` em ordem de HashMap (varia por execução — vale também para o código antigo) |
| A4 checkpoint antigo | APROVADO | `test_checkpoint_antigo_retoma`: ckpt só com as chaves antigas retoma, avisa, e o novo ckpt sai com `estado` |
| A5 shards | APROVADO | legado `p####_s#.pt` sem `size` treina (`test_shard_legado_treina_e_valida`); `valida_shard` recusa prompt/passos; legado com índice trocado não pula o prompt certo (`test_legado_com_indice_trocado...`); vazamento pelos prompts dos shards recusado |
| A6 config / globais | APROVADO | `--so-escalas --l1-corpo` e `--cruzado-frac 1.5` recusados sem carregar modelo; retomada com `--lr` diferente → RECUSADO (status RECUSADO), aceita com flag; nenhum `NIVEIS=`/`STE_LIGADO=`/`global` (só o cache `_KERNEL`) |
| A7 lowbit | APROVADO | ternário (G 128 e 32) e int4 (G 32) desquantizam bit a bit no desempacotado bf16 (`test_lowbit_desquantiza_bit_a_bit`, e no exportado real do treino); contrato de chaves idêntico ao `formats.to_comfy_state_dict` do loader (import em CPU). Default continua `desempacotado` — ver limites |
| A8 HF | APROVADO | API falsa: erro de rede em `baixa_se_faltar` → aborta ("recusando comecar do zero"); EntryNotFound → do zero; `envia` passo 5 com remoto 10 → nenhum commit; passo 12 → um commit com `ckpt/ultimo.pt` + `ckpt/passo.json` |
| A9 status/probe/parada | APROVADO | `status.json` FIM/PARADO(143)/RECUSADO nos testes de treino; probe único decide por status e cai no log sem ele (10 casos); probes gerados = molde; `limpa_braco_sem_ckpt` apaga só journal/melhor(/status) e mantém shards e ref_sens |
| A10 dataset | APROVADO | metadados por VM: lê legado + outras VMs, reescreve só o próprio; erro de rede recusa, ausência vira vazio; sobra local legada com estatísticas entra no lote renomeada pela chave; sem estatísticas é MOVIDA para `_sem_estat/` (não apagada) |
| A11 lowbit_canon | APROVADO | `Quantizador` ≡ código antigo do QAT bit a bit (niveis 1/7, bf16/fp32); `ternariza` ≡ `constroi` antigo bit a bit; `rtn_simetrico` ≡ `mistura_klein.rtn4` (2/3/4 bits); `constroi_ternario_ingenuo` HEAD × novo num safetensors sintético: **idênticos byte a byte** (`depois_constroi_antigo_novo.log`) |
| A12 nós | APROVADO | void: sem NVML só a placa já inicializada recebe `mem_get_info`; com NVML nenhuma; `audit_dir`/`VOID_AUDIT_DIR`/`off`; 3 classes antigas = uma implementação. qwen21: `add_patches` com LoRA → RuntimeError; `clone()` mantém a classe. stream: 2ª iteração → RuntimeError, quadro 0 não decodifica 2x |
| A13 qwen21 QAT | APROVADO | smoke CPU antes e depois idênticos (RTN 0,0466 → QAT 0,0448; to_q rel 3,18e-3); agora as 12 camadas passam por assert (códigos simulados = ck, escalas 0, rel ≤ 3,6e-3 < 2e-2) |

Equivalência extra (uma vez, código antigo guardado × novo, `depois_equivalencia_antigo_novo.log`): forward,
16 gradientes e 16 tensores exportados bit a bit iguais nos modos ste-ternário, ste-int4 e so-escalas.

## Testes antes/depois

- Antes (baseline desta sessão): `qat_qwen21_blocos --smoke` rc 0 (`antes_qwen21_smoke.log`); `test_tiles` 5
  passed; QAT sem nenhum teste.
- Depois: `pytest tools/qat_klein tools/colab_qat test_tiles` **51 passed**; `testes/test_void.py` 5 passed;
  `testes/test_qwen21_patcher.py` 2 passed; smoke qwen21 rc 0 com asserts (`depois_qwen21_smoke.log`);
  `test_lowbit.py` do loader (arquivo de outro agente, só executado) passa em CPU com o teste de GPU pulado.
- ruff (só checagem; `--fix` apenas nos arquivos criados nesta tarefa e nas linhas que eu escrevi): arquivos
  novos limpos; arquivos pré-existentes sem item novo além dos que já tinham.

## Limites e o que NÃO foi feito

1. **Loader não carrega lowbit em nomes diffusers do FLUX.2** (medido, `prova_fusao.py` no scratchpad): o
   `diffusers_flux2_to_bfl` agrupa q/k/v só pela letra; num arquivo `lowbit_affine` os `weight_scale`/
   `weight_zeros`/`comfy_quant` de to_q/k/v se sobrescrevem e sobra só `...qkv.weight`. Não editei o loader
   (posse de outro agente). Por isso `--exporta` fica `desempacotado` por padrão; `lowbit`/`ambos` existem e
   são bit a bit, mas o arquivo só carrega no ComfyUI depois de o loader agrupar por sufixo.
2. Sinal do zero: o QAT sempre gravou +0.0 e o `constroi` −0.0 (round de negativo). O canônico preserva as
   duas receitas (`quantiza(zero_com_sinal=...)`). O lowbit não representa −0.0; `confere_exato` conta esses
   casos à parte (valor idêntico) — nos testes o QAT não produziu nenhum.
3. `cod_ant` (códigos do último log, ~bytes = nº de params) não vai no checkpoint: após retomar, o primeiro
   `codigos_trocados_desde_ultimo` é 0 (como já era no passo 0). Não afeta a trajetória.
4. Checkpoint antigo (sem estado): comportamento antigo de propósito (permutação re-sorteada, `melhor` do
   melhor.json). A trajetória dele NÃO é a de uma corrida sem interrupção — só a dos novos é.
5. Viewer do HF: o imagefolder só associa `metadata.jsonl` da raiz; amostras novas (em `metadata/<vm>.jsonl`)
   ficam no repo mas sem colunas na prévia. O QAT e o gerador leem os dois.
6. `celula_prepara.py` (klein) e as células do qwen21 continuam autocontidas (rodam antes do upload / outra
   pasta de VM); não usam `colab_ops`.
7. Arquivos a subir para VMs NOVAS: resolvido depois, com autorização do coordenador -- ver "Scripts de
   subida para o Colab" abaixo (manifesto gerado de `colab_ops.ARQUIVOS_QAT`, kernel do loader incluído como
   `lowbit_kernel.py`; as células de lançamento recusam com `FALTA ...` se faltar algo).
8. `transplanta_klein.py`, `mistura_klein.py`, `ajusta_denso_diffusers.py`: não tocados (posse do coordenador).
   Prontos para trocar: `lowbit_canon.pilhas_reais/BLOCO/corpo`, `rtn_simetrico` (= `rtn4`), `ternariza`
   (= receita do constroi, com −0.0 — o C0 do transplanta depende disso).

## O que exige GPU/Colab para validar

- Retomada real de um `ckpt/ultimo.pt` remoto existente (torchao AdamW8bit/4bit na VM, `opt.load_state_dict`
  na GPU) — testado aqui só com `adam-fp32` em CPU e com estrutura de nomes/ordem idêntica à antiga.
- Custo do índice de shards legados na primeira execução (abre cada `p####_s#.pt` com mmap; cacheado depois em
  `indice_shards.json`) — [ESTIMATIVA] 1-3 min para 2×1.756 shards, não medido.
- SIGTERM real na VM (Linux) com `para()` esperando o checkpoint e o envio antes do SIGKILL; o caminho foi
  testado via `--para-no-passo` (o Windows não entrega SIGTERM a handler).
- Probe novo com o supervisor real (`colab_job_supervisor.py`): saída tem as mesmas chaves antigas + `job_status`.
- Envio real com `create_commit` (ckpt + passo.json no mesmo commit) e `super_squash_history`.
- Qualidade (render/epsilon) — inalterada por construção (forward/gradiente/exportado bit a bit), mas não medida.

## Scripts de subida para o Colab (pedido posterior do coordenador)

Fonte única: `colab_ops.ARQUIVOS_QAT` (agora com `lowbit_kernel.py`) + `ORIGEM` → `colab_ops.manifesto()`;
`colab_ops.py monta` gera probes e `tools/colab_qat/arquivos_qat.txt` (origem:destino, 17 linhas). Novo
`.scratch/sobe_qat_comum.sh` (função `sobe_qat SESSAO`): confere que cada origem existe ANTES de qualquer
upload, executa `celula_pastas.py` (cria `/content/qat/qat_klein`) e sobe o manifesto (`< /dev/null` em cada
comando para o `colab` não consumir o manifesto). `L=` para quando o shell local vê a bancada noutro caminho.

Atualizados (backup dos originais, que não são rastreados pelo git, no scratchpad `scratch_sh_antes/`):
`lanca_b11L.sh`, `lanca_ds_a100.sh`, `lanca_fila_a100b.sh`, `lanca_fila_a100c.sh`, `relanca_a100b.sh`,
`relanca_ds_l4.sh`, `sobe_a100b.sh`, `sobe_ds_l4.sh`, `tenta_a100.sh` -- cada um chama `sobe_qat` e mantém
no próprio laço só o que é do job (fila/config/prompts/células/wheel/token). Junto:
- `relanca_a100b.sh`/`relanca_ds_l4.sh`: timeouts do `exec` das células de parada aumentados (a parada agora
  espera o checkpoint: 900 s / 120 s) e o probe correspondente passa a subir junto;
- `lanca_fila_a100b.sh` sobe também o `probe_qat.py` (gerado);
- `tenta_a100.sh` deixou de REGRAVAR `tools/colab_qat/celula_pastas.py` com o conteúdo antigo (desfaria a
  criação de `qat_klein/`);
- `.scratch/celula_confere_ds.py` (checagem) passou a procurar o código no novo lugar.
Não precisaram: `diag_hf_vm.sh` (só exec) e `.scratch/qat_qwen21_2026-09-27/*.sh` (sobem só os arquivos do
qwen21 para `/content/qatq`, sem dependência nova).

Teste seco (`depois_sobe_qat_seco.log` + 3 testes novos em `test_colab_qat.py`, 20 passed nesse arquivo):
todos os 10 `.sh` passam em `bash -n` (bash do Git; o teste recusa o `bash.exe` do System32, que abriria o
WSL); manifesto em dia com `ARQUIVOS_QAT` e as 17 origens existem; com `colab` falso (`echo`) o `sobe_qat`
faz exatamente os 17 uploads esperados + 1 exec, e com raiz errada recusa sem nenhum upload; nenhum script
sobe mais `qat_ternario_klein.py`/`ajusta_denso_diffusers.py` avulso. Suíte completa: 53 passed. Nada foi
executado contra Colab/HF/WSL.

## Incidentes desta sessão (registrar)

- Na revisão anterior (somente leitura) rodei `test_lowbit.py` com `CUDA_VISIBLE_DEVICES=""`; no Git Bash do
  Windows isso não esconde a GPU e o teste de Triton rodou em cuda:0 por segundos, sem lock. Desde então todo
  Python rodou com `-1` e conferência de `is_available() == False`.
- Um comando `python - <<'EOF'` vazio chamou o Python GLOBAL (`C:/Program Files/Python312`) com script vazio
  (não executou nada; a edição pretendida foi refeita com o Edit). Nenhuma outra chamada fora do `python_embeded`.
- Normalizei CRLF→LF em arquivos da minha posse que eu tinha escrito com `write_text` no Windows; isso incluiu
  `tools/colab_qat_qwen21/celula_prepara.py`, que já estava CRLF em disco (mudança só de fim de linha; com
  `core.autocrlf=true` o git não mostra diferença de conteúdo).
