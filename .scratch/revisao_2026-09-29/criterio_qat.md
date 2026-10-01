# Critério — refatoração QAT klein + nós (revisão 2026-09-29)

Escrito ANTES de implementar. Escopo autorizado pelo dono via coordenador ("faça tudo"): achados 1-5, 7-10
da revisão e movimentos estruturais do QAT. Proibido: GPU (todo Python com `CUDA_VISIBLE_DEVICES=-1`,
conferido por `torch.cuda.is_available() == False`), treino real, Colab, escrita no HF, instalar pacotes,
editar comfy-lowbit-loader / transplanta / mistura / ajusta_denso_diffusers / docs de progresso.

## Aprovação (cada item com teste CPU próprio, modelo sintético minúsculo)

A1. `tools/qat_ternario_klein.py` continua aceitando a MESMA CLI (todas as flags atuais, mesmos defaults);
    células do Colab chamam o mesmo caminho. Flags novas só opcionais.
A2. Um passo de treino roda em CPU com modelo sintético (nomes `transformer_blocks.N.*`), perda finita,
    códigos ternários no forward (valores do peso efetivo ∈ {−s, 0, s} por grupo).
A3. Determinismo: corrida contínua de N passos ≡ corrida parada no passo k (caminho de SIGTERM/parada) e
    retomada até N: mestre final **bit a bit igual**, mesmas perdas por passo no journal, exportado idêntico.
A4. Checkpoint no formato ANTIGO (sem `estado`) retoma: carrega mestre/otim/RNG/passo, avisa, segue o
    comportamento antigo (ordem re-embaralhada, `melhor.json` como fonte).
A5. Shard no formato ANTIGO (`p####_s#.pt`, sem `size`) é lido; o nome por conteúdo é o novo padrão;
    `valida()` recusa prompt/semente/passos/size divergentes; shard legado com prompt ≠ lista é avisado e o
    prompt certo é gravado (não pulado); vazamento treino∩holdout pelos PROMPTS DOS SHARDS é recusado.
A6. Config: `--so-escalas` + `--l1-corpo > 0` recusado antes de carregar modelo; retomada com impressão
    digital diferente recusada (salvo flag explícita); sem globais `NIVEIS`/`STE_LIGADO` (grep).
A7. Exportação lowbit_affine: dequant (kernel do loader, torch) **bit a bit igual** ao desempacotado bf16
    para ternário; int4 idem ou recusa explícita por camada; o formato antigo continua disponível e é o
    default enquanto o loader não fundir q/k/v de arquivo lowbit em nomes diffusers (verificar e relatar).
A8. `baixa_se_faltar`: só EntryNotFound/RepositoryNotFound viram "do zero"; outro erro aborta.
    `envia` nunca sobe passo menor que o `ckpt/passo.json` remoto. Testado com HfApi falso (sem rede).
A9. `status.json` atômico (fase, passo, heartbeat, fim/falha/parado); probe único parametrizado com
    fallback ao log (runs antigos); probes existentes gerados do mesmo molde e iguais ao gerado (teste).
    Parada: SIGTERM → checkpoint no fim do passo; `para()` espera a saída antes de SIGKILL; limpeza de
    braço sem checkpoint apaga só journal/melhor.json.
A10. gera_dataset: metadata por VM (sem reescrever arquivo compartilhado); só EntryNotFound vira vazio;
    shard local sobrando de corrida morta é enviado, não pulado.
A11. lowbit_canon: `ternariza`/`pilhas_reais`/`BLOCO` idênticos (bit a bit / igualdade) às cópias antigas
    em tensores aleatórios; `rtn_simetrico` idêntico ao `rtn4` do mistura; empacotamento via
    `kernel.pack_codes` do loader (import por caminho, sem editar o loader).
A12. Nós: qwen21 recusa `add_patches` não vazio; void sem `mem_get_info` em placa não inicializada
    (NVML), caminho de auditoria configurável, uma implementação de unload com as 3 classes antigas
    registradas; stream-video recusa segunda iteração.
A13. qwen21 QAT: códigos via `ck.quantize_convrot_w4a4_weight` quando importável; smoke vira assert;
    `--smoke` CPU passa antes e depois.

## Reprovação

Qualquer divergência bit a bit em A3/A7/A11; CLI antiga quebrada; teste tocando GPU; escrita fora dos
arquivos de posse. Limites declarados: nada disso prova qualidade de treino, velocidade na A100, nem
compatibilidade com torchao/diffusers da VM — isso exige Colab.
