# Repository Guidelines

Instalação ComfyUI Portable viva, com repositórios sobrepostos. Estas diretrizes são a fonte única; CLAUDE.md importa este arquivo.
Antes de editar ComfyUI/, ler também [ComfyUI/AGENTS.md](ComfyUI/AGENTS.md). A organização não revoga decisões ou notas preservadas.

## Restrições e autorização

- Apenas `python_embeded/python.exe`; scripts diretos com `-s`. Nunca Python/pip/Conda globais.
- Nunca apagar, mover, sobrescrever ou requantizar modelos originais. W4A4 novo: ao lado da fonte, sufixo `_w4a4_convrot`, sidecar `.quant.json`; nunca sobrescrever saída/sidecar/partial existente. Não quantizar VAE.
- QAT klein: o dono vetou treino na RTX 3090 (memória de 23/09); inferência/render local é outro escopo. Não tratar estimativa de capacidade de QAT como autorização para treino.
- Conversores: manter streaming, escrita atômica `.partial` → `os.replace`, recusas e perfis estritos. Não reintroduzir `safe_open`/mmap para fontes grandes; confirmar loader/config antes de estender arquitetura.
- W4A4 exige execução ConvRot CUDA nativa, não peso INT4 com matemática BF16. Troca para INT8/W4A8 depende da decisão do dono; não inferir autorização do histórico.
- Não atualizar Torch/CUDA/ComfyUI/comfy-kitchen em massa, nem executar `ruff --fix` em massa. Inspecionar versões/APIs e preservar pins. Instalação depende do escopo autorizado pelo dono.
- Não parar/reiniciar/reconfigurar WSL ou containers; não mudar pagefile, matar processos ou podar volumes automaticamente. Mudança de topologia não revoga a regra.
- Não restaurar `cudart64_12.dll.disabled`. Antes de carga grande, conferir commit livre: guardas de RAM/disco não bastam; leitor normal pode comprometer 2× arquivo, mais o modelo.
- Janela GPU é decisão do dono. Dentro de tarefa autorizada, ler [protocolo completo](.agent-reference/comfy/15-lock-gpu.md): bloco nomeado de 30 min, dispositivo explícito e ocupação das duas placas conferida.
- `_timing.compare()`/BenchGuard toma próprio lock; não aninhar lock de outro processo. Demais trabalhos GPU usam `Assert-GpuLock`, nunca `Take-GpuLock | Out-Null` nem escrita manual de lock.
- Nunca quebrar lock vivo. Se recusado: conferir placa/PID/heartbeat por cerca de um minuto, informar dono e seguir trabalho sem GPU. Soltar só após trabalho parar e apenas lock ainda próprio.

## Estrutura, Git e edição

- `ComfyUI/` é checkout próprio; `ComfyUI/custom_nodes/` pode conter outros repos. Verificar fronteira antes de editar/commitar; contagens, versões, branches e remotes devem ser consultados, não copiados do histórico.
- Modelos em `ComfyUI/models/`, raízes extras no YAML; estado/workflows em `ComfyUI/user/`; utilitários em `tools/`. Raízes extras podem ser compartilhamentos de rede.
- Root usa allowlist de Git. Antes de stage autorizado, listar arquivos e conferir `git add -An --dry-run`; verificar ignore efetivo de checkpoints/payloads, inclusive sob `.scratch/`. Diretório de tickets incluído não implica que pesos e resultados grandes estejam ignorados.
- Quatro espaços, imports de módulo, mudanças pequenas, estilo local; sem abstrações/dependências especulativas. Preservar interfaces, dtype/device/offload, carregamento e compatibilidade.
- Sem novos caminhos de internet no core ComfyUI; download de modelo apenas quando explicitamente solicitado/autorizado, limitado ao artefato, conforme regra do checkout.

## Evidência, aceitação e registro

- [REGRA_ZERO](.agent-reference/REGRA_ZERO.md): confirmar ausência por outro caminho; distinguir medido, lido, hipótese e relato histórico. Documento não é medição desta sessão.
- [Incidentes de medição](.agent-reference/MEDICAO_INCIDENTES.md): registrar métrica, unidade, placa, dispatch, controles, repetições e dispersão; escrever critério antes de medir. Evidência posterior pode corrigir conclusão anterior.
- Estrutura, backend declarado, despacho e erro por camada não aprovam qualidade. Aceitação: integridade/metadata → loader real → encoding → benchmark pareado e avaliação visual, dentro do escopo autorizado. LTX inclui áudio.
- Fidelidade entre quantizações usa entradas/trajetória casadas; render final continua necessário para qualidade. Divergência de latente isolada não aceita build. Registrar negativos e limites.
- Testes `test_*.py`; escolher comandos adequados à tarefa em [comandos e launchers](.agent-reference/comfy/09-comandos-launchers.md). Comando documentado não concede autorização nova.
- Atualizar [W4A4_PROGRESS.md](W4A4_PROGRESS.md), [W4A4_HANDOFF.md](W4A4_HANDOFF.md), inventário gerado e sidecars como parte do trabalho; nunca editar inventário manualmente.
- Rastreadores locais `.scratch/`, critério antes do resultado; não criar issue remota por inferência. Commit curto e imperativo. PR: problema, mudança, testes, modelo/hardware e números comparáveis; upstream exige execução, não só leitura.

## Ler conforme a tarefa

- Atualização, restauração, clones ou custom nodes: [fontes canônicas e recuperação](docs/RESTAURO_BANCADA.md). Preservar patches e arquivos novos antes de atualizar; wrappers instalados não se restauram sozinhos.
- [Índice completo e alertas](.agent-reference/comfy/INDEX.md): original integral por trechos, inclusive autocorreções; [AGENTS anterior integral](.agent-reference/comfy/AGENTS-original.md).
- Conversão/modelos: [hard rules quantização](.agent-reference/comfy/05-hard-rules-quantizacao.md), [pipeline](.agent-reference/comfy/10-pipeline.md), [aceitação](.agent-reference/comfy/14-aceitacao-tracking.md).
- Encoders: [travas e medições](.agent-reference/comfy/06-hard-rules-text-encoder.md), [rodada posterior](.agent-reference/comfy/12-fechamento-20260914.md); memória `cadeado-do-text-encoder-sai` registra implementação de 12/09. Não presumir bloqueio atual pelo relato antigo.
- Launcher/dependências/dynamic VRAM: [ambiente](.agent-reference/comfy/08-ambiente.md), [comandos](.agent-reference/comfy/09-comandos-launchers.md). Não generalizar flag: há incompatibilidades por leitor/modelo.
- LTX/LoRA: [LTX, áudio e commit](.agent-reference/comfy/11-ltx-audio-memoria.md), [LoRA](.agent-reference/comfy/13-lora.md), [RAM/commit](.agent-reference/comfy/17-memoria-commit.md).
- Cortiq: [CORTIQ_LTX25_HANDOFF.md](CORTIQ_LTX25_HANDOFF.md); pinar placa e conferir artefato. Plano: [.scratch/estado-entregavel/map.md](.scratch/estado-entregavel/map.md).
- QAT klein: ler `bench/plano_qat_noite_2026-09-23.md` e a memória `qat-klein-plano-ancora` antes de retomar; VM UTC e local UTC−3, holdout VM/local não é comparação casada. Estado narrado não substitui consulta atual autorizada.
- Publicação/treino: [recado TTS](.agent-reference/comfy/01-recado-tts.md). Antes de operar Colab, ler [COLAB](.agent-reference/COLAB.md): checkpoints fora do runtime e desfecho conferido; não importar protocolo de lock de outra bancada.

Versões, quotas, processos, ocupação GPU e commit são estados variáveis; consultar somente por meios autorizados. Notas técnicas preservam fatos datados, hipóteses e correções, sem promover histórico a estado atual.
