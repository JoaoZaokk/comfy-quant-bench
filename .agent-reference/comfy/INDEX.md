# Referências Comfy

Leitura sob demanda. Fontes completas e faixas sem lacunas; preservadas inclusive autocorreções.
Não executar comando histórico sem escopo autorizado. Caminhos literais em código continuam relativos à raiz do projeto.

- [Recado TTS, publicação e backward](01-recado-tts.md) — CLAUDE original 1–85.
- [Regra zero e exemplos](02-regra-zero.md) — CLAUDE original 86–135.
- [Instalação e repositórios](03-topologia-repos.md) — CLAUDE original 136–145.
- [Regras rígidas: host, WSL, VRAM compartilhada](04-hard-rules-host.md) — CLAUDE original 146–260.
- [Regras rígidas: formatos, qualidade e critérios](05-hard-rules-quantizacao.md) — CLAUDE original 261–553.
- [Regras rígidas: text encoders](06-hard-rules-text-encoder.md) — CLAUDE original 554–609.
- [Leitura versus execução e evidência](07-proveniencia.md) — CLAUDE original 610–657.
- [Ambiente e dependências](08-ambiente.md) — CLAUDE original 658–685.
- [Comandos, launchers, testes e dynamic VRAM](09-comandos-launchers.md) — CLAUDE original 686–836.
- [Conversão, verificadores e avaliação](10-pipeline.md) — CLAUDE original 837–1025.
- [LTX, áudio, condicionamento e commit](11-ltx-audio-memoria.md) — CLAUDE original 1026–1166.
- [Rodada de fechamento 14/09/2026](12-fechamento-20260914.md) — CLAUDE original 1167–1231.
- [LoRA sobre pesos quantizados](13-lora.md) — CLAUDE original 1232–1322.
- [Aceitação e rastreadores](14-aceitacao-tracking.md) — CLAUDE original 1323–1339.
- [Protocolo GPU e histórico do lock](15-lock-gpu.md) — CLAUDE original 1340–1423.
- [Rastreadores e habilidades](16-agent-skills.md) — CLAUDE original 1424–1446.
- [RAM, commit e processos de download](17-memoria-commit.md) — CLAUDE original 1447–1463.
- [Referências trazidas do global](18-referencias-global.md) — CLAUDE original 1464–1471.

## Alertas de contexto, por leitura dos documentos

Não são testes do projeto nem atualização silenciosa das notas. Os originais foram mantidos.

- Host: CLAUDE 151 chama Ubuntu de parado; 194 registra rodando. Contêineres/ocupação GPU são fotos datadas. A proibição de mexer em WSL permanece.
- Commit/pagefile: 192 e 1109 registram 98,72 GiB; 1130 ainda diz pagefile 61,2; 1454 registra limite 135,70 GiB. Ler contadores atuais antes de carga autorizada.
- Native preflight: 887 usa `normal_comfy_backend`; 893 registra substituição por `_native_probe.native_backend_ready`. Não procurar função antiga como requisito atual.
- Perfis: 951 diz só gemma/qwen, enquanto o documento descreve conversões de outras arquiteturas. Conferir perfis no código antes de planejar.
- `convrot_groupsize`: 928 diz 64/256; 946 documenta W4A4 16/64/256/1024 e W4A8 somente 256. Formatos têm caminhos distintos.
- Encoders: 608 diz nenhuma via exposta; memória `cadeado-do-text-encoder-sai` registra implementação em 12/09. Não concluir bloqueio atual pelo texto antigo.
- Smooth: MEMORY e `mapa-estado-entregavel` dizem pendente; CLAUDE 907–924 registra execução; 1192–1198 fecha comparação de qualidade. Preservar nota, apontar atualização posterior.
- Heretic W4A4: 1199 e 1223 dizem pesos fora do Hub; 1224–1225 registra publicação pedida depois pelo dono. Não refazer publicação por leitura parcial.
- Capybara/Z-Image: 507 diz teto Z-Image não medido; 479–505 contém a medição posterior. Ordem do texto não é ordem cronológica.
- LoRA: 1296 documenta sobrevivência W4A8 abaixo de 1; 1299 generaliza 1,000. Ler formato e instrumento; peso/condicionamento/saída não são a mesma métrica.
- Render: 525 proíbe imagem livre como medida de fidelidade; 1325 exige visual para aceitação. São perguntas distintas, não licença para omitir qualquer validação.
- Dynamic VRAM: desligar protege alguns modelos/fluxos, mas 757–765 relata arquivos incompatíveis com o leitor alternativo. Não generalizar flag.
- Lock: resumo antigo cita 23 testes; memória chega a 36. Correções são históricas, não uma execução de testes nesta preparação. Usar ferramenta existente e preservar lock vivo.
- `REGRA_ZERO` global diz “[MEDIDO] não volta atrás”; o próprio histórico contém instrumentos refutados. Manter proveniência e corrigir conclusão quando evidência posterior refuta.
- ComfyUI AGENTS rege inferência no core (sem novos wrappers no_grad/freeze); recado TTS trata treino em outro contexto. Não transportar implementação entre escopos por analogia.
- Quota HF, Colab, pacote/versão, ausência de remote e contagens são datados. Nota HF de recriar repo não autoriza exclusão: depende de tarefa explícita e preservação dos artefatos.
