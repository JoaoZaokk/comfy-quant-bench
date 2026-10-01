> Preservado do CLAUDE.md global em 23/09/2026. Datas, medições e relatos abaixo são históricos; não representam revalidação atual. Regras específicas do projeto continuam aplicáveis.

## COLAB — ler ANTES de criar qualquer VM

Em 2026-09-20/21 eu matei **quatro VMs** e dei **tres diagnosticos errados**
("acabou o credito" — havia 130 CU; "concorrencia no sqlite" — falso, eu repeti
do runbook como se fosse medicao minha; "a A100 tinha poll real" — era log de
outra sessao). As causas reais sao tres e todas tem regra:

1. **`nohup` sozinho deixa o KERNEL ocioso e a VM e' podada em 22-26 min.**
   O Colab conta atividade de kernel do notebook, nao CPU da maquina, e o
   keep-alive HTTP do CLI **nao** substitui isso.
   -> **Nunca so' `nohup`. Background + supervisor com probe REAL de kernel a
   cada 4 min.** Supervisor validado (90 min ao vivo) em
   `C:/Users/joaoz/projetos/project_quant_merge_frankestein/.worktrees/colab-gpu-workspace/scripts/colab/colab_job_supervisor.py`
2. **O token do runtime-proxy expira perto de 60 min** (upstream #106): a VM
   vive, `exec`/`download` passam a dar 404/401.
   -> Reconsultar `/assignments` e adotar `token`+`url` novos antes de cada probe.
3. **Erro de rede transitorio nao e' VM morta.** Retry ate' 3x, confirmando
   primeiro que a sessao ainda existe.

**Regras, sem excecao:**

- **Nunca montar ambiente dentro da GPU paga.** Minhas tres primeiras VMs
  morreram **durante o `pip`**, com o bundle binario pronto do lado. Usar o
  bundle (A100 `sm_80` + L4 `sm_89`); nao recompilar FlashAttention/Sage. T4 fora.
- **Antes de subir VM, comparar com o RTF local — no MESMO REGIME.** O "3,6x"
  que eu mesmo escrevi aqui estava errado: era 3090 **com** compile (RTF 0,699)
  contra L4 **sem** compile (2,49). Regua diferente nos dois lados.
  No mesmo regime, eager: **3090 10,58 tok/s contra L4 8,65 — 1,22x**, e a
  A100-40 do Colab deu 8,4, mais LENTA que a L4 (a geracao e' limitada por
  overhead, nao por banda). Com compile a 3090 vai a ~31 tok/s e a L4 e' sm_89,
  tem o mesmo caminho.
  **O que isso muda:** uma placa contra a outra o Colab perde; o que ganha e'
  PARALELISMO. A conta certa e' vazao total e CU/h, nao placa contra placa —
  e por CU a L4 (1,54/h) ganha da A100 com folga neste workload.
- **`colab exec` NAO passa argumentos.** Nao existe `-- --flag`: as opcoes sao
  so' `-s`, `-f`, `--output-image`, `--timeout`, `--env`. Argumento vai DENTRO
  do codigo — um wrapper que seta `sys.argv` e chama `runpy.run_path`.
- **`colab upload` devolve 500 se a pasta destino nao existir.** Criar o
  diretorio com um `exec` antes; o erro nao diz isso.
- **Um dono unico do CLI por sessao.** Nada de segundo poller nem `colab exec`
  manual enquanto o supervisor roda.
- **`/content` e' descartavel.** Journal e checkpoint fora dele, escrita atomica.
- **A100-80GB exige `--gpu A100 --high-mem`** (sem isso vem a de 40 GB);
  conferir com `nvidia-smi` antes de gastar.
- **Terminar com `colab stop` e conferir `colab sessions`.**

**E dois erros de LEITURA que custaram tanto quanto:**

- **`colab.log` e' global e mistura sessoes de projetos diferentes.** Escopar por
  endpoint (`gpu-l4` / `gpu-a100`) antes de concluir qualquer coisa dele.
- **Silencio nao e' saude.** A mensagem real de morte e'
  `appears to be lost (404/401)`, que meu filtro nao cobria — li sessao morta
  como "rodando". Filtro de monitor cobre **todo desfecho**, nao so' o feliz.
