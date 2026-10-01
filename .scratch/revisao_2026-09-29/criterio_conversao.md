# Critério — refatoração dos conversores (achados 1–10 da revisão de 2026-09-29)

Escrito ANTES de mexer em qualquer arquivo de `tools/`. Autorização: coordenador, repassando o dono ("faça tudo").
GPU proibida: todo Python com `CUDA_VISIBLE_DEVICES=-1 python_embeded/python.exe -s`.

## O que conta como aprovado

1. **Baseline e depois dos testes da área.** Os mesmos 9 `test_*.py` rodados antes (`conv/testes_antes/`) e
   depois (`conv/testes_depois/`). Nenhum teste que passava pode passar a falhar. `test_smooth_guards` falha
   no baseline por ambiente ("CUDA is unavailable" antes de qualquer guarda); se passar depois, registrar
   que foi porque a checagem de CUDA do smooth foi movida para depois das recusas (achado 10), não por
   afrouxar guarda.
2. **Prova de equivalência por conversor**, em safetensors sintéticos pequenos gerados uma vez
   (`conv/prova/fontes/`, sha256 registrado) e rodados com o código de ANTES (cópia congelada em
   `conv/antes_tools/`) e com o de DEPOIS, no mesmo processo-modelo:
   - quantizador REAL do comfy-kitchen em CPU (backend eager) onde ele existe — w4a4, w4a8, int8, mixed,
     smooth; `gguf` real para awq; `nunchaku` falso (o de `test_svdq_verify.py`) para svdq;
   - `torch.cuda.is_available` e `.to("cuda")` desviados para CPU, e `native_backend_ready` respondendo
     "cuda" — isto prova LAYOUT e BYTES do caminho de escrita, não o backend nativo;
   - critério: **arquivo de saída byte a byte idêntico** (sha256 igual) e **sidecar idêntico** depois de
     normalizar só os campos voláteis (`conversion_seconds`, e o prefixo do diretório de saída antes/depois).
   - Exceção decidida pelo dono: `quant_w4a4_smooth` passa a quantizar em FP32. Para ele: (a) rodar o DEPOIS
     com o quantizador envolvido para receber BF16 como antes — tem de sair idêntico ao ANTES, provando que a
     única mudança é o dtype de entrada; (b) rodar o DEPOIS nativo e MEDIR a diferença (fração de códigos
     int4 diferentes, erro relativo das escalas), com header e tensores preservados idênticos.
   - Exceção de header por decisão de formato que eu tomar: nenhuma planejada. Se algum script legado
     (`extrai_transformer`, `transplanta_klein`) precisar de ordem/`ensure_ascii` do header antigo, o núcleo
     ganha a opção e os bytes continuam iguais; se eu não conseguir, registro como falha deste critério.
   - `ajusta_denso_diffusers` e `grava_pesos_recuperados` gravavam com `safetensors.save_file`: o novo
     caminho em streaming tem de reproduzir o arquivo do `save_file` byte a byte (ordem por dtype+nome,
     metadata ordenada). `ajusta_denso` não roda ponta a ponta sem diffusers+treino: a prova é da função de
     escrita isolada contra o trecho antigo, e isso fica registrado como cobertura parcial.
3. **Header planejado** (`_conversion.header_bytes`) do DEPOIS igual ao header gravado pelo ANTES, para cada
   caso — decorre do item 2, e `tools/verificar_migracao.py --par w4a4` continua passando nos pares reais do
   disco (só lê headers, CPU).
4. **Seleção de camadas inalterada nos modelos reais**: para cada `.safetensors` real das raízes de modelo
   cujo perfil é detectado, a lista de camadas selecionadas por w4a4/w4a8/int8/awq/mixed(file) com o código
   novo é idêntica à do antigo (lendo só headers). Diferença aceitável única: autodetecção LTX no w4a4
   passa a funcionar (decisão do dono) — antes levantava ValueError.
5. **Contratos novos testados** num teste novo sem GPU: escrita ponta a ponta por formato com safetensors
   sintético + quantizador falso em CPU; sidecar dentro do commit atômico (falha no meio não deixa saída
   nem sidecar; sidecar existente recusa); dry-run do w4a4 recusa saída existente; `_ram_guard.check_commit`.
6. Regras duras preservadas e conferidas por leitura do diff: streaming (nenhum `safe_open`/`load_file`
   novo; os que existiam em svdq/ajusta saem), `.partial` → `os.replace`, recusas, perfis estritos, nunca
   sobrescrever saída/sidecar.

## O que NÃO prova, dito antes

- Nada disto roda kernel CUDA nem prova que o backend nativo resolve. Os bytes vêm do backend eager em CPU;
  o CUDA pode produzir outros números (é outro backend), e isso não muda com a refatoração — mas a
  equivalência CUDA antes/depois só se prova reconvertendo na GPU. Validações GPU vão listadas no resultado.
- Modelos reais não são reconvertidos.
