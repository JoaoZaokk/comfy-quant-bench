# Atualização e restauração da bancada

Revisado em 24/09/2026 por leitura, hashes, replay de patch em árvore temporária e teste CPU. Não houve atualização da instalação, build de kernels ou validação GPU.

## Fontes e destinos

| Componente | Fonte de manutenção | Destino que executa |
|---|---|---|
| Conversores atuais | `tools/quant_w4a4.py`, `quant_w4a8.py`, `quant_mixed.py` e `_conversion.py` na raiz | O script escolhido explicitamente, com Python embutido |
| `comfy-convrot-w4a4/` | Checkout legado, com contrato/perfis diferentes | Não escolher como conversor atual por semelhança de nome |
| Node ConvRot | `__init__.py` e `compile_support.py` na raiz | Cópias em `ComfyUI/custom_nodes/comfy_convrot_native/` |
| Preflight e VOID | Pacotes em `custom_nodes/` da raiz | Wrappers em `ComfyUI/custom_nodes/` apontam para essas fontes |
| ComfyUI, ComfyLite, deepcompressor e vários custom nodes | Checkouts separados | Conferir a fronteira do Git antes de editar, atualizar ou restaurar |

As três cópias dos dois arquivos do node ConvRot eram idênticas em 24/09; isso não significa sincronização automática. Depois de editar a fonte, comparar e instalar deliberadamente no destino. Não importar o node para comparar: seu import registra operações Torch. Não apagar o checkout legado antes de conferir os comandos/workflows que o referenciam.

## Antes de atualizar o ComfyUI

`update/update.py` faz stash e pull, mas não reaplica o stash. Stash preserva o trabalho, porém o runtime atualizado pode ficar sem as modificações. Arquivos não rastreados exigem backup separado. Não usar o updater como limpeza rotineira.

1. Confirmar autorização, ausência de execução afetada, HEAD, status e alterações em cada checkout. Preservar também arquivos novos; não usar clean/reset/restore para obter árvore limpa.
2. Guardar diffs e novos arquivos fora do checkout. Os patches abaixo não garantem compatibilidade com upstream futuro.
3. Na raiz da bancada, `git -C ComfyUI apply --reverse --check ../patches/<arquivo>.patch` verifica se o efeito atual pode ser revertido, sem reverter. Se já aplicado, não aplicar de novo. `git -C ComfyUI apply --check ../patches/<arquivo>.patch` verifica se a aplicação seria possível.
4. Se nenhum sentido passar, investigar o conflito; não forçar nem descartar alterações. Após atualização autorizada, validar loader, testes e caminho real afetado.

- `patches/comfyui_text_encoder_quantized_math.patch`: mudanças já existentes em CLI/ops/sd/sd1_clip.
- `patches/comfyui_symmetric_patchifier_cpu_scalars.patch`: patchifier e teste CPU, incluído em 24/09. Replay a partir do HEAD reproduziu ambos os arquivos; reverse-check na árvore atual passou; os dois testes CPU passaram. Captura CUDA não foi reexecutada.

## Wrappers e dependências locais

Templates literais dos dois wrappers: `tools/node_stubs/<pacote>/__init__.py.template`. Na restauração, criar o diretório correspondente em `ComfyUI/custom_nodes/` e copiar como `__init__.py`, após comparar/preservar eventual destino existente. Os caminhos relativos só fazem sentido no destino; não executar os templates em `tools/`.

O código de negócio continua em `custom_nodes/` na raiz. Verificar carregamento na próxima inicialização autorizada; existência do wrapper não comprova que os nodes carregam.

`deepcompressor` contém adaptações locais e compila extensão durante import. Preservar diffs, configs locais e `run_deepcompressor.bat` separadamente. Não usar import como inspeção passiva nem transplantar flags entre versões sem verificar APIs. Backups privados desta organização constam no relatório externo da tarefa; configs locais não foram publicadas.

## Versionamento

Referências `.agent-reference/**/*.md` são elegíveis para Git; binários nessa árvore continuam excluídos. Checkpoints `.safetensors`, `.pt`, `.pth` e `.ckpt` em `.scratch` são ignorados; tickets Markdown e scripts permanecem elegíveis. Revisar `git add -An --dry-run`: outros tipos de payload exigem inspeção.

Memórias do Claude ficam no perfil local, fora do repo. Snapshots desta organização não são backup dos modelos ou da instalação inteira.
