#!/bin/bash
# Sobe para /content/qat o codigo do QAT klein (qat_ternario_klein.py + qat_klein/ + lowbit_canon.py +
# ajusta_denso_diffusers.py + colab_ops.py + probe_job.py + kernel do loader como lowbit_kernel.py).
# Fonte unica: tools/colab_qat/arquivos_qat.txt, GERADO de colab_ops.ARQUIVOS_QAT por
#   python_embeded\python.exe -s tools/colab_qat/colab_ops.py monta
# (as celulas de lancamento conferem a mesma lista com `exige` e recusam com FALTA).
#
# Uso (depois de definir C=<colab> e W=<raiz da bancada vista pelo colab>):
#   . "$(dirname "$0")/sobe_qat_comum.sh"; sobe_qat "$S"
# L = raiz da bancada vista por ESTE shell, se diferente de W (ex.: Git Bash com colab via wsl.exe).
# Os jobs dos scripts continuam subindo o que e' so' deles (fila.json, config, prompts, celulas, token).

sobe_qat() {
    local s="$1" raiz="${L:-$W}" origem destino n=0
    local man="$raiz/tools/colab_qat/arquivos_qat.txt"
    if [ -z "$s" ] || [ -z "$C" ] || [ -z "$W" ]; then
        echo "sobe_qat: defina C e W e passe a sessao" >&2; return 2
    fi
    if [ ! -f "$man" ]; then echo "FALTA $man" >&2; return 1; fi
    while IFS=: read -r origem destino; do
        case "$origem" in ''|\#*) continue ;; esac
        if [ ! -f "$raiz/$origem" ]; then echo "FALTA $raiz/$origem" >&2; return 1; fi
    done < "$man"
    # /content/qat/qat_klein precisa existir antes do upload do pacote
    timeout 120 $C exec -s "$s" --timeout 60 -f "$W/tools/colab_qat/celula_pastas.py" < /dev/null 2>&1 \
        | grep -v '^\[colab\]' | tail -1
    while IFS=: read -r origem destino; do
        case "$origem" in ''|\#*) continue ;; esac
        timeout 300 $C upload -s "$s" "$W/$origem" "/content/qat/$destino" < /dev/null 2>&1 \
            | grep -v '^\[colab\]' | tail -1
        n=$((n + 1))
    done < "$man"
    echo "sobe_qat: $n arquivos do QAT enviados para $s"
}
