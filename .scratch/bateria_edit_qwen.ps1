# A edicao que eu deveria ter feito desde o comeco.
#
# Tres bracos x tres pares (imagem, instrucao) x duas sementes = 18 renderizacoes.
# O eixo e o TRANSFORMER; encoder, VAE, escala, passos, cfg, shift e sampler ficam fixos.
#
# O braco BF16 FOI REMOVIDO em 2026-09-13, depois de quatro falhas registradas, nao por
# escolha de escopo:
#   UNETLoader, placa conferida limpa em 287 MiB  -> CUDA error: out of memory (de mem_get_info)
#   DisTorch2 cpu,40gb                            -> Windows fatal exception: access violation
#   depois dessa, a maquina paginou tanto que nvidia-smi levou >120s
#   --lowvram                                     -> servidor travou no proprio arranque
# No grafo de EDICAO o transformer de 38,05 GiB coexiste com o encoder de 15,45 GiB e ainda ha
# um VAEEncode. O mesmo arquivo roda a 113,9 s/render no ladder de t2i porque LA a ferramenta
# forca NORMAL_VRAM antes de carregar; o servidor nao faz isso. Um checkpoint que sabidamente
# funciona pode ser inalcancavel por outra porta de entrada.
#
# Consequencia para a leitura dos resultados: a REFERENCIA desta bateria NAO e o original. E o
# int8 do Comfy-Org, que e o braco que venceu em fidelidade em cinco familias seguidas -- teste
# duro, mas outro teste.
#
# Os bracos sao os dois que EXISTEM e prestam: o BF16 (referencia), o int8 do Comfy-Org, e o
# nosso W4A8 -- que e o unico que publiquei. O `w4a4` e o `misto` sairam ruido em t2i e foram
# apagados do disco; nao ha o que medir neles aqui.
#
# ORDEM: braco por fora, par por dentro. Isso carrega cada modelo UMA vez em vez de dezoito.
# A ordem oposta foi tentada nesta bancada e custou 24 cargas onde bastavam 3.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE

$bracos = @(
  @{ arq = 'qwen_image_edit_2511_int8_convrot.safetensors'; rot = 'int8'; dist = $false },
  @{ arq = 'qwen_image_edit_2511_w4a8.safetensors';         rot = 'w4a8'; dist = $false }
)

# Instrucoes escolhidas para serem CONFERIVEIS a olho, nao vagas: cada uma tem um
# criterio de acerto que nao depende de gosto.
$pares = @(
  @{ img = 'edit_maca.png';     ins = 'change the apple to a green pear, keep the table, the window light and the composition exactly the same'; id = 'pera' },
  @{ img = 'edit_pescador.png'; ins = 'add a red knitted scarf around his neck, change nothing else'; id = 'cachecol' },
  @{ img = 'edit_placa.png';    ins = 'change the word on the sign to CLOSED, keep the same enamel sign, the same brick wall and the same lighting'; id = 'closed' }
)

foreach ($b in $bracos) {
  foreach ($p in $pares) {
    foreach ($s in @(1, 2)) {
      $nome = "qedit_$($b.rot)_$($p.id)_s$s"
      Write-Host "== $nome"
      & .\python_embeded\python.exe -s .\tools\qwen_edit_test.py `
          --transformer $b.arq `
          --imagem $p.img `
          --instrucao $p.ins `
          --seed $s `
          --saida $nome `
          --json ".scratch\qedit_$($b.rot)_$($p.id)_s$s.json" `
          --limite 3600 @(if ($b.dist) { '--distorch' })
      Write-Host "EXIT=$LASTEXITCODE"
    }
  }
}
Write-Host "BATERIA_EDIT_FIM"
