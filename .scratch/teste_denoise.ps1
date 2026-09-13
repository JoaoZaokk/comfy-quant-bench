# Testa a hipotese de que `denoise=1.0` explica a deriva de textura na edicao.
#
# O QUE SE OBSERVOU (medido): os DOIS bracos -- o int8 do Comfy-Org e o nosso w4a8 -- destroem a
# textura da mesa de madeira e a do tijolo, e preservam bem um rosto. Mesmo par, mesma semente.
# Como os dois fazem igual, nao e da quantizacao.
#
# A HIPOTESE (lida do grafo, nao medida): com `denoise=1.0` o KSampler regenera a imagem inteira,
# e a preservacao vem so do condicionamento, nunca do latente de entrada. Se for isso, baixar o
# denoise tem de salvar a textura.
#
# O QUE REFUTA: se a textura continuar destruida a 0.6 e 0.4, a hipotese morre e o mecanismo e
# outro -- provavelmente o proprio VAE, ou o regime de 30 passos / cfg 2.5.
#
# UM EIXO: so `denoise` muda. Mesmo braco (int8, que e a referencia desta bateria), mesma imagem,
# mesma instrucao, mesma semente, mesmos passos, mesmo cfg, mesmo shift.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE

foreach ($d in @(1.0, 0.8, 0.6, 0.4)) {
  $nome = "qdenoise_int8_pera_d$($d.ToString('0.0').Replace('.',''))"
  Write-Host "== $nome  denoise=$d"
  & .\python_embeded\python.exe -s .\tools\qwen_edit_test.py `
      --transformer 'qwen_image_edit_2511_int8_convrot.safetensors' `
      --imagem 'edit_maca.png' `
      --instrucao 'change the apple to a green pear, keep the table, the window light and the composition exactly the same' `
      --seed 1 --denoise $d --saida $nome `
      --json ".scratch\$nome.json" --limite 2400
  Write-Host "EXIT=$LASTEXITCODE"
}
Write-Host "TESTE_DENOISE_FIM"
