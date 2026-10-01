"""Parte 50 do W4A4_PROGRESS.md: audio do 2.5 medido, LoRA na saida, dois servidores mortos, 2.3 em curso."""
import io

p = 'W4A4_PROGRESS.md'
s = io.open(p, encoding='utf-8').read()
parte = """

## Parte 50 -- 2026-09-13 (noite): o som do LTX 2.5, o LoRA na saida, dois servidores mortos e o 2.3 pela metade

**O audio do LTX 2.5, medido.** Tres bracos rerenderizados com o ramo de audio decodificado, mesma
semente. Os quadros voltaram PIXEL A PIXEL iguais aos da primeira rodada nos tres bracos (MAE 0,0
em 1/63/125/187/249, atravessando reinicio de servidor e, no BF16, outro disco) -- entao os numeros
de video sao os mesmos e os de audio sao novos:

    braco               MAE   PSNR   SSIM  | log-mel L1  SNR      lag   conv   RMS
    int8 Lightricks    4,10  29,71  0,941  |   0,041    11,2 dB  0 ms  0,124  -38,8 dBFS
    W4A8 nosso         7,81  25,39  0,895  |   0,120     3,4 dB  0 ms  0,311  -38,3 dBFS
    controle silencio                      |   6,980     0,0 dB
    controle ruido branco, mesmo RMS       |   1,471    -3,0 dB         1,097

O som ordena igual ao quadro e por margem maior (1,9x no quadro, 2,9x no som); ninguem mudou de
nivel nem deslocou no tempo. A1, A2 e A3 do `bench/criterio_ltx23.md` confirmadas. Card do 2.5
reescrito com a correcao no topo; MP4 com faixa, FLAC, folha e JSON no repo do Hub
(`LTX25_AV_UPLOAD_OK`, 10 arquivos conferidos por listagem).

**O LoRA na saida contradiz o LoRA no peso.** Qwen-Image-Edit W4A8 + Lightning 4 passos, tres
pares, cinco bracos (`bench/qwen_edit_lora/grade_lightning_4passos.png`): sem o LoRA, INT8 e W4A8
FALHAM igual (sem cachecol, OPEN continua OPEN, maca-pera pintalgada, tudo sobreafiado -- o
controle que tinha de falhar, falhou); com o LoRA, INT8, W4A8 fundido e W4A8 bypass obedecem as
tres instrucoes; fundido vs bypass divergem 1,26 na regiao quieta, contra 3,5-3,9 de cada um para
o INT8. No peso, este era o pior caso da tabela: sobrevivencia 0,86, ruido 103x o delta. **Numero
por peso ordena e alarma, nao decide** -- a mesma licao do erro por camada entre formatos. No LTX
2.5 + squish (sem palavra-gatilho, furo de desenho): fundido e bypass caem na MESMA composicao,
6,3 MAE entre si, contra 15-18 da referencia e 28 de outra semente. Tudo em
`bench/criterio_lora.md`, com R1-R6 escritas antes.

**Dois servidores mortos pelo mesmo golpe.** `Windows fatal exception: access violation` em
`torch/storage.py __getitem__`, dentro do mmap de `load_torch_file`: (1) o transformer BF16 do
2.5 (39 GiB) lido de D: (SMB) com ~24 GiB de RAM livre; (2) o checkpoint UNICO do 2.3 (43 GiB)
pelo caminho de checkpoint + DisTorch2, lido de W: (disco local) com ~40 GiB livres. O que
sobreviveu, duas vezes: o transformer sozinho pelo `UNETLoader`, de disco local, 39 GiB. Por isso
`tools/extrai_transformer.py` extrai o DiT do checkpoint unico por faixa de bytes (sem mmap) e o
braco BF16 do 2.3 roda por esse caminho, com VAEs e projecao de arquivos pequenos byte a byte
iguais aos do checkpoint (1503 tensores conferidos por hash entre BF16, W4A8 e W4A4). E cada
loader auxiliar do ComfyUI le o arquivo INTEIRO que recebe, entao apontar `LTXVAudioVAELoader` e
`LTXAVTextEncoderLoader` para o checkpoint de 43 GiB mapeava 43 GiB duas vezes a mais por braco.

**LTX 2.3 distilled 1.1, ate agora.** W4A8 (16,65 GB, 1440 camadas) e W4A4 (15,37 GB, controle)
convertidos na 3080 Ti; Gemma 3 12B de fabrica baixado (Comfy-Org, 22,71 GiB) e convertido em
W4A8 (8,31 GiB, 336 camadas). Bracos de 10 s com audio, 8 passos, ja renderizados: W4A8 (638 s
com carga do SMB), W4A4 (482 s), GGUF Q6_K de terceiro (622 s) -- os dois nossos coerentes a olho
(farol, ondas, passaros; o W4A4 NAO quebrou, como o riftcast no 2.5). O BF16 e a comparacao dos
quatro estao na fila, e a rodada de LoRA do 2.3 (Product Commercial) tambem -- a primeira tentativa
dela falhou por erro meu (apaguei a copia que o probe usava de fonte, e o servidor ainda subia).

**Perfil W4A4 para LTX** acrescentado a `quant_w4a4.py` (mesmo regex do W4A8, 1440/1772 nos dois
headers), porque o controle W4A4 do 2.3 recusou em dois segundos por falta dele.

**GitHub.** O remoto `comfy-quant-bench` (19 commits, pacote curado) foi fundido no master com
`--allow-unrelated-histories -X ours` (versoes locais em todo conflito, README/LICENSE/docs deles
preservados) e o master empurrado para `main` (`5537141..ea9e417`). README reescrito com o que
mudou desde 2026-09-01. Varredura de segredos nos 480 arquivos rastreados: nenhum token; os
`password` que apareceram eram `desenha`.

Nao coberto nesta parte: os numeros do 2.3 contra o BF16 (na fila); LoRA do 2.3 (na fila); uma
semente e um prompt em tudo.
"""
io.open(p, 'w', encoding='utf-8').write(s.rstrip('\n') + parte)
print('parte 50 appended')
