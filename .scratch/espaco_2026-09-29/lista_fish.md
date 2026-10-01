Revisão de espaço em F:/tts_lab (Fish Speech S2-Pro). Nada foi apagado. sha256 COMPLETO conferido em 2026-09-29.

Iguais byte a byte (candidatos a manter 1 cópia):
1. codec.pth (1,74 GiB) — idêntico nas 6 pastas (sha256 74fc41c5a7151c6f350a...):
   - F:/tts_lab/models/s2-pro/codec.pth
   - F:/tts_lab/models/s2-pro-w8-sim/codec.pth
   - F:/tts_lab/models/s2-pro-w4-sim/codec.pth
   - F:/tts_lab/noite_v9/checkpoints/s2-pro/codec.pth
   - F:/tts_lab/release/_teste_reconstrucao/w8/codec.pth
   - F:/tts_lab/release/_teste_reconstrucao/w4g128/codec.pth
   Economia se ficar 1: ~8,7 GiB (ou hardlink/symlink para não quebrar os carregadores).
2. F:/tts_lab/noite_v9/checkpoints/s2-pro/ == F:/tts_lab/models/s2-pro/ (os 2 shards idênticos:
   c4218e8ac93be83b35ee..., 76738d23465deaac4314...). Economia: ~8,5 GiB (+ codec).

Parecidos mas NÃO iguais (precisa decidir se ainda servem):
3. F:/tts_lab/release/_teste_reconstrucao/w4g128/ x F:/tts_lab/models/s2-pro-w4-sim/:
   shard 1 igual (0f7b93510868b4583454...), shard 2 diferente. ~8,5 GiB cada pasta.
4. F:/tts_lab/release/_teste_reconstrucao/w8/ x F:/tts_lab/models/s2-pro-w8-sim/: os dois shards diferentes. ~8,5 GiB cada.
   (Pelo nome, `_teste_reconstrucao` parece teste de reconstrução do release; se o teste terminou, as duas pastas
   dele somam ~17 GiB.)

Outras pastas grandes do tts_lab (não conferidas): noite_v9/saida 22,8 GiB, venvs 17,9 GiB,
models/s2-pro-w8-sim, s2-pro-w4-sim, s2-pro (10,3 GiB cada).

Perguntas para o chat do Fish: qual das pastas é a canônica; o teste de reconstrução (release/_teste_reconstrucao)
ainda é necessário; noite_v9/saida e checkpoints podem ir; os *-sim ainda são usados.
