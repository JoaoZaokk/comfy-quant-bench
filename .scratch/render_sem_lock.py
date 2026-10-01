"""quality_ladder SEM o BenchGuard, para renderizar na 3080 Ti enquanto o QAT segura o lock da 3090.

O lock e da MAQUINA (um arquivo), nao da placa; quem o segura e o proprio treino deste projeto. Este
render roda com CUDA_VISIBLE_DEVICES=1 e nao toca a 3090. O s/passo que sair daqui NAO vale como
tempo (outra placa, com o embedding do cortex residente): so as imagens contam.
"""
import runpy, sys
sys.argv = ["quality_ladder.py", *sys.argv[1:]]
g = runpy.run_path(r"F:/COMFY_PORTABLE/tools/quality_ladder.py", run_name="ladder_sem_lock")
raise SystemExit(g["main"]())
