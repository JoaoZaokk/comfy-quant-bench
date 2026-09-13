# INSTALADO EM: python_embeded/Lib/site-packages/sitecustomize.py
# Esta copia existe porque python_embeded/ NAO e rastreado por este repo e um
# update/ ou uma reinstalacao apagaria o arquivo vivo sem deixar rastro.

"""Contorno OPT-IN para o WMI travado desta maquina. Nao faz nada sem a variavel.

O PROBLEMA, LOCALIZADO COM faulthandler
---------------------------------------
Em 2026-09-13, depois de um `Windows fatal exception: access violation` derrubar o ComfyUI, o
servico WMI (`winmgmt`) ficou em RUNNING mas travado: toda consulta pendura. `import torch`
passou a pendurar junto, e a pilha diz exatamente onde:

    File "platform.py", line 330 in _wmi_query
    File "platform.py", line 391 in _win32_ver
    File "platform.py", line 450 in win32_ver
    File "platform.py", line 999 in uname
    File "platform.py", line 1110 in machine
    File "...\\torch\\__init__.py", line 247 in _load_dll_libraries

Nao e CUDA (pendura igual com CUDA_VISIBLE_DEVICES=""), nao e disco (as DLLs leem a 1,7 GB/s),
nao e o carregador (nvcuda.dll, torch_cpu.dll e c10.dll carregam em 0,0-0,1 s por ctypes). E o
`platform.machine()` do Python, que no Windows desce para WMI.

POR QUE NAO SE CONSERTA O SERVICO
---------------------------------
`sc enumdepend winmgmt` lista **vmms**, o Hyper-V Virtual Machine Management -- que e onde o WSL2
roda. Reiniciar o WMI derruba o WSL e com ele os containers do dono, o que e proibido nesta
maquina. Um reboot faz o mesmo. Entao o servico fica como esta e o contorno e por processo.

O QUE ISTO FAZ
--------------
Preenche `platform._uname_cache` ANTES de qualquer import pedir `platform.machine()`. Os valores
saem do **registro** (`HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion`) e de
`PROCESSOR_ARCHITECTURE` -- a mesma informacao que o WMI reportaria, por um caminho que responde.
Nao e um valor inventado, e a mesma verdade por outra porta.

Com o cache preenchido, `import torch` volta a 1,3 s e `torch.cuda.device_count()` da 2.

COMO LIGAR E DESLIGAR
---------------------
So age com `COMFY_BYPASS_WMI_UNAME=1` no ambiente. Sem a variavel este arquivo nao faz nada, de
proposito: uma instalacao compartilhada nao deve mudar de comportamento em silencio, e quando o
WMI voltar ao normal o contorno deve sair de cena sozinho -- basta parar de exportar a variavel.

Toda vez que age, escreve UMA linha no stderr. Um contorno invisivel vira uma armadilha para o
proximo que ler um `platform.uname()` e acreditar nele sem saber de onde veio.
"""
import os
import sys

if sys.platform == "win32" and os.environ.get("COMFY_BYPASS_WMI_UNAME") == "1":
    try:
        import platform

        if getattr(platform, "_uname_cache", None) is None:
            versao, lancamento = "10.0", "10"
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                ) as chave:
                    build = winreg.QueryValueEx(chave, "CurrentBuildNumber")[0]
                    produto = winreg.QueryValueEx(chave, "ProductName")[0]
                versao = f"10.0.{build}"
                # O registro ainda diz "Windows 10 Pro" numa build 26200, que e Windows 11. O
                # numero da build e o dado confiavel; o nome do produto nao e.
                lancamento = "11" if int(build) >= 22000 else "10"
                del produto
            except Exception:  # noqa: BLE001 -- sem registro, o default acima serve
                pass

            platform._uname_cache = platform.uname_result(
                "Windows",
                os.environ.get("COMPUTERNAME", "localhost"),
                lancamento,
                versao,
                os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
            )
            print(
                "[sitecustomize] COMFY_BYPASS_WMI_UNAME=1: platform.uname() preenchido do "
                f"registro ({lancamento}/{versao}) porque o WMI desta maquina esta travado. "
                "Ver o cabecalho de sitecustomize.py.",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[sitecustomize] contorno do WMI falhou: {exc!r}", file=sys.stderr)
