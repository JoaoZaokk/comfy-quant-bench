cd /d "%~dp0"

.\python_embeded\python.exe -s ComfyUI\custom_nodes\ComfyUI-Nuvu\pre_launch.py
set CUDA_VISIBLE_DEVICES=0,1
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --fast fp16_accumulation
echo If you see this and ComfyUI did not start try updating your Nvidia Drivers to the latest. If you get a c10.dll error you need to install vc redist that you can find: https://aka.ms/vc14/vc_redist.x64.exe
pause
