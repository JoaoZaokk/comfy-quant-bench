#!/bin/bash
# Bateria por configuração de driver: M1 (cópia), M2 (render p2s42 no ComfyUI), M3 (llama.cpp Vulkan). Uso: bench_driver.sh <tag>
TAG=$1; O=~/arc_leve/$TAG; mkdir -p $O; R=$O/resumo.txt; : > $R
T0=$(date +%s); DESDE="@$T0"
drv=$(basename "$(readlink /sys/bus/pci/devices/0000:01:00.0/driver)")
echo "tag=$TAG driver=$drv iommu_group=$(cat /sys/bus/pci/devices/0000:01:00.0/iommu_group/type)" >> $R
echo "cmdline=$(cat /proc/cmdline)" >> $R
for f in /sys/class/drm/card*/device/tile0/gt0/engines/bcs/job_timeout_ms; do [ -e $f ] && echo "bcs_job_timeout_ms=$(cat $f)" >> $R; done
docker stop spark-x25-heretic >/dev/null 2>&1
cd ~/ComfyUI && . venv/bin/activate
for a in "page 64" "page 256" "pin 256"; do
  timeout -s KILL 60 python ~/bench_copia.py $a 2>&1 | grep "^M1" >> $R || echo "M1 $a TRAVOU(60s)" >> $R
done
deactivate
curl -s -m 10 -X POST http://127.0.0.1:8188/manager/reboot >/dev/null 2>&1; sleep 5
for i in $(seq 60); do curl -s -m 3 -o /dev/null http://127.0.0.1:8188/system_stats && break; sleep 3; done
T1=$(date +%s)
timeout 900 python3 ~/roda_arc.py $O p2s42 > $O/roda.log 2>&1
echo "M2 $(cut -c1-200 $O/roda.log | tr '\n' ' ')" >> $R
journalctl -u comfyui --no-pager --since "@$T1" -o cat | tr '\r' '\n' | grep -E "25/25" | tail -1 | sed 's/^/M2 sampler: /' >> $R
journalctl -u comfyui --no-pager --since "@$T1" -o cat | tr '\r' '\n' | grep -E "Prompt executed|loaded (completely|partially)|OUT_OF|DEVICE_LOST" | head -6 | sed 's/^/M2 log: /' >> $R
echo "M2 resets_kernel=$(journalctl -k --no-pager --since "$DESDE" -o cat | grep -c -i 'engine reset\|GPU HANG\|reset')" >> $R
docker start spark-x25-heretic >/dev/null
for i in $(seq 60); do curl -s -m 3 http://192.168.3.52:8088/health | grep -q ok && break; sleep 3; done
for k in 1 2; do
  curl -s -m 180 http://192.168.3.52:8088/completion -d '{"prompt":"Write a short paragraph about rivers.","n_predict":128,"temperature":0,"seed":1,"cache_prompt":false}' \
   | python3 -c 'import json,sys;t=json.load(sys.stdin)["timings"];print("M3 tg_tok_s",round(t["predicted_per_second"],2),"pp_tok_s",round(t["prompt_per_second"],1))' >> $R 2>&1 || echo "M3 falhou" >> $R
done
echo "fim $(( $(date +%s) - T0 ))s" >> $R
cat $R
