import json, os, subprocess
MODEL = os.path.join("ComfyUI","models","diffusion_models","capybara_v0.1.safetensors")
code = ("import sys,os,json,subprocess,traceback\n"
 "sys.path.insert(0,'ComfyUI')\n"
 "sys.argv=['main.py']+json.loads(os.environ['ARGS'])\n"
 "import comfy.options; comfy.options.enable_args_parsing()\n"
 "import torch, comfy.sd, comfy.model_management as mm\n"
 "def u():\n"
 "    return int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True).stdout.strip())\n"
 "cap={}\n"
 "orig=mm.LoadedModel.model_load\n"
 "def spy(self, lowvram_model_memory=0, force_patch_weights=False):\n"
 "    cap['lowvram_arg']=int(lowvram_model_memory)\n"
 "    cap['smi_before_modelload']=u()\n"
 "    try:\n"
 "        r=orig(self, lowvram_model_memory, force_patch_weights)\n"
 "        cap['model_load_exc']=''\n"
 "        return r\n"
 "    except Exception as e:\n"
 "        cap['model_load_exc']=type(e).__name__+': '+str(e).splitlines()[0][:160]\n"
 "        raise\n"
 "mm.LoadedModel.model_load=spy\n"
 "b=u()\n"
 "m=comfy.sd.load_diffusion_model(os.environ['MODEL'])\n"
 "cap['smi_after_read']=u()\n"
 "cap['vram_state']=str(mm.vram_state)\n"
 "cap['free_torch']=mm.get_free_memory(torch.device('cuda:0'))//(1024*1024)\n"
 "cap['model_size']=m.model_size()//(1024*1024)\n"
 "cap['extra_reserved']=int(mm.extra_reserved_memory())//(1024*1024)\n"
 "cap['min_inference']=int(mm.minimum_inference_memory())//(1024*1024)\n"
 "cap['RATIO']=mm.MIN_WEIGHT_MEMORY_RATIO\n"
 "err=''\n"
 "try:\n"
 "    mm.load_models_gpu([m], memory_required=m.model_size())\n"
 "except Exception as e:\n"
 "    err=type(e).__name__+': '+str(e).splitlines()[0][:160]\n"
 "lm=mm.current_loaded_models\n"
 "cap['loaded_weight_mib']=(lm[0].model_loaded_memory()//(1024*1024)) if lm else -1\n"
 "cap['smi_before']=b; cap['smi_after']=u(); cap['outer_exc']=err\n"
 "cap['lowvram_arg_mib']=cap.get('lowvram_arg',0)//(1024*1024)\n"
 "print(json.dumps(cap))\n")
for label, args in [("--gpu-only",["--gpu-only"]), ("default",[])]:
    env=dict(os.environ, ARGS=json.dumps(args), MODEL=MODEL)
    r=subprocess.run([os.path.join("python_embeded","python.exe"),"-s","-c",code],capture_output=True,text=True,env=env,timeout=1800)
    ln=[l for l in r.stdout.splitlines() if l.startswith("{")]
    print("="*70); print("ARM:", label)
    if not ln:
        print("  sem resultado. stderr:", r.stderr.strip()[-900:]); continue
    d=json.loads(ln[-1])
    for k in ("vram_state","RATIO","model_size","free_torch","extra_reserved","min_inference",
              "lowvram_arg_mib","loaded_weight_mib","smi_before","smi_after_read","smi_after",
              "model_load_exc","outer_exc"):
        print(f"  {k:22s} {d.get(k)}")
    print(f"  {'delta_smi':22s} {d['smi_after']-d['smi_before']:+d} MiB")
