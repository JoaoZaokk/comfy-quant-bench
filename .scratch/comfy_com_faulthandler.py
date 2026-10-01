"""Sobe o ComfyUI com um despertador do faulthandler, para ele mesmo dizer onde pendura."""
import faulthandler, os, runpy, sys
faulthandler.dump_traceback_later(int(os.environ.get("DESPERTA", "90")), exit=True)
sys.argv = ["main.py", "--use-sage-attention", "--disable-dynamic-vram", "--cache-none",
            "--preview-method", "none", "--listen", "127.0.0.1", "--port", "8192"]
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ComfyUI"))
sys.path.insert(0, os.getcwd())
runpy.run_path("main.py", run_name="__main__")
