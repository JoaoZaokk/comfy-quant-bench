import sys, time, torch
gib = int(sys.argv[1])
blocks = [torch.empty(1024*2**20, dtype=torch.uint8, device="cuda:0") for _ in range(gib)]
torch.cuda.synchronize()
print(f"HOLDING {gib} GiB on cuda:0", flush=True)
time.sleep(int(sys.argv[2]))
