"""Record ComfyUI node events with process, system RAM and per-GPU memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import psutil
import pynvml


def gpu_rows():
    rows = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        rows.append(
            {
                "index": index,
                "name": pynvml.nvmlDeviceGetName(handle),
                "used": memory.used,
                "free": memory.free,
                "total": memory.total,
                "gpu_utilization": utilization.gpu,
            }
        )
    return rows


def sample(pid, event, data):
    process = psutil.Process(pid)
    memory = process.memory_info()
    system = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "data": data,
        "pid": pid,
        "process_rss": memory.rss,
        "process_private": getattr(memory, "private", None),
        "system_total": system.total,
        "system_available": system.available,
        "swap_used": swap.used,
        "gpus": gpu_rows(),
    }


async def run(args):
    pynvml.nvmlInit()
    client_id = uuid.uuid4().hex
    ws_url = f"{args.server.rstrip('/')}/ws?clientId={client_id}".replace("http://", "ws://").replace("https://", "wss://")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with aiohttp.ClientSession() as session, session.ws_connect(ws_url, heartbeat=30) as ws:
            with args.output.open("a", encoding="utf-8") as handle:
                while True:
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=args.interval)
                    except asyncio.TimeoutError:
                        row = sample(args.pid, "interval", {})
                    else:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(message.data)
                            event = payload.get("type", "message")
                            data = payload.get("data", {})
                            if event not in args.events:
                                continue
                            row = sample(args.pid, event, data)
                        elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        else:
                            continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if row["event"] in ("execution_success", "execution_error", "execution_interrupted"):
                        break
    finally:
        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8190")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--events",
        nargs="+",
        default=("execution_start", "executing", "execution_cached", "execution_success", "execution_error", "execution_interrupted"),
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

