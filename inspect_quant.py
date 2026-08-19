import json
import struct
import sys
from collections import Counter
from pathlib import Path

def inspect_safetensors(path):
    path = Path(path)

    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: shorter than a safetensors header prefix")
        header_size = struct.unpack("<Q", raw)[0]
        # Bounds check before reading. Without it a truncated or non-safetensors file makes this
        # try to allocate whatever the first 8 bytes happened to say -- the sibling parser in
        # tools/quant_audit.py has this check and the other three copies do not.
        limit = min(path.stat().st_size - 8, 1024 ** 3)
        if header_size <= 2 or header_size > limit:
            raise SystemExit(f"{path}: header size {header_size} is outside 2..{limit}; "
                             "this is probably not a safetensors file")
        blob = f.read(header_size)
        if len(blob) != header_size:
            raise SystemExit(f"{path}: header truncated ({len(blob)} of {header_size} bytes)")
        header = json.loads(blob)

    tensors = {
        k: v for k, v in header.items()
        if k != "__metadata__"
    }

    dtypes = Counter(v.get("dtype", "?") for v in tensors.values())

    # "offset" was in this list and matches no tensor name -- it appears in `data_offsets`, which
    # is a field of the header entry, not a key. Dropped rather than left as a needle that can
    # only produce false positives.
    quant_keys = [
        k for k in tensors
        if any(x in k.lower() for x in (
            "scale", "zero", "quant", "amax", "codebook", "s_rel", "s_channel"
        ))
    ]

    print(f"\nFILE: {path}")
    print(f"Tensors: {len(tensors)}")
    print("\nDTYPES:")
    for dtype, count in dtypes.most_common():
        print(f"  {dtype:12} {count}")

    metadata = header.get("__metadata__", {})
    if metadata:
        print("\nMETADATA:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

    if quant_keys:
        print("\nPOSSIBLE QUANTIZATION TENSORS:")
        for k in quant_keys[:30]:
            print(" ", k)

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} <file.safetensors>")
inspect_safetensors(sys.argv[1])