"""Verify the upstream LTX patchifier CUDA-graph hot-path fix."""

from unittest.mock import patch

import torch

from comfy.ldm.lightricks.symmetric_patchifier import (
    SymmetricPatchifier,
    latent_to_pixel_coords,
)


patchifier = SymmetricPatchifier(2, start_end=True)
coords = patchifier.get_latent_coords(2, 4, 4, 1, torch.device("cpu"))
for axis, patch_size in enumerate(patchifier.patch_size):
    assert torch.equal(coords[:, axis, :, 1], coords[:, axis, :, 0] + patch_size)

real_tensor = torch.tensor
latent_coords = real_tensor([[[0, 1], [0, 2], [0, 3]]])
expected = real_tensor([[[0, 8], [0, 64], [0, 96]]])

with patch(
    "torch.tensor",
    side_effect=AssertionError("unexpected tensor-from-Python construction in hot path"),
):
    pixel_coords = latent_to_pixel_coords(latent_coords, (8, 32, 32))
    second_coords = patchifier.get_latent_coords(2, 4, 4, 1, torch.device("cpu"))

assert torch.equal(pixel_coords, expected)
assert torch.equal(second_coords, coords)
print("PASS: LTX patchifier hot path is capture-safe and preserves coordinate values")
