# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Generate independent SciPy/skimage oracles; run with /usr/bin/python3 (without -I)."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, generate_binary_structure
from skimage.morphology import reconstruction

generator = np.random.default_rng(6147)
reconstruction_cases = []
for connectivity in [4, 8]:
    footprint = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    for method in ["dilation", "erosion"]:
        mask = generator.integers(0, 12, size=(6, 7)).astype(np.float32)
        marker = np.minimum(mask, generator.integers(0, 6, size=mask.shape))
        if method == "erosion":
            marker = 12 - marker
            mask = 12 - mask
        expected = reconstruction(marker, mask, method=method, footprint=footprint)
        reconstruction_cases.append({
            "width": 7, "height": 6, "method": method, "connectivity": connectivity,
            "marker": marker.ravel().tolist(), "mask": mask.ravel().tolist(),
            "expected": expected.ravel().tolist()
        })

fill_cases = []
inputs = [
    # Several holes, a passage from the boundary and diagonal-only contact.
    np.array([
        [0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.uint32),
    np.zeros((7, 9), dtype=np.uint32),
    np.ones((4, 5), dtype=np.uint32),
    (generator.random((9, 8)) > 0.6).astype(np.uint32) * 7
]
for connectivity in [4, 8]:
    for values in inputs:
        expected = binary_fill_holes(values, structure=generate_binary_structure(2, 1 if connectivity == 4 else 2))
        fill_cases.append({
            "width": values.shape[1], "height": values.shape[0], "connectivity": connectivity,
            "input": values.ravel().tolist(), "expected": expected.astype(np.uint32).ravel().tolist()
        })

Path(__file__).with_suffix(".json").write_text(json.dumps({
    "oracle": "scipy.ndimage.binary_fill_holes and skimage.morphology.reconstruction",
    "reconstruction": reconstruction_cases, "fillHoles": fill_cases
}, indent=2) + "\n")
