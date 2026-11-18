// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const reachabilityInputs = {
  topology: {
    offsets: Uint32Array.of(0, 0, 1, 1),
    neighbors: Uint32Array.of(2),
    weights: Float32Array.of(2)
  },
  startNodes: Uint32Array.of(1),
  costLimit: Float32Array.of(10),
  costBreaks: Float32Array.of(0, 2, 5, 10)
};
