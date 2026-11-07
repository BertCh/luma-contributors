// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUNaturalNeighbor,
  type GPUNaturalNeighborProps
} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Declares the public GPU contributor; the executable lens supplies and captures real views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  properties: GPUNaturalNeighborProps
) {
  // #region example
  graph.add(new GPUNaturalNeighbor(properties));
  return properties.output;
  // #endregion example
}
