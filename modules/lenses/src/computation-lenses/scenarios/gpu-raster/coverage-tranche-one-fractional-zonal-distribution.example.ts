// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterFractionalZonalDistribution,
  type GPURasterFractionalZonalDistributionProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public contributor with caller-owned input and output views. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterFractionalZonalDistributionProps
): Promise<void> {
  // #region example
  graph.add(new GPURasterFractionalZonalDistribution(input));
  // #endregion example
}
