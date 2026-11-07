// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterWalkingCost,
  type GPURasterWalkingCostProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public GPURasterWalkingCost contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterWalkingCostProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterWalkingCost(input);
  graph.add(contributor);
  // #endregion example
}
