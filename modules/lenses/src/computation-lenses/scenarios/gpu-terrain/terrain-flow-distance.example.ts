// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUTerrainFlowDistance,
  type GPUTerrainFlowDistanceProps
} from '@luma-contributors/analysis/gpu-terrain';

/** Adds the public GPUTerrainFlowDistance contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPUTerrainFlowDistanceProps
): Promise<void> {
  // #region example
  const contributor = new GPUTerrainFlowDistance(input);
  graph.add(contributor);
  // #endregion example
}
