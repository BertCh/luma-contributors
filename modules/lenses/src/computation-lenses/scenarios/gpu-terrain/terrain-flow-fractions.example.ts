// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUTerrainFlowFractions,
  type GPUTerrainFlowFractionsProps
} from '@luma-contributors/graph/gpu-terrain';

/** Adds the public contributor with explicit caller-owned views. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPUTerrainFlowFractionsProps
): Promise<void> {
  // #region example
  graph.add(new GPUTerrainFlowFractions(input));
  // #endregion example
}
