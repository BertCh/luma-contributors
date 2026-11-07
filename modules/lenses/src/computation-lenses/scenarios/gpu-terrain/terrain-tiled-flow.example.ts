// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUTerrainTiledFlow,
  type GPUTerrainTiledFlowProps
} from '@luma-contributors/graph/gpu-terrain';

/** Adds the public contributor with explicit caller-owned views. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPUTerrainTiledFlowProps
): Promise<void> {
  // #region example
  graph.add(new GPUTerrainTiledFlow(input));
  // #endregion example
}
