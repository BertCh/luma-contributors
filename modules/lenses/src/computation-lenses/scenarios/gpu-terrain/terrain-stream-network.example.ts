// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUTerrainStreamNetwork,
  type GPUTerrainStreamNetworkProps
} from '@luma-contributors/analysis/gpu-terrain';

/** Adds the public GPUTerrainStreamNetwork contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPUTerrainStreamNetworkProps
): Promise<void> {
  // #region example
  const contributor = new GPUTerrainStreamNetwork(input);
  graph.add(contributor);
  // #endregion example
}
