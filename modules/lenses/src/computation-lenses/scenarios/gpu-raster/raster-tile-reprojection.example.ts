// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterTileReprojection,
  type GPURasterTileReprojectionProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public GPURasterTileReprojection contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterTileReprojectionProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterTileReprojection(input);
  graph.add(contributor);
  // #endregion example
}
