// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterTileReprojection,
  type GPURasterTileReprojectionProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds tile reprojection; destroy the returned contributor after its compiled graph. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterTileReprojectionProps
): Promise<GPURasterTileReprojection> {
  // #region example
  const contributor = new GPURasterTileReprojection(input);
  graph.add(contributor);
  // #endregion example
  return contributor;
}
