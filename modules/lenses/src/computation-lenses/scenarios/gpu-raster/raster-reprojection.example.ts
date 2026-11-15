// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterReprojection,
  type GPURasterReprojectionProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds reprojection; destroy the returned contributor after its compiled graph. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterReprojectionProps
): Promise<GPURasterReprojection> {
  // #region example
  const contributor = new GPURasterReprojection(input);
  graph.add(contributor);
  // #endregion example
  return contributor;
}
