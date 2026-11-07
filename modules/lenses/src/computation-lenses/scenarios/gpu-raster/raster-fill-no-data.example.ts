// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterFillNoData,
  type GPURasterFillNoDataProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterFillNoData contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterFillNoDataProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterFillNoData(input);
  graph.add(contributor);
  // #endregion example
}
