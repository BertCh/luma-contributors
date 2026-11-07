// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterRankFilter,
  type GPURasterRankFilterProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterRankFilter contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterRankFilterProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterRankFilter(input);
  graph.add(contributor);
  // #endregion example
}
