// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterFractionalCategoricalStatistics,
  type GPURasterFractionalCategoricalStatisticsProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public contributor with caller-owned input and output views. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterFractionalCategoricalStatisticsProps
): Promise<void> {
  // #region example
  graph.add(new GPURasterFractionalCategoricalStatistics(input));
  // #endregion example
}
