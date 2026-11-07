// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterFractionalZonalStatistics,
  type GPURasterFractionalZonalStatisticsProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterFractionalZonalStatistics contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterFractionalZonalStatisticsProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterFractionalZonalStatistics(input);
  graph.add(contributor);
  // #endregion example
}
