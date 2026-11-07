// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterFractionalCoverage,
  type GPURasterFractionalCoverageProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public GPURasterFractionalCoverage contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterFractionalCoverageProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterFractionalCoverage(input);
  graph.add(contributor);
  // #endregion example
}
