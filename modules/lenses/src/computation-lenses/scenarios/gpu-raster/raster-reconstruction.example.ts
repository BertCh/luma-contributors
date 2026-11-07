// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterReconstruction,
  type GPURasterReconstructionProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public GPURasterReconstruction contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterReconstructionProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterReconstruction(input);
  graph.add(contributor);
  // #endregion example
}
