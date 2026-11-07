// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterWatershedSegmentation,
  type GPURasterWatershedSegmentationProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public GPURasterWatershedSegmentation contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterWatershedSegmentationProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterWatershedSegmentation(input);
  graph.add(contributor);
  // #endregion example
}
