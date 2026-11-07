// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPURasterSkeleton, type GPURasterSkeletonProps} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterSkeleton contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterSkeletonProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterSkeleton(input);
  graph.add(contributor);
  // #endregion example
}
