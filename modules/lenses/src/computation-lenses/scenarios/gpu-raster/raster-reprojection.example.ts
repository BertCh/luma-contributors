// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterReprojection,
  type GPURasterReprojectionProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterReprojection contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterReprojectionProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterReprojection(input);
  graph.add(contributor);
  // #endregion example
}
