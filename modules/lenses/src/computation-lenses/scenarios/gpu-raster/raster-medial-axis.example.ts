// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterMedialAxis,
  type GPURasterMedialAxisProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterMedialAxis contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterMedialAxisProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterMedialAxis(input);
  graph.add(contributor);
  // #endregion example
}
