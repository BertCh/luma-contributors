// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterFillHoles,
  type GPURasterFillHolesProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public GPURasterFillHoles contributor with explicit caller-owned inputs and outputs. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterFillHolesProps
): Promise<void> {
  // #region example
  const contributor = new GPURasterFillHoles(input);
  graph.add(contributor);
  // #endregion example
}
