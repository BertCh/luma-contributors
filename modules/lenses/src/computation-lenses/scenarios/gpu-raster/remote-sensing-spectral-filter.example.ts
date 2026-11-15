// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterSpectralFilter,
  type GPURasterSpectralFilterProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public contributor with explicit caller-owned buffers and model options. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterSpectralFilterProps
): Promise<void> {
  // #region example
  graph.add(new GPURasterSpectralFilter(input));
  // #endregion example
}
