// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterSpectralUnmixing,
  type GPURasterSpectralUnmixingProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public contributor with explicit caller-owned buffers and model options. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterSpectralUnmixingProps
): Promise<void> {
  // #region example
  graph.add(new GPURasterSpectralUnmixing(input));
  // #endregion example
}
