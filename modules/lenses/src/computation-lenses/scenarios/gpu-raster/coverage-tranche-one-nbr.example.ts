// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPURasterNBR, type GPURasterNBRProps} from '@luma-contributors/graph/gpu-raster';

/** Adds the public contributor with caller-owned input and output views. */
export async function runExample(graph: GPUCommandGraph, input: GPURasterNBRProps): Promise<void> {
  // #region example
  new GPURasterNBR(input).addToGraph(graph);
  // #endregion example
}
