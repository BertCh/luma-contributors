// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPURasterSAVI, type GPURasterSAVIProps} from '@luma-contributors/graph/gpu-raster';

/** Adds the public contributor with caller-owned input and output views. */
export async function runExample(graph: GPUCommandGraph, input: GPURasterSAVIProps): Promise<void> {
  // #region example
  new GPURasterSAVI(input).addToGraph(graph);
  // #endregion example
}
