// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPURasterNDWI, type GPURasterNDWIProps} from '@luma-contributors/graph/gpu-raster';

/** Adds the public contributor with caller-owned input and output views. */
export async function runExample(graph: GPUCommandGraph, input: GPURasterNDWIProps): Promise<void> {
  // #region example
  new GPURasterNDWI(input).addToGraph(graph);
  // #endregion example
}
