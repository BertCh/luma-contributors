// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterNDWI, type GPURasterNDWIProps} from '@luma-contributors/analysis/raster-analysis';

/** Adds the public contributor with caller-owned input and output views. */
export async function runExample(graph: GPUCommandGraph, input: GPURasterNDWIProps): Promise<void> {
  // #region example
  graph.add(new GPURasterNDWI(input));
  // #endregion example
}
