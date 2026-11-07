// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addSpatialRateRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Append the analysis with explicit resident inputs and caller-selected outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addSpatialRateRecipe>[1]
) {
  // #region example
  return addSpatialRateRecipe(graph, props);
  // #endregion example
}
