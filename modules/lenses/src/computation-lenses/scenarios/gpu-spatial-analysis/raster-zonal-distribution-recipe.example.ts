// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addRasterZonalDistributionRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Runs the contributor shown in this lens with caller-supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addRasterZonalDistributionRecipe>[1]
) {
  // #region example
  return addRasterZonalDistributionRecipe(graph, props);
  // #endregion example
}
