// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addBuildingAdjacencyRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Runs the actual recipe shown in this lens with supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addBuildingAdjacencyRecipe>[1]
) {
  // #region example
  return addBuildingAdjacencyRecipe(graph, props);
  // #endregion example
}
