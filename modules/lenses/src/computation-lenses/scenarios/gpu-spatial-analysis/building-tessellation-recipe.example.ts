// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addBuildingTessellationRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Runs the actual recipe shown in this lens with supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addBuildingTessellationRecipe>[1]
) {
  // #region example
  return addBuildingTessellationRecipe(graph, props);
  // #endregion example
}
