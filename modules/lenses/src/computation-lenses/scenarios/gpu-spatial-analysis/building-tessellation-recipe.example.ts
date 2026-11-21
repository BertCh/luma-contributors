// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addBuildingTessellationReferenceRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Runs the actual recipe shown in this lens with supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addBuildingTessellationReferenceRecipe>[1]
) {
  // #region example
  return addBuildingTessellationReferenceRecipe(graph, props);
  // #endregion example
}
