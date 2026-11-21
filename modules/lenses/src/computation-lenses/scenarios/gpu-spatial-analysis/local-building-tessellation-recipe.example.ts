// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addLocalBuildingTessellationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Runs the bounded local route with its certified sample-cell-union output layout. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addLocalBuildingTessellationRecipe>[1]
) {
  // #region example
  return addLocalBuildingTessellationRecipe(graph, props);
  // #endregion example
}
