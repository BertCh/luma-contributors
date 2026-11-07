// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addSparseResistanceConnectivityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Executes the graph-native API with caller-owned inputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addSparseResistanceConnectivityRecipe>[1]
) {
  // #region example
  return addSparseResistanceConnectivityRecipe(graph, props);
  // #endregion example
}
