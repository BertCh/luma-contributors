// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addSegregationCounterfactualRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Executes the graph-native API with caller-owned inputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addSegregationCounterfactualRecipe>[1]
) {
  // #region example
  return addSegregationCounterfactualRecipe(graph, props);
  // #endregion example
}
