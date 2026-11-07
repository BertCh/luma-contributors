// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addDasymetricRedistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds the public recipe; the returned views remain owned by this graph. */
export async function runExample(
  graph: GPUCommandGraph,
  input: Parameters<typeof addDasymetricRedistributionRecipe>[1]
) {
  // #region example
  return addDasymetricRedistributionRecipe(graph, input);
  // #endregion example
}
