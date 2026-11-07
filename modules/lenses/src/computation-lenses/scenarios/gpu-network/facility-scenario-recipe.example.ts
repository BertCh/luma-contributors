// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addFacilityScenarioRecipe,
  type GPUFacilityScenarioRecipeProps
} from '@luma-contributors/analysis/gpu-network';

/** Declare this contributor using caller-owned graph views; the lens executes and captures it. */
export async function runExample(
  _device: Device,
  graph: GPUCommandGraph,
  properties: GPUFacilityScenarioRecipeProps
) {
  // #region example
  const result = addFacilityScenarioRecipe(graph, properties);
  return result.outputs;
  // #endregion example
}
