// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  addSurfacePreparationRecipe,
  type GPUSurfacePreparationRecipeProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds bounded fill with optional rank filtering and stage-specific provenance outputs. */
export function addMissingSurfacePreparation(
  graph: GPUCommandGraph<void>,
  props: GPUSurfacePreparationRecipeProps
): ReturnType<typeof addSurfacePreparationRecipe<void>> {
  return addSurfacePreparationRecipe(graph, props);
}

export type MissingSurfacePreparationViews = {
  values: GraphDataView<'float32'>;
  validity: GraphDataView<'uint32'>;
  originalValidity: GraphDataView<'uint32'>;
  filled: GraphDataView<'uint32'>;
  filtered: GraphDataView<'uint32'>;
  unresolvedCount: GraphDataView<'uint32'>;
};
