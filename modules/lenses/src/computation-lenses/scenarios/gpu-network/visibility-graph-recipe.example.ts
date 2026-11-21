// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addVisibilityGraphRecipe,
  type GPUNetworkVisibilityGraphCSRProps
} from '@luma-contributors/analysis/gpu-network';

/** Build CSR visibility and pivot centrality using caller-owned planar geometry. */
// #region example
export async function runExample(graph: GPUCommandGraph, props: GPUNetworkVisibilityGraphCSRProps) {
  const visibility = addVisibilityGraphRecipe(graph, props);
  return visibility;
}
// #endregion example
