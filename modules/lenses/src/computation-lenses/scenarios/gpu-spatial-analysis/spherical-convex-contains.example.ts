// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSphericalConvexContains,
  type GPUSphericalConvexContainsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Interior, exterior and closed-boundary queries on an octant; input and output views are caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUSphericalConvexContainsProps
) {
  // #region example
  graph.add(new GPUSphericalConvexContains(props));
  // Interpret every output together with its per-row status.
  return props.output;
  // #endregion example
}
