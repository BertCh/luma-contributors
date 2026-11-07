// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUSphericalPolygonContains,
  type GPUSphericalPolygonContainsProps
} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Nonconvex winding containment with explicit boundaries; output views remain caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUSphericalPolygonContainsProps
) {
  // #region example
  graph.add(new GPUSphericalPolygonContains(props));
  return props.output;
  // #endregion example
}
