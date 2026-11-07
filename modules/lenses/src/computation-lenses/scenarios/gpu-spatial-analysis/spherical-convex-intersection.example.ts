// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUSphericalConvexIntersection,
  type GPUSphericalConvexIntersectionProps
} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Identical octants with intersection area and overlap fractions; input and output views are caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUSphericalConvexIntersectionProps
) {
  // #region example
  graph.add(new GPUSphericalConvexIntersection(props));
  // Interpret every output together with its per-row status.
  return props.output;
  // #endregion example
}
