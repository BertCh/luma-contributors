// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSphericalPolygonOverlay,
  type GPUSphericalPolygonOverlayProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Chart polygon intersection with captured topology; output views remain caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUSphericalPolygonOverlayProps
) {
  // #region example
  graph.add(new GPUSphericalPolygonOverlay(props));
  return props.output;
  // #endregion example
}
