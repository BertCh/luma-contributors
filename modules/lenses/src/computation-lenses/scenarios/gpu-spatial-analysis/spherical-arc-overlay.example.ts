// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSphericalArcOverlay,
  type GPUSphericalArcOverlayProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Intrinsic spherical intersection with anchored membership; output views remain caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUSphericalArcOverlayProps
) {
  // #region example
  graph.add(new GPUSphericalArcOverlay(props));
  return props.output;
  // #endregion example
}
