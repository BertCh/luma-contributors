// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUS2PolygonCovering,
  type GPUS2PolygonCoveringProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adaptive covering of an octant; output views remain caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUS2PolygonCoveringProps
) {
  // #region example
  graph.add(new GPUS2PolygonCovering(props));
  return props.output;
  // #endregion example
}
