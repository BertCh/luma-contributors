// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUS2CapCovering,
  type GPUS2CapCoveringProps
} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Exterior root-face covering of a small north-pole cap; input and output views are caller-owned. */
export async function runExample(graph: GPUCommandGraph<undefined>, props: GPUS2CapCoveringProps) {
  // #region example
  graph.add(new GPUS2CapCovering(props));
  // Interpret every output together with its per-row status.
  return props.output;
  // #endregion example
}
