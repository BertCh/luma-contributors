// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUS2CellHierarchy,
  type GPUS2CellHierarchyProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Exact children of the six S2 root faces; input and output views are caller-owned. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUS2CellHierarchyProps
) {
  // #region example
  graph.add(new GPUS2CellHierarchy(props));
  // Interpret every output together with its per-row status.
  return props.output;
  // #endregion example
}
