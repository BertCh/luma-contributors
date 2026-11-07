// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUS2CellMeasures,
  type GPUS2CellMeasuresProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Unit-sphere area and perimeter of the six S2 root faces; input and output views are caller-owned. */
export async function runExample(graph: GPUCommandGraph<undefined>, props: GPUS2CellMeasuresProps) {
  // #region example
  graph.add(new GPUS2CellMeasures(props));
  // Interpret every output together with its per-row status.
  return props.output;
  // #endregion example
}
