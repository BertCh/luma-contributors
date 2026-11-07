// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPolygonPrecisionReduction} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Add the real contributor with explicit caller-owned inputs, capacities and outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUPolygonPrecisionReduction>[0]
) {
  // #region example
  graph.add(new GPUPolygonPrecisionReduction(props));
  // #endregion example
}
