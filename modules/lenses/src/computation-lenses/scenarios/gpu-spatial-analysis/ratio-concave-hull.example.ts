// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPURatioConcaveHull} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Add the real contributor with explicit caller-owned inputs, capacities and outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPURatioConcaveHull>[0]
) {
  // #region example
  graph.add(new GPURatioConcaveHull(props));
  // #endregion example
}
