// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPULineLoopRemoval} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Add the real contributor with explicit caller-owned inputs, capacities and outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPULineLoopRemoval>[0]
) {
  // #region example
  graph.add(new GPULineLoopRemoval(props));
  // #endregion example
}
