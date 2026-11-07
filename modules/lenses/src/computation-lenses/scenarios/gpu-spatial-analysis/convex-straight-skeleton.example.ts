// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUConvexStraightSkeleton} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Add the real contributor with explicit caller-owned inputs, capacities and outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUConvexStraightSkeleton>[0]
) {
  // #region example
  graph.add(new GPUConvexStraightSkeleton(props));
  // #endregion example
}
