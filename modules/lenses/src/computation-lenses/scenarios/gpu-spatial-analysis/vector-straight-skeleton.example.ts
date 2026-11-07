// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUVectorStraightSkeleton} from '@luma-contributors/graph/gpu-spatial-analysis';

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUVectorStraightSkeleton>[0]
) {
  // #region example
  graph.add(new GPUVectorStraightSkeleton(props));
  // #endregion example
}
