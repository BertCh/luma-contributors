// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUConstrainedDelaunayTriangulation} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Runs the contributor shown in this lens with caller-supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUConstrainedDelaunayTriangulation>[0]
) {
  // #region example
  graph.add(new GPUConstrainedDelaunayTriangulation(props));
  // #endregion example
}
