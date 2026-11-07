// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUGroupConcaveHull} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Runs the contributor shown in this lens with caller-supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUGroupConcaveHull>[0]
) {
  // #region example
  graph.add(new GPUGroupConcaveHull(props));
  // #endregion example
}
