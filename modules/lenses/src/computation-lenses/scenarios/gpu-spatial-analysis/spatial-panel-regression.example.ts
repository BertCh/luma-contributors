// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialPanelRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Runs the contributor shown in this lens with caller-supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUSpatialPanelRegression>[0]
) {
  // #region example
  graph.add(new GPUSpatialPanelRegression(props));
  // #endregion example
}
