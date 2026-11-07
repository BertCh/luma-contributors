// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addInterpolationDiagnosticsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Runs the contributor shown in this lens with caller-supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addInterpolationDiagnosticsRecipe>[1]
) {
  // #region example
  return addInterpolationDiagnosticsRecipe(graph, props);
  // #endregion example
}
