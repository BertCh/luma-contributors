// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCoregionalizationFit,
  type GPUCoregionalizationFitProps
} from '@luma-contributors/analysis/dataframe-analysis';

/** Fits PSD structure matrices and loading factors without reading bins back to the CPU. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  properties: GPUCoregionalizationFitProps
) {
  // #region example
  graph.add(new GPUCoregionalizationFit(properties));
  return properties.output;
  // #endregion example
}
