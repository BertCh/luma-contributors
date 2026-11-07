// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUVariogramModelFit,
  type GPUVariogramModelFitProps
} from '@luma-contributors/graph/gpu-dataframe';

/** Bins, parameters, residual diagnostics and status remain GPU resident during the fit. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  properties: GPUVariogramModelFitProps
) {
  // #region example
  graph.add(new GPUVariogramModelFit(properties));
  return properties.output;
  // #endregion example
}
