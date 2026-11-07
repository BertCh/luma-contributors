// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Declares the public GPU contributor; the executable lens supplies and captures real views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  properties: GPULocalKrigingProps
) {
  // #region example
  graph.add(new GPULocalKriging(properties));
  return properties.output;
  // #endregion example
}
