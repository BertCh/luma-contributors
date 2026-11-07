// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUSpatialBalancedSample,
  type GPUSpatialBalancedSampleProps
} from '@luma-contributors/graph/gpu-spatial-analysis';

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: GPUSpatialBalancedSampleProps
) {
  // #region example
  const computation = new GPUSpatialBalancedSample(input);
  graph.add(computation);
  // #endregion example
  return input.output;
}
