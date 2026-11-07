// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkKernelDensity,
  type GPUNetworkKernelDensityProps
} from '@luma-contributors/analysis/gpu-network';

export async function runExample(graph: GPUCommandGraph, input: GPUNetworkKernelDensityProps) {
  // #region example
  const computation = new GPUNetworkKernelDensity(input);
  graph.add(computation);
  // #endregion example
  return input.output;
}
