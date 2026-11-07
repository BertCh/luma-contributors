// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUNetworkInhomogeneousK,
  type GPUNetworkInhomogeneousKProps
} from '@luma-contributors/graph/gpu-network';

export async function runExample(graph: GPUCommandGraph, input: GPUNetworkInhomogeneousKProps) {
  // #region example
  const computation = new GPUNetworkInhomogeneousK(input);
  graph.add(computation);
  // #endregion example
  return input.output;
}
