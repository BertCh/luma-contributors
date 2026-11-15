// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkStronglyConnectedComponents,
  type GPUNetworkStronglyConnectedComponentsProps
} from '@luma-contributors/analysis/gpu-network';

/** Declare this contributor using caller-owned graph views; the lens executes and captures it. */
export async function runExample(
  _device: Device,
  graph: GPUCommandGraph,
  properties: GPUNetworkStronglyConnectedComponentsProps
) {
  // #region example
  const contributor = new GPUNetworkStronglyConnectedComponents(properties);
  graph.add(contributor);
  return properties.output;
  // #endregion example
}
