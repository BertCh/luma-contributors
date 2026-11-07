// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUNetworkStructure,
  type GPUNetworkStructureProps
} from '@luma-contributors/graph/gpu-network';

/** Declare this contributor using caller-owned graph views; the lens executes and captures it. */
export async function runExample(
  _device: Device,
  graph: GPUCommandGraph,
  properties: GPUNetworkStructureProps
) {
  // #region example
  const contributor = new GPUNetworkStructure(properties);
  graph.add(contributor);
  return properties.output;
  // #endregion example
}
