// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCrossVariogram,
  type GPUCrossVariogramProps
} from '@luma-contributors/analysis/dataframe-analysis';

/** Declare this contributor using caller-owned graph views; the lens executes and captures it. */
export async function runExample<Parameters>(
  _device: Device,
  graph: GPUCommandGraph<Parameters>,
  properties: GPUCrossVariogramProps
) {
  // #region example
  const contributor = new GPUCrossVariogram(properties);
  graph.add(contributor);
  return properties.output;
  // #endregion example
}
