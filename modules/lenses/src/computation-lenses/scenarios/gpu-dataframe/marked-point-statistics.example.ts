// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUMarkedPointStatistics,
  type GPUMarkedPointStatisticsProps
} from '@luma-contributors/graph/gpu-dataframe';

/** Declare this contributor using caller-owned graph views; the lens executes and captures it. */
export async function runExample<Parameters>(
  _device: Device,
  graph: GPUCommandGraph<Parameters>,
  properties: GPUMarkedPointStatisticsProps
) {
  // #region example
  const contributor = new GPUMarkedPointStatistics(properties);
  graph.add(contributor);
  return properties.output;
  // #endregion example
}
