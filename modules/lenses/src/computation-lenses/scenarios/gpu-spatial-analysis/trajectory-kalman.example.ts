// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUTrajectoryKalman,
  type GPUTrajectoryKalmanProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds the documented contributor; the lens owns compilation, execution and capture. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUTrajectoryKalmanProps
) {
  // #region example
  const computation = new GPUTrajectoryKalman(props);
  graph.add(computation);
  const {output} = computation.props;
  // #endregion example
  return output;
}
