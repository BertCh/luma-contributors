// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUTrajectoryRestriction,
  type GPUTrajectoryRestrictionProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds the documented contributor; the lens owns compilation, execution and capture. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUTrajectoryRestrictionProps
) {
  // #region example
  const computation = new GPUTrajectoryRestriction(props);
  graph.add(computation);
  const {output} = computation.props;
  // #endregion example
  return output;
}
