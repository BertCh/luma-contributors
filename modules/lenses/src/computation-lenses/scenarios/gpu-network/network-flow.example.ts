// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNetworkFlow, type GPUNetworkFlowProps} from '@luma-contributors/analysis/gpu-network';

/** Adds the documented contributor; the lens owns compilation, execution and capture. */
export async function runExample(graph: GPUCommandGraph, props: GPUNetworkFlowProps) {
  // #region example
  const computation = new GPUNetworkFlow(props);
  graph.add(computation);
  const {output} = computation.props;
  // #endregion example
  return output;
}
