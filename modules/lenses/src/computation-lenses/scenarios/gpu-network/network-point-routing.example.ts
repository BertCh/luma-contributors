// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUNetworkPointRouting,
  type GPUNetworkPointRoutingProps
} from '@luma-contributors/graph/gpu-network';

/** Adds the documented contributor; the lens owns compilation, execution and capture. */
export async function runExample(graph: GPUCommandGraph, props: GPUNetworkPointRoutingProps) {
  // #region example
  const computation = new GPUNetworkPointRouting(props);
  graph.add(computation);
  const {output} = computation.props;
  // #endregion example
  return output;
}
