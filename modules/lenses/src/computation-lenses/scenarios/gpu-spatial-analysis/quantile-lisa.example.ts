// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUQuantileLISA} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Append the analysis with explicit resident inputs and caller-selected outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUQuantileLISA>[0]
) {
  // #region example
  graph.add(new GPUQuantileLISA(props));
  // #endregion example
}
