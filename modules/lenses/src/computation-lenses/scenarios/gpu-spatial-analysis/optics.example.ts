// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUOPTICS, type GPUOPTICSProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

export async function runExample(graph: GPUCommandGraph<undefined>, input: GPUOPTICSProps) {
  // #region example
  const computation = new GPUOPTICS(input);
  graph.add(computation);
  // #endregion example
  return input.output;
}
