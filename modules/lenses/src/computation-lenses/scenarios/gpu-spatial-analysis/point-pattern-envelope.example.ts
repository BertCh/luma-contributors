// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUPointPatternEnvelope,
  type GPUPointPatternEnvelopeProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: GPUPointPatternEnvelopeProps
) {
  // #region example
  const computation = new GPUPointPatternEnvelope(input);
  graph.add(computation);
  // #endregion example
  return input.output;
}
