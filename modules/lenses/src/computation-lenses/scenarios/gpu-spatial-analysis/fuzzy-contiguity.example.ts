// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUFuzzyContiguity,
  type GPUFuzzyContiguityProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Add the actual contributor using caller-owned input and output views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUFuzzyContiguityProps
) {
  // #region example
  const contributor = new GPUFuzzyContiguity(props);
  graph.add(contributor);
  return props.output;
  // #endregion example
}
