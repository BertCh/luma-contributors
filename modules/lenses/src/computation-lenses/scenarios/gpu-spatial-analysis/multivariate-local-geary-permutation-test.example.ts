// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUMultivariateLocalGearyPermutationTest,
  type GPUMultivariateLocalGearyPermutationTestProps
} from '@luma-contributors/graph/gpu-spatial-analysis';

/** Add the actual contributor using caller-owned input and output views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUMultivariateLocalGearyPermutationTestProps
) {
  // #region example
  const contributor = new GPUMultivariateLocalGearyPermutationTest(props);
  graph.add(contributor);
  return props.output;
  // #endregion example
}
