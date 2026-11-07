// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUNestedInterpolationValidation,
  type GPUNestedInterpolationValidationProps
} from '@luma-contributors/graph/gpu-spatial-analysis';
/** Fit, choose and evaluate using resident outer training/testing masks. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: GPUNestedInterpolationValidationProps
) {
  // #region example
  graph.add(new GPUNestedInterpolationValidation(props));
  return props.output;
  // #endregion example
}
