// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUGaussianRandomField,
  type GPUGaussianRandomFieldProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds the contributor with caller-owned inputs, outputs and per-frame controls. */
export async function runExample(
  graph: GPUCommandGraph,
  props: GPUGaussianRandomFieldProps
): Promise<void> {
  // #region example
  graph.add(new GPUGaussianRandomField(props));
  // #endregion example
}
