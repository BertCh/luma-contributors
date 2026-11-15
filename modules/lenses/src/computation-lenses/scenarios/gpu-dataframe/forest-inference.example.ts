// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUForestInference,
  type GPUForestInferenceProps
} from '@luma-contributors/analysis/dataframe-analysis';

/** Adds the contributor with caller-owned inputs, outputs and per-frame controls. */
export async function runExample(
  graph: GPUCommandGraph,
  props: GPUForestInferenceProps
): Promise<void> {
  // #region example
  graph.add(new GPUForestInference(props));
  // #endregion example
}
