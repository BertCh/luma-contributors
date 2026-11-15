// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterContrastEqualization,
  type GPURasterContrastEqualizationProps
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the contributor with caller-owned inputs, outputs and per-frame controls. */
export async function runExample(
  graph: GPUCommandGraph,
  props: GPURasterContrastEqualizationProps
): Promise<void> {
  // #region example
  graph.add(new GPURasterContrastEqualization(props));
  // #endregion example
}
