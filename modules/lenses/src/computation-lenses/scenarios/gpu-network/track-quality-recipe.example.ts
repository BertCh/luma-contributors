// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addTrackQualityRecipe} from '@luma-contributors/graph/gpu-network';

/** Runs the contributor shown in this lens with caller-supplied graph views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: Parameters<typeof addTrackQualityRecipe>[1]
) {
  // #region example
  return addTrackQualityRecipe(graph, props);
  // #endregion example
}
