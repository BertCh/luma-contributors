// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDorlingCartogram} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Append the analysis with explicit resident inputs and caller-selected outputs. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUDorlingCartogram>[0]
) {
  // #region example
  graph.add(new GPUDorlingCartogram(props));
  // #endregion example
}
