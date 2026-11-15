// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPopulationSupportAlignment} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Aligns explicit support IDs into caller-owned output views. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUPopulationSupportAlignment>[0]
) {
  // #region example
  graph.add(new GPUPopulationSupportAlignment(props));
  return props.output;
  // #endregion example
}
