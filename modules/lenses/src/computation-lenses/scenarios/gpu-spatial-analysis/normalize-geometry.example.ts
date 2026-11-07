// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNormalizeGeometry} from '@luma-contributors/analysis/gpu-spatial-analysis';

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUNormalizeGeometry>[0]
) {
  // #region example
  graph.add(new GPUNormalizeGeometry(props));
  // #endregion example
}
