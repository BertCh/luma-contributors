// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUOffsetCurveComponents} from '@luma-contributors/graph/gpu-spatial-analysis';

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  props: ConstructorParameters<typeof GPUOffsetCurveComponents>[0]
) {
  // #region example
  graph.add(new GPUOffsetCurveComponents(props));
  // #endregion example
}
