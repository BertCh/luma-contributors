// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUFittingGradientSums} from '../../../src/analysis-core/kernels/gpu-fitting-gradient-sums';

/** Shared wiring for the shorter-run device regression and its symbolic construction pin. */
export function createShorterGradientProducer(
  graph: GPUCommandGraph,
  gradients: GraphDataView<'float32'>,
  sums: GraphDataView<'float32'>
): GPUFittingGradientSums {
  return new GPUFittingGradientSums({
    id: 'shorter',
    rowCount: 256,
    parameterCount: 1,
    gradients: graph.createDataView(gradients.buffer, {
      format: 'float32',
      length: 256,
      byteOffset: gradients.byteOffset
    }),
    output: {sums}
  });
}
