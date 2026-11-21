// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {
  GPULocalMoran,
  type GPULocalMoranProps
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import type {WeightsRig} from './spatial-weights-harness';

/** The same island-to-Moran construction is checked on Node and submitted by the GPU spec. */
export function createIslandMoran(
  rig: WeightsRig,
  weights: GPULocalMoranProps['weights'],
  zScores: GraphDataView<'float32'>,
  quadrants: GraphDataView<'uint32'>
) {
  const parameters = getGPUSpatialAutocorrelationParameterValues();
  return new GPULocalMoran({
    weights,
    values: rig.input(Float32Array.of(1, 2, 4), 'float32', 3),
    parameters: rig.input(parameters, 'float32', parameters.length),
    output: {zScores, quadrants}
  });
}

/** Rebind the published port in a consumer graph without copying its data or inventing status. */
export function importPublishedWeights(
  graph: GPUCommandGraph,
  weights: GPUSpatialWeights
): GPUSpatialWeights {
  let serial = 0;
  const reimport = <Format extends 'float32' | 'uint32'>(view: GraphDataView<Format>) => {
    if (!view.buffer.defaultBuffer)
      throw new Error('published fixture views must have physical buffers');
    const imported = importGraphBuffer(
      graph,
      `published-${serial++}`,
      getCoreDefaultBuffer(view.buffer.defaultBuffer),
      view.format,
      view.length
    );
    return graph.createDataView(imported.buffer, {
      format: view.format,
      length: view.length,
      byteOffset: view.byteOffset
    });
  };
  return {
    ...weights,
    offsets: reimport(weights.offsets),
    neighbors: reimport(weights.neighbors),
    weights: reimport(weights.weights),
    distances: weights.distances && reimport(weights.distances),
    status:
      weights.status &&
      Object.fromEntries(
        Object.entries(weights.status).map(([name, view]) => [name, view && reimport(view)])
      )
  };
}
