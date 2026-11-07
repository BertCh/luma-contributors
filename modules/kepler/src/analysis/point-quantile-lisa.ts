// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {matchesKeplerPointWeightSpatialContext} from './point-weight-coordinates';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {parseKeplerPointWeightsSettings} from '../recipes/analysis-recipe';
import {
  getKeplerPointWeightsMaximumNeighbors,
  getKeplerPointWeightsCapacity,
  type KeplerPointWeightsBinding
} from './point-weights';
import {
  addWeightedQuantileLISA,
  type WeightedQuantileLISAOutputs,
  type WeightedQuantileLISAProps
} from './weighted-quantile-lisa';
import {isGPUChunkedSpatialWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';

export type KeplerPointQuantileLISAOutputs = WeightedQuantileLISAOutputs;
export type KeplerPointQuantileLISAProps = Omit<WeightedQuantileLISAProps, 'topology'> & {
  pointWeights: KeplerPointWeightsBinding;
};

/** Selected-quantile joins on fixed source-wide point neighbors, with legacy directed inference. */
export function addKeplerPointQuantileLISA<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointQuantileLISAProps
) {
  const {pointWeights} = props;
  const rowCount = props.values.length;
  const weightsSettings = parseKeplerPointWeightsSettings(pointWeights.settings);
  getKeplerPointWeightsCapacity(weightsSettings, rowCount);
  const information = pointWeights.weights.information;
  const context = pointWeights.weights.spatialContext;
  if (
    information?.rowCount !== rowCount ||
    information.transform !== 'binary' ||
    // Streamed (chunked) weights keep one self slot per row; the kernels skip self slots.
    information.hasSelfWeights !== isGPUChunkedSpatialWeights(pointWeights.weights) ||
    information.provenance?.source !== 'GPUNeighborSearch' ||
    information.provenance.criterion !== weightsSettings.kind ||
    (weightsSettings.kind === 'knn' && information.provenance.k !== weightsSettings.k) ||
    (weightsSettings.kind === 'distance-band' &&
      // Chunks are sized per row range, so only a whole CSR has the recipe's slot capacity.
      !isGPUChunkedSpatialWeights(pointWeights.weights) &&
      pointWeights.weights.neighbors.length !== weightsSettings.neighborCapacity) ||
    !matchesKeplerPointWeightSpatialContext(weightsSettings, context)
  )
    throw new Error('Quantile LISA requires binary point weights without self edges');
  const result = addWeightedQuantileLISA(graph, {
    ...props,
    id: props.id ?? 'kepler-point-quantile-lisa',
    topology: {
      source: pointWeights.source,
      weights: pointWeights.weights,
      mask: pointWeights.coordinateMask,
      status: pointWeights.status,
      maximumNeighbors: getKeplerPointWeightsMaximumNeighbors(weightsSettings)
    }
  });
  return {...result, pointWeights};
}
