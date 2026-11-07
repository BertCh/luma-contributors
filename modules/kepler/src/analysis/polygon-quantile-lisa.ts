// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  validateKeplerPolygonWeightsBinding,
  type KeplerPolygonWeightsBinding
} from './polygon-weights';
import {addWeightedQuantileLISA, type WeightedQuantileLISAProps} from './weighted-quantile-lisa';

export type KeplerPolygonQuantileLISAProps = Omit<WeightedQuantileLISAProps, 'topology'> & {
  polygonWeights: KeplerPolygonWeightsBinding;
};

/**
 * Selected-quantile joins on exact source-XY polygon contiguity. Geometry validity is applied
 * before type-5 quantiles; valid islands affect boundaries but remain excluded from inference.
 * P-values retain the existing anti-conservative legacy directed rule, without correction.
 */
export function addKeplerPolygonQuantileLISA<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPolygonQuantileLISAProps
) {
  const {polygonWeights} = props;
  validateKeplerPolygonWeightsBinding(graph, polygonWeights, props.source, props.values.length);
  const result = addWeightedQuantileLISA(graph, {
    ...props,
    id: props.id ?? 'kepler-polygon-quantile-lisa',
    topology: {
      source: polygonWeights.source,
      weights: polygonWeights.weights,
      mask: polygonWeights.geometryMask,
      status: polygonWeights.status,
      maximumNeighbors: 64
    }
  });
  return {...result, polygonWeights};
}
