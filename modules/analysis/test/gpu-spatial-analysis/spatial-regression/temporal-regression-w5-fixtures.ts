// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUGeographicallyTemporallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {GPUMultiscaleGeographicallyTemporallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-temporally-weighted-regression';

export type FinalTemporalScenario =
  | 'mixed-sign'
  | 'rounded'
  | 'exact'
  | 'negligible'
  | 'component-outliers'
  | 'orthogonal-rounded';

export function createFinalTemporalCase(
  graph: GPUCommandGraph,
  port: <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => GraphDataView<Format>,
  scenario: FinalTemporalScenario,
  route: 'dense' | 'indexed' | 'search',
  multiscale = false
) {
  const mixed = scenario === 'mixed-sign';
  const predictors =
    scenario === 'negligible'
      ? [-1, 1, -1, 1, 0]
      : scenario === 'component-outliers'
        ? [0, 0, -1, 1, -1, 1, -1, 1]
        : scenario === 'orthogonal-rounded'
          ? [-4, -3, -2, -1, 1, 2, 3, 4]
          : mixed
            ? [-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]
            : [0, 1, 2, 3, 9, 10, 11, 12];
  const response =
    scenario === 'negligible'
      ? [1, 3, 1, 3, 1e8]
      : scenario === 'component-outliers'
        ? [33554432, -33554432, 1, 1, 1, 1, 1, 1]
        : scenario === 'orthogonal-rounded'
          ? predictors.map(value => 1e6 + (value > 0 ? 1 / 16 : 0))
          : mixed
            ? [1e7, -1e7, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
            : predictors.map((_, row) => 1e6 + (scenario === 'rounded' && row >= 4 ? 1 / 16 : 0));
  const rows = response.length;
  const outputs = {
    coefficients: port('float32', Array(rows * 2).fill(999)),
    fitted: port('float32', Array(rows).fill(999)),
    residuals: port('float32', Array(rows).fill(999)),
    localStatus: port('uint32', Array(rows).fill(999)),
    hatDiagonal: port('float32', Array(rows).fill(999)),
    summary: port('float32', Array(12).fill(999)),
    ...(multiscale ? {scores: port('float32', Array(2).fill(999))} : {})
  };
  const common = {
    id: `w5-${scenario}-${route}`,
    positions: port(
      'float32x2',
      scenario === 'negligible' ? [0, 0, 0, 0, 0, 0, 0, 0, 8, 0] : Array(rows * 2).fill(0)
    ),
    times: port('float32', Array(rows).fill(0)),
    temporalContext: {
      unit: 'second' as const,
      epoch: 0n,
      encoding: 'float32-relative' as const,
      axis: {kind: 'irregular' as const}
    },
    predictors: port('float32', predictors),
    predictorCount: 1,
    response: port('float32', response),
    kernel: 'gaussian' as const
  };
  const bandwidth = scenario === 'negligible' ? 1 : 30;
  const contributor = multiscale
    ? new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
        ...common,
        iterationCount: 2,
        parameters: port('float32', [bandwidth, 0, bandwidth, 0, bandwidth, 0]),
        output: {...outputs, scores: outputs.scores!}
      })
    : new GPUGeographicallyTemporallyWeightedRegression({
        ...common,
        searchStructure: route === 'indexed' ? 'grid' : 'brute-force',
        gaussianCutoff: route === 'indexed' ? (scenario === 'negligible' ? 9 : 4) : undefined,
        bandwidthSearch:
          route === 'search'
            ? {parameters: port('float32', [1, 2, 0, 0, 0.0001, 0.0001]), maximumIterations: 1}
            : undefined,
        parameters: port('float32', [bandwidth, 0, 0, 0]),
        output: outputs
      });
  return {
    nodes: contributor.getCommandNodes(graph),
    outputs,
    expected: {} as Record<string, number[]>
  };
}
