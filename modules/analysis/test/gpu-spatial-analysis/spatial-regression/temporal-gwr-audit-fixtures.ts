// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUGeographicallyTemporallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {GPUMultiscaleGeographicallyTemporallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-temporally-weighted-regression';

/** QR and simultaneous component block solves from temporal-gwr-audit.oracle.py. */
export const TEMPORAL_AUDIT_MODELS = [
  {
    kernel: 'gaussian',
    adaptive: false,
    coefficients: [
      [2.2121678244787253, 0.3634965265638243],
      [2.2878321755212747, 0.13650347343617608]
    ],
    fitted: [1.848671297914901, 2.5756643510425494, 2.151328702085099, 2.4243356489574506],
    hat: 0.19188811701418318,
    summary: [
      25.401822289284222,
      2.302657404170198,
      53.351046949117844,
      0.1008204499368417,
      12,
      NaN,
      0,
      2.6194621916531955,
      NaN,
      NaN,
      -21.526747990469936,
      12
    ],
    multiscale: {
      coefficients: [
        [2.2121678244787253, 0.3637000804237005],
        [2.2878321755212747, 0.1362999195762995]
      ],
      fitted: [1.8484677440550248, 2.5758679049024256, 2.151532255944975, 2.4241320950975744],
      scores: [0.009323308968429587, 0]
    }
  },
  {
    kernel: 'gaussian',
    adaptive: true,
    coefficients: [
      [2.188770346149256, 0.43368896155222997],
      [2.3112296538507424, 0.0663110384477704]
    ],
    fitted: [1.755081384597026, 2.6224593077014857, 2.244918615402972, 2.377540692298513],
    hat: 0.2074864359004955,
    summary: [
      24.276109230237225,
      2.489837230805946,
      53.661924769830435,
      0.1406686998146115,
      12,
      NaN,
      0,
      2.552649183763079,
      NaN,
      NaN,
      -21.254778744601726,
      12
    ],
    multiscale: {
      coefficients: [
        [2.1887703461492567, 0.4336889615522298],
        [2.3112296538507433, 0.06631103844777021]
      ],
      fitted: [1.7550813845970268, 2.6224593077014866, 2.244918615402973, 2.3775406922985134],
      scores: [0, 0]
    }
  },
  {
    kernel: 'bisquare',
    adaptive: false,
    coefficients: [
      [2.0660098949743952, 0.8019703150768162],
      [2.4339901050256056, -0.3019703150768163]
    ],
    fitted: [1.2640395798975792, 2.8679802100512113, 2.735960420102422, 2.132019789948789],
    hat: 0.289326736683737,
    summary: [
      20.52287674814367,
      3.4719208402048443,
      56.93481351300934,
      0.2735264867913745,
      12,
      NaN,
      0,
      2.406506361350031,
      NaN,
      NaN,
      -20.247063717015237,
      12
    ],
    multiscale: {
      coefficients: [
        [2.066009894974395, 0.8029490616621985],
        [2.4339901050256056, -0.3029490616621985]
      ],
      fitted: [1.2630608333121964, 2.868958956636593, 2.736939166687804, 2.1310410433634073],
      scores: [0.04628799843023003, 0]
    }
  },
  {
    kernel: 'bisquare',
    adaptive: true,
    coefficients: [
      [2, 1],
      [2.5, -0.5]
    ],
    fitted: [1, 3, 3, 2],
    hat: 1 / 3,
    summary: [
      20,
      4,
      60.184432282102854,
      0.2920353982300885,
      12,
      NaN,
      0,
      2.5,
      NaN,
      NaN,
      -20.092216141052013,
      12
    ],
    multiscale: {
      coefficients: [
        [2, 1],
        [2.5, -0.5]
      ],
      fitted: [1, 3, 3, 2],
      scores: [0, 0]
    }
  }
] as const;
export type TemporalAuditModel = (typeof TEMPORAL_AUDIT_MODELS)[number];
export type TemporalAuditOptions = {
  straightLine?: boolean;
  rescaled?: boolean;
  multiscale?: boolean;
  indexed?: boolean;
  search?: boolean;
};

export function createTemporalAuditCase(
  graph: GPUCommandGraph,
  port: <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => GraphDataView<Format>,
  model: TemporalAuditModel,
  options: TemporalAuditOptions = {}
) {
  const locationScale = options.rescaled ? 4 : 1;
  const predictorScale = options.rescaled ? 2 : 1;
  const timeScale = options.rescaled ? 4 : 1;
  const bandwidth = model.adaptive ? 8 : 5 * locationScale;
  const parameters =
    options.multiscale && !options.search
      ? [
          model.adaptive ? 10 : 6 * locationScale,
          0.5 * timeScale,
          bandwidth,
          1.25 * timeScale,
          model.adaptive ? 10 : 6 * locationScale,
          2 * timeScale
        ]
      : [
          bandwidth,
          1.25 * timeScale,
          ...(options.multiscale
            ? [bandwidth, 1.25 * timeScale, bandwidth, 1.25 * timeScale]
            : [0, 0])
        ];
  const response = [0, 1, 2, 1, 3, 5, 2, 3, 4, 0, 2, 4, NaN, NaN];
  if (options.straightLine) {
    for (let row = 0; row < 12; row++) response[row] = 1000000 + (row % 6 < 3 ? -1 : 1);
  }
  const reference = options.multiscale && !options.search ? model.multiscale : model;
  const expected: Record<string, number[]> = {
    coefficients: [],
    fitted: [],
    residuals: [],
    localStatus: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
  };
  for (let row = 0; row < 14; row++) {
    const coefficients = options.straightLine
      ? [1000000, 1]
      : reference.coefficients[row < 6 ? 0 : 1];
    expected['coefficients'].push(
      ...(row < 12 ? [coefficients[0], coefficients[1] / predictorScale] : [NaN, NaN])
    );
    expected['fitted'].push(
      row < 12
        ? options.straightLine
          ? response[row]
          : reference.fitted[Math.floor(row / 3)]
        : NaN
    );
    expected['residuals'].push(response[row] - expected['fitted'][row]);
  }
  if (options.multiscale) {
    expected['scores'] = options.search ? [0, 0] : [...model.multiscale.scores];
    expected['selectedParameters'] = parameters;
  } else {
    expected['hatDiagonal'] = [...Array<number>(12).fill(model.hat), NaN, NaN];
    expected['summary'] = [...model.summary];
  }
  if (options.search)
    Object.assign(expected, {
      selectedParameters: options.multiscale ? parameters : parameters.slice(0, 2),
      // One outer comparison for each singleton search; MGTWR: initial + two components x two sweeps.
      iterationCount: [options.multiscale ? 5 : 1],
      converged: [1],
      iterationLimitReached: [0]
    });
  const outputs = Object.fromEntries(
    Object.entries(expected).map(([name, values]) => [
      name,
      port(
        ['localStatus', 'iterationCount', 'converged', 'iterationLimitReached'].includes(name)
          ? 'uint32'
          : 'float32',
        Array<number>(values.length).fill(999)
      )
    ])
  );
  const props = {
    predictorCount: 1,
    kernel: model.kernel,
    adaptive: model.adaptive,
    temporalContext: {
      unit: 'second' as const,
      epoch: 0n,
      encoding: 'float32-relative' as const,
      axis: {kind: 'irregular' as const}
    },
    positions: port(
      'float32x2',
      [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1].flatMap(value =>
        options.rescaled ? [32 + 4 * value, -16] : [value, 0]
      )
    ),
    times: port(
      'float32',
      [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1].map(value =>
        options.rescaled ? 16 + 2 * value : value
      )
    ),
    predictors: port(
      'float32',
      [-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1].map(value => value * predictorScale)
    ),
    response: port('float32', response),
    mask: port('uint32', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    parameters: port('float32', parameters),
    bandwidthSearch: options.search
      ? {
          parameters: port('float32', [
            bandwidth,
            bandwidth,
            1.25 * timeScale,
            1.25 * timeScale,
            0.001,
            0.001
          ]),
          maximumIterations: 1
        }
      : undefined
  };
  const output = {
    coefficients: outputs['coefficients'] as GraphDataView<'float32'>,
    fitted: outputs['fitted'] as GraphDataView<'float32'>,
    residuals: outputs['residuals'] as GraphDataView<'float32'>,
    localStatus: outputs['localStatus'] as GraphDataView<'uint32'>,
    selectedParameters: outputs['selectedParameters'] as GraphDataView<'float32'> | undefined,
    status: options.search
      ? {
          iterationCount: outputs['iterationCount'] as GraphDataView<'uint32'>,
          converged: outputs['converged'] as GraphDataView<'uint32'>,
          iterationLimitReached: outputs['iterationLimitReached'] as GraphDataView<'uint32'>
        }
      : undefined
  };
  const contributor = options.multiscale
    ? new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
        ...props,
        iterationCount: 2,
        output: {...output, scores: outputs['scores'] as GraphDataView<'float32'>}
      })
    : new GPUGeographicallyTemporallyWeightedRegression({
        ...props,
        searchStructure: options.indexed ? 'grid' : 'brute-force',
        output: {
          ...output,
          hatDiagonal: outputs['hatDiagonal'] as GraphDataView<'float32'>,
          summary: outputs['summary'] as GraphDataView<'float32'>
        }
      });
  return {expected, outputs, nodes: contributor.getCommandNodes(graph)};
}
