// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {computeSpatialRegressionDiagnosticsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {computeSpatialTwoStageLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {computeSpatialPanelErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {REVIEW_REGRESSION_FIXTURE} from './rr-weights-regression-review-fixture';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {CPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {GPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import {CPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {CPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {GPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {computeLocalMoranOracle} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {
  computeSpatialWeightsTracesOnCPU,
  CPUSpatialWeightsTracesOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';

const TRACE_LAYOUT_WEIGHTS = {
  offsets: Uint32Array.of(0, 1, 2, 3),
  neighbors: Uint32Array.of(0, 1, 2),
  weights: Float32Array.of(1, 2, 4)
};
const TRACE_LAYOUT_CHUNKS = [
  {
    rowStart: 0,
    rowCount: 1,
    weights: {
      offsets: Uint32Array.of(0, 1),
      neighbors: Uint32Array.of(0),
      weights: Float32Array.of(1)
    }
  },
  {
    rowStart: 1,
    rowCount: 2,
    weights: {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 2),
      weights: Float32Array.of(2, 4)
    }
  }
];

for (const metadata of ['absent', 'undefined', 'recursive'] as const) {
  it(`TSC-01/TSC-02/TSC-03/TSC-09 ordinary CSR uses offsets with ${metadata} loss chunks`, () => {
    const weights = {
      ...TRACE_LAYOUT_WEIGHTS,
      ...(metadata === 'undefined' ? {chunks: undefined} : {}),
      ...(metadata === 'recursive'
        ? {
            chunks: [
              {
                weights: {
                  offsets: Uint32Array.of(0),
                  status: {complete: Uint32Array.of(1)},
                  chunks: [{weights: {status: {candidateOverflow: Uint32Array.of(0)}}}]
                }
              }
            ]
          }
        : {})
    };
    const result = computeSpatialWeightsTracesOnCPU(
      {weights},
      {maximumPower: 3, mode: 'exact', includeGramTraces: true}
    );
    // W = diag(1,2,4), norm = 4. Sum and trace of (W/4)^p agree.
    // Unscaled Gram moments are sum(d^2)=21, sum(d^3)=73, sum(d^4)=273 (twice).
    expect(Array.from(result.powerTraces)).toEqual([1.75, 1.3125, 1.140625]);
    expect(Array.from(result.powerSums)).toEqual([1.75, 1.3125, 1.140625]);
    expect(Array.from(result.gramTraces ?? [])).toEqual([21, 73, 273, 273]);
    expect(Array.from(result.infinityNorm)).toEqual([4]);
  });
}

it('TSC-04/TSC-08/TSC-09 numerical chunks retain later rows and zero-row chunks', () => {
  const empty = {
    rowStart: 1,
    rowCount: 0,
    weights: {offsets: Uint32Array.of(0), neighbors: new Uint32Array(), weights: new Float32Array()}
  };
  const result = computeSpatialWeightsTracesOnCPU(
    {weights: {rowCount: 3, chunks: [TRACE_LAYOUT_CHUNKS[0], empty, TRACE_LAYOUT_CHUNKS[1]]}},
    {maximumPower: 3, mode: 'exact'}
  );
  // The second chunk contributes 2/4 + 4/4 to tr(W/4); later powers contribute
  // (2/4)^p + (4/4)^p. The first chunk contributes (1/4)^p.
  expect(Array.from(result.powerTraces)).toEqual([1.75, 1.3125, 1.140625]);
  expect(Array.from(result.powerSums)).toEqual([1.75, 1.3125, 1.140625]);
  expect(Array.from(result.powerTraceStandardErrors)).toEqual([0, 0, 0]);
  expect(() =>
    computeSpatialWeightsTracesOnCPU(
      {weights: {rowCount: 3, chunks: TRACE_LAYOUT_CHUNKS}},
      {maximumPower: 1, includeGramTraces: true}
    )
  ).toThrow(/weights chunks do not support Gram traces/);
  const emptyResult = computeSpatialWeightsTracesOnCPU(
    {weights: {rowCount: 0, chunks: [{...empty, rowStart: 0}]}},
    {maximumPower: 1, mode: 'exact'}
  );
  expect(Array.from(emptyResult.powerTraces)).toEqual([0]);
  expect(Array.from(emptyResult.infinityNorm)).toEqual([1]);
});

it('TSC-03/TSC-04/TSC-05/TSC-07/TSC-09 rejects malformed numerical partitions', () => {
  const first = TRACE_LAYOUT_CHUNKS[0];
  const second = TRACE_LAYOUT_CHUNKS[1];
  for (const [chunks, message] of [
    [[], /weights.chunks must cover/],
    [[first, {...second, rowStart: 2}], /rowStart must equal 1/],
    [
      [first, {...second, rowCount: -1}],
      /computeSpatialWeightsTracesOnCPU weights\.chunks\[1\]\.rowCount must be an integer in \[0, 4294967295\]/
    ],
    [[first, {...second, rowCount: 1}], /offsets length must equal rowCount/],
    [[first], /weights.chunks must cover/]
  ] as const) {
    expect(() =>
      computeSpatialWeightsTracesOnCPU({weights: {rowCount: 3, chunks}}, {maximumPower: 1})
    ).toThrow(message);
  }
  expect(() =>
    computeSpatialWeightsTracesOnCPU(
      {weights: {rowCount: 3, chunks: undefined as never}},
      {maximumPower: 1}
    )
  ).toThrow(/weights.chunks must be an array/);
});

for (const layout of ['undefined-offsets', 'null-prototype-chunk'] as const) {
  it(`review B1/B3 accepts a numerical partition with ${layout}`, () => {
    const chunk = {
      rowStart: 0,
      rowCount: 1,
      weights: {
        offsets: Uint32Array.of(0, 1),
        neighbors: Uint32Array.of(0),
        weights: Float32Array.of(7)
      }
    };
    const weights = {
      rowCount: 1,
      ...(layout === 'undefined-offsets' ? {offsets: undefined} : {}),
      chunks: [
        layout === 'null-prototype-chunk'
          ? Object.assign(Object.create(null) as typeof chunk, chunk)
          : chunk
      ]
    };
    const result = computeSpatialWeightsTracesOnCPU({weights}, {maximumPower: 2, mode: 'exact'});
    // W = [7] has norm 7, so every normalized power is [1].
    expect(Array.from(result.powerTraces)).toEqual([1, 1]);
    expect(Array.from(result.powerSums)).toEqual([1, 1]);
    expect(() =>
      computeSpatialWeightsTracesOnCPU({weights}, {maximumPower: 1, includeGramTraces: true})
    ).toThrow(/weights chunks do not support Gram traces/);
  });
}

for (const complete of [0, 1]) {
  it(`review B2 metadata-only recursive tree honors complete=${complete}`, () => {
    const weights = {
      ...TRACE_LAYOUT_WEIGHTS,
      chunks: [{weights: {chunks: [{weights: {status: {complete: Uint32Array.of(complete)}}}]}}]
    };
    const compute = () => computeSpatialWeightsTracesOnCPU({weights}, {maximumPower: 2});
    if (complete === 0) expect(compute).toThrow(/incomplete weights/);
    else expect(Array.from(compute().powerTraces)).toEqual([1.75, 1.3125]);
  });
}

it('review S1 validates the numerical collection before traversing loss metadata', () => {
  expect(() =>
    computeSpatialWeightsTracesOnCPU(
      {weights: {rowCount: 0, chunks: {} as never}},
      {maximumPower: 1}
    )
  ).toThrow(/computeSpatialWeightsTracesOnCPU weights.chunks must be an array/);
});

it('review unfinished asymmetric matrix retains signed walks in both layouts', () => {
  const weights = {
    offsets: Uint32Array.of(0, 2, 3),
    neighbors: Uint32Array.of(0, 1, 0),
    weights: Float32Array.of(2, 2, -1)
  };
  const partition = {
    rowCount: 2,
    chunks: [
      {
        rowStart: 0,
        rowCount: 1,
        weights: {
          offsets: Uint32Array.of(0, 2),
          neighbors: Uint32Array.of(0, 1),
          weights: Float32Array.of(2, 2)
        }
      },
      {
        rowStart: 1,
        rowCount: 0,
        weights: {
          offsets: Uint32Array.of(0),
          neighbors: new Uint32Array(),
          weights: new Float32Array()
        }
      },
      {
        rowStart: 1,
        rowCount: 1,
        weights: {
          offsets: Uint32Array.of(0, 1),
          neighbors: Uint32Array.of(0),
          weights: Float32Array.of(-1)
        }
      }
    ]
  };
  for (const input of [weights, partition]) {
    const result = computeSpatialWeightsTracesOnCPU(
      {weights: input},
      {maximumPower: 2, mode: 'exact'}
    );
    // W/4 = [[1/2,1/2],[-1/4,0]]; its square is [[1/8,1/4],[-1/8,-1/8]].
    expect(Array.from(result.powerTraces)).toEqual([0.5, 0]);
    expect(Array.from(result.powerSums)).toEqual([0.75, 0.125]);
  }
});

for (const scope of ['whole', 'chunked-root', 'chunked-child'] as const) {
  it(`review earlier-fix note rejects unsupported named ${scope} candidateOverflow`, () => {
    const named = {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'};
    const weights =
      scope === 'whole'
        ? {...named, candidateOverflow: 'loss'}
        : {
            rowCount: 1,
            ...(scope === 'chunked-root' ? {candidateOverflow: 'loss'} : {}),
            chunks: [
              {
                rowStart: 0,
                rowCount: 1,
                weights: {
                  ...named,
                  ...(scope === 'chunked-child' ? {candidateOverflow: 'loss'} : {})
                }
              }
            ]
          };
    expect(
      () =>
        new CPUSpatialWeightsTracesOperation({
          id: 'review-named-trace-loss',
          rowCount: 1,
          maximumPower: 1,
          weights,
          output: {powerTraces: 'traces', powerSums: 'sums'}
        })
    ).toThrow(/review-named-trace-loss weights.*candidateOverflow.*unsupported/);
  });
}

for (const scope of [
  'ordinary-root',
  'ordinary-metadata',
  'chunked-root',
  'numerical-child'
] as const) {
  it(`TSC-01/TSC-06 preserves ${scope} loss evidence`, () => {
    const loss = {candidateOverflow: Uint32Array.of(1)};
    const weights = scope.startsWith('ordinary')
      ? {
          ...TRACE_LAYOUT_WEIGHTS,
          ...(scope === 'ordinary-root' ? {status: loss} : {chunks: [{weights: {status: loss}}]})
        }
      : {
          rowCount: 3,
          ...(scope === 'chunked-root' ? {status: loss} : {}),
          chunks: TRACE_LAYOUT_CHUNKS.map((chunk, index) => ({
            ...chunk,
            weights: {
              ...chunk.weights,
              ...(scope === 'numerical-child' && index === 1 ? {status: loss} : {})
            }
          }))
        };
    expect(() => computeSpatialWeightsTracesOnCPU({weights}, {maximumPower: 1})).toThrow(
      /incomplete weights/
    );
  });
}

it('TSC-06 validates the numerical child CSR arrays', () => {
  expect(() =>
    computeSpatialWeightsTracesOnCPU(
      {
        weights: {
          rowCount: 3,
          chunks: TRACE_LAYOUT_CHUNKS.map(chunk => ({
            ...chunk,
            weights: {...chunk.weights, weights: new Float32Array()}
          }))
        }
      },
      {maximumPower: 1}
    )
  ).toThrow(/weights.weights length must equal weights.neighbors length/);
});

it('v3 residual Moran uses the documented sample normalization on the eight-row ring', () => {
  const actual = computeLocalMoranOracle({
    values: Float32Array.of(
      -7 / 6,
      1 / 42,
      -143 / 42,
      17 / 14,
      -31 / 14,
      143 / 42,
      -5 / 6,
      125 / 42
    ),
    weights: REVIEW_REGRESSION_FIXTURE.weights
  }).localI;
  // Closed form: residuals sum to zero, M2 = 850/21 and each lag is the next ring residual.
  // I_i = 7 c_i c_(i+1) / M2. Population normalization would multiply each answer by 8/7.
  const expected = [
    -49 / 10200,
    -143 / 10200,
    -143 / 200,
    -93 / 200,
    -4433 / 3400,
    -1001 / 2040,
    -175 / 408,
    -245 / 408
  ];
  for (const [index, value] of expected.entries()) {
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-6);
  }
});

for (const scope of ['whole', 'root', 'child'] as const) {
  it(`R1 ${scope} retained terminal offsets expose loss hidden by spare capacity`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    let serial = 0;
    const input = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const view = createTransientView(graph, `retained-${serial++}`, format, values.length);
      memory.write(view, values);
      return view;
    };
    const requiredCount = input('uint32', Uint32Array.of(scope === 'child' ? 2 : 3));
    const createWeights = (offsets: number[], neighbors: number[]) => ({
      offsets: input('uint32', Uint32Array.from(offsets)),
      neighbors: input('uint32', Uint32Array.from(neighbors)),
      weights: input(
        'float32',
        Float32Array.from(neighbors, () => 1)
      )
    });
    const weights =
      scope === 'whole'
        ? {...createWeights([0, 1, 2, 3], [1, 0, 1, 99, 99]), status: {requiredCount}}
        : {
            rowCount: 3,
            ...(scope === 'root' ? {status: {requiredCount}} : {}),
            chunks: [
              {rowStart: 0, rowCount: 1, weights: createWeights([0, 1], [1])},
              {
                rowStart: 1,
                rowCount: 2,
                weights: {
                  ...createWeights([0, 1, 2], [0, 1, 99, 99]),
                  ...(scope === 'child' ? {status: {requiredCount}} : {})
                }
              }
            ]
          };
    const norm = input('float32', Float32Array.of(77, 98, 99));
    try {
      const nodes = new GPUSpatialWeightsTraces({
        weights,
        maximumPower: 1,
        mode: 'exact',
        output: {
          infinityNorm: norm,
          powerTraces: input('float32', Float32Array.of(77)),
          powerSums: input('float32', Float32Array.of(77))
        }
      }).getCommandNodes(graph);
      // Three retained unit links give norm 1; the extra allocated slots are not retained links.
      for (const [index, required] of (scope === 'child' ? [2, 3, 5, 2] : [3, 4, 6, 3]).entries()) {
        memory.write(requiredCount, Uint32Array.of(required));
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readFloat32(norm))).toEqual([
          index === 1 || index === 2 ? NaN : 1,
          98,
          99
        ]);
      }
    } finally {
      device.destroy();
    }
  });
}

it('R10 aggregate chunk capacity never emits an out-of-range uint32 literal', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const offsets = createTransientView(graph, 'large-offsets', 'uint32', 2);
  // Logical read-only views are shared by 128 chunks; no aggregate storage is allocated.
  const neighbors = createTransientView(graph, 'large-neighbors', 'uint32', 2 ** 25);
  const weights = createTransientView(graph, 'large-weights', 'float32', 2 ** 25);
  try {
    const nodes = new GPUSpatialWeightsTraces({
      weights: {
        rowCount: 128,
        status: {requiredCount: createTransientView(graph, 'large-required', 'uint32', 1)},
        chunks: Array.from({length: 128}, (_, rowStart) => ({
          rowStart,
          rowCount: 1,
          weights: {offsets, neighbors, weights}
        }))
      },
      maximumPower: 1,
      mode: 'stochastic',
      probeCount: 1,
      output: {
        powerTraces: createTransientView(graph, 'large-traces', 'float32', 1),
        powerSums: createTransientView(graph, 'large-sums', 'float32', 1)
      }
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.includes('weights-loss-count'))
    );
    expect(sources.length).toBeGreaterThan(0);
    expect(sources.map(source => source.source).join('\n')).not.toContain('4294967296u');
  } finally {
    device.destroy();
  }
});

for (const inherited of [false, true]) {
  it(`R5 declared approximation map ignores a ${inherited ? 'inherited view' : 'buffer property'}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    let serial = 0;
    const input = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const view = createTransientView(graph, `status-shape-${serial++}`, format, values.length);
      memory.write(view, values);
      return view;
    };
    const unrelated = input('uint32', Uint32Array.of(91));
    const approximation = input('uint32', Uint32Array.of(77));
    const loss = input('uint32', Uint32Array.of(0));
    const status = inherited
      ? Object.assign(Object.create(unrelated), {approximation})
      : {approximation, buffer: unrelated.buffer};
    try {
      const nodes = new GPUSpatialWeightsTraces({
        maximumPower: 1,
        mode: 'exact',
        weights: {
          offsets: input('uint32', Uint32Array.of(0, 1, 2)),
          neighbors: input('uint32', Uint32Array.of(1, 0)),
          weights: input('float32', Float32Array.of(1, 1)),
          status: {candidateOverflow: loss}
        },
        output: {
          powerTraces: input('float32', Float32Array.of(77)),
          powerSums: input('float32', Float32Array.of(77)),
          status
        }
      }).getCommandNodes(graph);
      for (const value of [0, 1, 0]) {
        memory.write(loss, Uint32Array.of(value));
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readUint32(unrelated))).toEqual([91]);
        expect(Array.from(memory.readUint32(approximation))).toEqual([0]);
      }
    } finally {
      device.destroy();
    }
  });
}

it('R7 CPU diagnostics suppress count-proven loss and recover independently checked LM p-values', () => {
  const requiredCount = Uint32Array.of(8);
  const residuals = Float32Array.of(
    -1.166666666666667,
    0.023809523809522837,
    -3.404761904761907,
    1.214285714285713,
    -2.214285714285717,
    3.4047619047619033,
    -0.8333333333333366,
    2.9761904761904736
  );
  for (const required of [8, 9, 8]) {
    requiredCount[0] = required;
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      {
        ...REVIEW_REGRESSION_FIXTURE,
        residuals,
        weights: {
          ...REVIEW_REGRESSION_FIXTURE.weights,
          status: {count: Uint32Array.of(8), requiredCount}
        }
      } as never,
      {predictorCount: 1}
    );
    expect(Array.from(result.status)).toEqual([required === 9 ? 3 : 0]);
    if (required === 9) expect(Number.isNaN(result.tests[2])).toBe(true);
    else {
      expect(Number.isFinite(result.tests[2])).toBe(true);
      expect(result.tests[2]).toBeCloseTo(0.4229256619310794, 5);
    }
  }
});

it('R8 CPU HAC count loss suppresses covariance independently of the point fit', () => {
  const requiredCount = Uint32Array.of(8);
  for (const required of [8, 9, 8]) {
    requiredCount[0] = required;
    const result = computeSpatialTwoStageLeastSquaresOnCPU(
      {
        ...REVIEW_REGRESSION_FIXTURE,
        robust: {
          method: 'HAC',
          kernel: {
            offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
            neighbors: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
            weights: new Float32Array(8).fill(1),
            status: {count: Uint32Array.of(8), requiredCount}
          }
        }
      } as never,
      {predictorCount: 1}
    );
    expect(Array.from(result.status)).toEqual([0]);
    expect(Array.from(result.robust!.status)).toEqual([required === 9 ? 4 : 0]);
    expect(result.table[0]).toBeCloseTo(1.294032023289664, 5);
    if (required === 9)
      expect(Array.from(result.robust!.covariance).every(Number.isNaN)).toBe(true);
    else expect(result.robust!.covariance[0]).toBeCloseTo(2.7335123628111035, 5);
  }
});

it('R9 CPU panel completeness belongs to the four-entity graph, not eight expanded slots', () => {
  const requiredCount = Uint32Array.of(4);
  for (const required of [4, 5, 4]) {
    requiredCount[0] = required;
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
      {
        ...REVIEW_REGRESSION_FIXTURE,
        weights: {
          offsets: Uint32Array.of(0, 1, 2, 3, 4),
          neighbors: Uint32Array.of(1, 2, 3, 0),
          weights: Float32Array.of(1, 1, 1, 1),
          status: {requiredCount}
        }
      } as never,
      {
        predictorCount: 1,
        periodCount: 2,
        maximumPower: 256,
        traceMode: 'exact',
        tailExtrapolation: true
      }
    );
    expect(Array.from(result.status)).toEqual([required === 5 ? 3 : 0]);
    expect(Array.from(result.diagnosticStatus)).toEqual([required === 5 ? 1 : 0]);
    if (required === 5) expect(Array.from(result.diagnosticTests).every(Number.isNaN)).toBe(true);
    else {
      expect(Number.isFinite(result.diagnosticTests[0])).toBe(true);
      expect(result.diagnosticTests[0]).toBeCloseTo(1.7313633782924958, 5);
    }
  }
});

it('R9 foreign count view is refused before panel allocation and a corrected same-ID retry succeeds', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const foreign = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `panel-retry-${serial++}`, format, length);
  const requiredCount = createTransientView(foreign, 'foreign-count', 'uint32', 1);
  const props = {
    id: 'panel-count-retry',
    predictorCount: 1,
    periodCount: 2,
    maximumPower: 2,
    traceMode: 'exact' as const,
    predictors: view('float32', 8),
    response: view('float32', 8),
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 4),
      weights: view('float32', 4),
      status: {requiredCount}
    },
    output: {
      table: view('float32', 12),
      summary: view('float32', 22),
      status: {code: view('uint32', 1)}
    },
    diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
  };
  try {
    expect(() => graph.add(new GPUSpatialPanelErrorMaximumLikelihood(props))).toThrow(/graph/);
    expect(() =>
      graph.add(
        new GPUSpatialPanelErrorMaximumLikelihood({
          ...props,
          weights: {...props.weights, status: {requiredCount: view('uint32', 1)}}
        })
      )
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

for (const kind of ['diagnostics', 'hac', 'panel'] as const) {
  it(`R7/R8/R9 named ${kind} forms retain count ports and CPU execution recovers`, async () => {
    const fixture = REVIEW_REGRESSION_FIXTURE;
    const vector = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const required = Uint32Array.of(kind === 'panel' ? 4 : 8);
    const base = {
      id: `named-${kind}`,
      predictorCount: 1,
      rowCount: 8,
      entityCount: 4,
      periodCount: 2,
      maximumPower: 256,
      traceMode: 'exact' as const,
      tailExtrapolation: true,
      predictors: 'predictors',
      response: 'response',
      residuals: 'residuals',
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {table: 'table', summary: 'summary', tests: 'tests', status: {code: 'code'}}
    };
    const countStatus = {count: 'retained', requiredCount: 'needed'};
    const named =
      kind === 'hac'
        ? {
            ...base,
            robust: {
              method: 'HAC' as const,
              kernel: {
                offsets: 'kernelOffsets',
                neighbors: 'kernelNeighbors',
                weights: 'kernelWeights',
                status: countStatus
              },
              output: {covariance: 'covariance', status: {code: 'robustCode'}}
            }
          }
        : {
            ...base,
            weights: {...base.weights, status: countStatus},
            diagnostics: {tests: 'panelTests', status: 'panelCode'}
          };
    const cpu =
      kind === 'diagnostics'
        ? new CPUSpatialRegressionDiagnosticsOperation(named)
        : kind === 'hac'
          ? new CPUSpatialTwoStageLeastSquaresOperation(named)
          : new CPUSpatialPanelErrorMaximumLikelihoodOperation(named);
    const gpu =
      kind === 'diagnostics'
        ? new GPUSpatialRegressionDiagnosticsOperation(named)
        : kind === 'hac'
          ? new GPUSpatialTwoStageLeastSquaresOperation(named)
          : new GPUSpatialPanelErrorMaximumLikelihoodOperation(named);
    const requiredPort =
      kind === 'diagnostics'
        ? 'weightsRequiredCount'
        : kind === 'hac'
          ? 'kernelRequiredCount'
          : 'requiredCount';
    expect(cpu.inputs[requiredPort]).toBe('needed');
    expect(gpu.inputs[requiredPort]).toBe('needed');
    const countPort =
      kind === 'diagnostics' ? 'weightsCount' : kind === 'hac' ? 'kernelCount' : 'count';
    const inputs = {
      predictors: vector(fixture.predictors),
      response: vector(fixture.response),
      residuals: vector(
        Float32Array.of(
          -1.166666666666667,
          0.023809523809522837,
          -3.404761904761907,
          1.214285714285713,
          -2.214285714285717,
          3.4047619047619033,
          -0.8333333333333366,
          2.9761904761904736
        )
      ),
      offsets: vector(kind === 'panel' ? Uint32Array.of(0, 1, 2, 3, 4) : fixture.weights.offsets),
      neighbors: vector(kind === 'panel' ? Uint32Array.of(1, 2, 3, 0) : fixture.weights.neighbors),
      weights: vector(new Float32Array(kind === 'panel' ? 4 : 8).fill(1)),
      kernelOffsets: vector(Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8)),
      kernelNeighbors: vector(Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7)),
      kernelWeights: vector(new Float32Array(8).fill(1)),
      [requiredPort]: vector(required),
      [countPort]: vector(Uint32Array.of(kind === 'panel' ? 4 : 8))
    };
    for (const missing of [false, true, false]) {
      required[0] = (kind === 'panel' ? 4 : 8) + Number(missing);
      const result = (await cpu.execute({inputs})) as Record<string, Float32Array | Uint32Array>;
      expect(Array.from(result['code'])).toEqual([missing && kind !== 'hac' ? 3 : 0]);
      if (kind === 'hac') expect(Array.from(result['robustCode'])).toEqual([missing ? 4 : 0]);
      if (kind === 'panel') expect(Array.from(result['panelCode'])).toEqual([missing ? 1 : 0]);
    }
  });
}

it('R2 kNN clears the bound shared candidate word on every cached-node submission', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const candidateOverflow = createTransientView(graph, 'shared-candidate', 'uint32', 1);
  try {
    const nodes = new GPUNeighborSearch({
      id: 'knn-reset',
      mode: 'knn',
      neighborCount: 1,
      gridSize: [2, 2],
      positions: createTransientView(graph, 'knn-positions', 'float32x2', 2),
      parameters: createTransientView(graph, 'knn-parameters', 'float32', 12),
      output: {
        offsets: createTransientView(graph, 'knn-offsets', 'uint32', 3),
        neighbors: createTransientView(graph, 'knn-neighbors', 'uint32', 2),
        weights: createTransientView(graph, 'knn-weights', 'float32', 2),
        status: {
          overflow: createTransientView(graph, 'knn-overflow', 'uint32', 1),
          candidateOverflow
        }
      }
    })
      .getCommandNodes(graph)
      .filter(node => node.id === 'knn-reset-candidate-status');
    expect(nodes).toHaveLength(1);
    for (const previousLoss of [0, 1, 9]) {
      memory.write(candidateOverflow, Uint32Array.of(previousLoss));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(candidateOverflow))).toEqual([0]);
    }
  } finally {
    device.destroy();
  }
});

it('R5 a named robust solver map takes precedence over inherited view metadata', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `solver-shape-${serial++}`, format, length);
  const unrelated = view('uint32', 1),
    code = view('uint32', 1),
    loss = view('uint32', 1);
  const status = Object.assign(Object.create(unrelated), {code});
  try {
    const nodes = new GPUSpatialTwoStageLeastSquares({
      predictorCount: 1,
      predictors: view('float32', 8),
      response: view('float32', 8),
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8),
        status: {candidateOverflow: loss}
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 16),
        status: {code: view('uint32', 1)}
      },
      robust: {method: 'HC0', output: {covariance: view('float32', 9), status}}
    })
      .getCommandNodes(graph)
      .filter(node => /weights-loss|suppress-incomplete/.test(node.id));
    memory.write(unrelated, Uint32Array.of(91));
    memory.write(code, Uint32Array.of(0));
    memory.write(loss, Uint32Array.of(1));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(unrelated))).toEqual([91]);
    expect(Array.from(memory.readUint32(code))).toEqual([3]);
  } finally {
    device.destroy();
  }
});
