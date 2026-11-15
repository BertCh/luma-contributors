// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createFillNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {getRegressionWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {
  CPUMultiscaleGeographicallyWeightedRegressionOperation,
  computeMultiscaleGeographicallyWeightedRegressionOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation';
import {GPUMultiscaleGeographicallyWeightedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation';
import {
  createMultiscaleBandwidthWorkspace,
  createMultiscaleBandwidthSelection
} from '../../../src/gpu-spatial-analysis/spatial-regression/mgwr-bandwidth-selection';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('MGWR AICc scores remain finite with large predictors and full inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const predictors = Float32Array.from({length: 8}, (_, row) => (row - 3.5) * 1e15);
  const response = Float32Array.from(
    predictors,
    (value, row) => value * 1e-5 + (row % 2 ? 1e9 : -1e9)
  );
  const outputs = [
    'coefficients',
    'fitted',
    'selectedBandwidths',
    'codes',
    'iterationCount',
    'standardErrors',
    'effectiveParameters',
    'summary',
    'inferenceStatus'
  ];
  const results = [];
  for (const Operation of [
    GPUMultiscaleGeographicallyWeightedRegressionOperation,
    CPUMultiscaleGeographicallyWeightedRegressionOperation
  ]) {
    results.push(
      await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array(16)},
          predictors: {format: 'float32', values: predictors},
          response: {format: 'float32', values: response},
          parameters: {format: 'float32', values: Float32Array.of(1e-3, 2, 2)},
          candidates: {format: 'float32', values: Float32Array.of(2, 10, 2, 10)}
        },
        operations: [
          new Operation({
            rowCount: 8,
            predictorCount: 1,
            kernel: 'bisquare',
            maximumIterations: 32,
            positions: 'positions',
            predictors: 'predictors',
            response: 'response',
            parameters: 'parameters',
            bandwidthCandidates: 'candidates',
            bandwidthCandidateCount: 2,
            bandwidthSelection: {criterion: 'AICc'},
            inference: {
              output: {
                standardErrors: 'standardErrors',
                effectiveParameters: 'effectiveParameters',
                summary: 'summary',
                status: 'inferenceStatus'
              }
            },
            output: {
              coefficients: 'coefficients',
              fitted: 'fitted',
              selectedBandwidths: 'selectedBandwidths',
              status: {codes: 'codes', iterationCount: 'iterationCount'}
            }
          })
        ],
        outputs
      })
    );
  }
  for (const result of results) {
    expect(Array.from(result['codes'])).toEqual(new Array(8).fill(0));
    expect(Array.from(result['selectedBandwidths'])).toEqual([2, 2]);
    expect(result['inferenceStatus'][0]).toBe(0);
    for (const port of outputs)
      expect(Array.from(result[port]).every(Number.isFinite), port).toBe(true);
  }
  const [actual, expected] = results;
  for (let row = 0; row < 8; row++) {
    // Intercept error is measured on the response scale; the exactly zero float64 intercept
    // can accumulate a few float32 ulps when summing responses of magnitude 1e10.
    expect(
      Math.abs(actual['coefficients'][2 * row] - expected['coefficients'][2 * row])
    ).toBeLessThan(2e5);
    expect(
      Math.abs(actual['coefficients'][2 * row + 1] - expected['coefficients'][2 * row + 1])
    ).toBeLessThan(1e-9);
    expect(Math.abs(actual['fitted'][row] - expected['fitted'][row])).toBeLessThan(5e5);
    const actualResidual = response[row] - actual['fitted'][row];
    const expectedResidual = response[row] - expected['fitted'][row];
    expect(Math.abs(actualResidual - expectedResidual)).toBeLessThan(5e5);
  }
  for (const port of ['standardErrors', 'effectiveParameters', 'summary']) {
    for (let index = 0; index < actual[port].length; index++) {
      expect(
        Math.abs(actual[port][index] - expected[port][index]),
        `${port}[${index}]`
      ).toBeLessThanOrEqual(2e-3 * Math.abs(expected[port][index]));
    }
  }
});

for (const indexGridSize of [false, [4, 1]] as const) {
  it(`MGWR finds feasible AICc above two invalid golden interiors: ${indexGridSize ? 'indexed' : 'dense'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = {
      positions: Float32Array.from({length: 16}, (_, index) => (index % 2 ? 0 : (1.5 * index) / 2)),
      predictors: Float32Array.of(1, -1, 1, -1, 1, -1, 1, -1),
      response: Float32Array.of(1, 2, 4, 3, 2, 1, 5, 3),
      parameters: Float32Array.of(1e-3, 2.2, 2.2),
      bandwidthCandidates: Float32Array.of(0.1, 2.2, 0.1, 2.2)
    };
    const options = {
      predictorCount: 1,
      kernel: 'bisquare' as const,
      indexGridSize,
      bandwidthCandidateCount: 2,
      bandwidthSelection: {
        criterion: 'AICc' as const,
        searchMethod: 'golden-section' as const,
        maximumSearchIterations: 12
      }
    };
    const expected = computeMultiscaleGeographicallyWeightedRegressionOnCPU(input, options);
    const actual = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: input.positions},
        predictors: {format: 'float32', values: input.predictors},
        response: {format: 'float32', values: input.response},
        parameters: {format: 'float32', values: input.parameters},
        bandwidthCandidates: {format: 'float32', values: input.bandwidthCandidates}
      },
      operations: [
        new GPUMultiscaleGeographicallyWeightedRegressionOperation({
          ...options,
          rowCount: 8,
          maximumIterations: 32,
          positions: 'positions',
          predictors: 'predictors',
          response: 'response',
          parameters: 'parameters',
          bandwidthCandidates: 'bandwidthCandidates',
          output: {
            coefficients: 'coefficients',
            fitted: 'fitted',
            selectedBandwidths: 'selectedBandwidths',
            status: {codes: 'codes', iterationCount: 'iterationCount'}
          }
        })
      ],
      outputs: ['coefficients', 'fitted', 'selectedBandwidths', 'codes', 'iterationCount']
    });
    expect(Array.from(expected.codes)).toEqual(new Array(8).fill(0));
    expect(Array.from(actual['codes'])).toEqual(new Array(8).fill(0));
    for (const port of ['coefficients', 'fitted', 'selectedBandwidths'] as const) {
      for (let index = 0; index < expected[port].length; index++) {
        expect(Number.isFinite(actual[port][index]), port).toBe(true);
        expect(Math.abs(actual[port][index] - expected[port][index]), port).toBeLessThan(5e-3);
      }
    }
  });
}

for (const searchMethod of ['finite-grid', 'golden-section'] as const) {
  it(`excluded rows disarm pass-zero selection without touching scratch: ${searchMethod}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format);
    };
    const output = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number,
      initial?: number
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      if (initial !== undefined) buffer.write(new Float32Array(length).fill(initial));
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const rows = 1024;
    const codes = output('codes', 'uint32', rows);
    const iterations = output('iterations', 'uint32', 1);
    const parameters = input('parameters', 'float32', Float32Array.of(1e-3, 2, 2));
    const positions = input('positions', 'float32x2', new Float32Array(2 * rows));
    const props = {
      positions,
      predictors: input('predictors', 'float32', new Float32Array(rows).fill(1)),
      response: input('response', 'float32', new Float32Array(rows)),
      parameters,
      mask: input('mask', 'uint32', new Uint32Array(rows)),
      predictorCount: 1,
      kernel: 'bisquare' as const,
      maximumIterations: 1,
      output: {
        coefficients: output('coefficients', 'float32', 2 * rows).view,
        fitted: output('fitted', 'float32', rows).view,
        status: {codes: codes.view, iterationCount: iterations.view}
      }
    };
    graph.add(new GPUMultiscaleGeographicallyWeightedRegression(props));
    // The real first sweep above writes EXCLUDED to every code. Use that GPU-written state
    // to exercise the exact selection gate that the node spec checks in all 128 sweeps.
    const selection = {
      criterion: 'AICc' as const,
      searchMethod,
      maximumSearchIterations: searchMethod === 'golden-section' ? 12 : undefined
    };
    const initialWorkspace = createMultiscaleBandwidthWorkspace(graph, 'probe', rows, 2, selection);
    const sentinel = -12345;
    const scores = output('probe-scores', 'float32', initialWorkspace.scores.length, sentinel);
    const partials = output(
      'probe-partials',
      'float32',
      initialWorkspace.partials.length,
      sentinel
    );
    const totals = output('probe-totals', 'float32', initialWorkspace.totals.length, sentinel);
    const table = output('probe-table', 'float32', initialWorkspace.table.length, sentinel);
    const workspace = {
      ...initialWorkspace,
      scores: scores.view,
      partials: partials.view,
      totals: totals.view,
      table: table.view
    };
    const pairs = input('pairs', 'float32', new Float32Array(2 * rows).fill(NaN));
    const candidates = input('candidates', 'float32', Float32Array.of(2, 10, 2, 10));
    const nodes = createMultiscaleBandwidthSelection(graph, {
      id: 'stopped-selection',
      positions,
      pairs,
      gridBindings: [],
      common: `${getRegressionWGSL(props, {neighborhoodOnly: true})}\nfn isActive(row: u32) -> bool { return isFiniteBits(pairs[pairsOffset + 2u * row]); }`,
      hasGrid: false,
      rows,
      column: 0,
      candidates,
      candidateCount: 2,
      selection,
      parameters,
      state: codes.view,
      workspace
    });
    for (const node of nodes) {
      graph.add(node);
      // Search initialization clears its small history table; sentinel it afterward so an
      // accidentally unconditional score pass is observable too.
      if (node.id.endsWith('-search-init'))
        graph.add(
          createFillNode(graph, {
            id: 'sentinel-table',
            operation: 'MGWRReviewProbe',
            view: table.view,
            type: 'f32',
            value: '-12345.0'
          })
        );
    }
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(codes.buffer)).toEqual(new Array(rows).fill(1));
      expect(await readUint32(iterations.buffer)).toEqual([1]);
      for (const scratch of [scores, partials, totals, table])
        expect(await readFloat32(scratch.buffer)).toEqual(
          new Array(scratch.view.length).fill(sentinel)
        );
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

function createOverflowingResidualScene(rowCount: number) {
  const signs = [1, -1, -1, 1, 1, -1, -1, 1];
  return {
    positions: new Float32Array(2 * rowCount),
    predictors: Float32Array.from({length: rowCount}, (_, row) => (row % 2 ? -1 : 1)),
    // Excluded padding has individually overflowing squares and must never enter the score.
    response: Float32Array.from({length: rowCount}, (_, row) =>
      row < 8 ? signs[row] * 1e19 : 3e19
    ),
    mask: Uint32Array.from({length: rowCount}, (_, row) => (row < 8 ? 1 : 0)),
    parameters: Float32Array.of(1e-3, 2, 2),
    bandwidthCandidates: Float32Array.of(2, 10, 2, 10)
  };
}

for (const rowCount of [8, 32]) {
  for (const searchMethod of ['finite-grid', 'golden-section'] as const) {
    it(`MGWR AICc accepts finite variance despite overflowing RSS: ${searchMethod}/${rowCount} rows`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scene = createOverflowingResidualScene(rowCount);
      const expectedBandwidth = searchMethod === 'finite-grid' ? 2 : Math.fround(2 + 0.38197 * 8);
      const results = [];
      for (const Operation of [
        GPUMultiscaleGeographicallyWeightedRegressionOperation,
        CPUMultiscaleGeographicallyWeightedRegressionOperation
      ]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: scene.positions},
              predictors: {format: 'float32', values: scene.predictors},
              response: {format: 'float32', values: scene.response},
              mask: {format: 'uint32', values: scene.mask},
              parameters: {format: 'float32', values: scene.parameters},
              bandwidthCandidates: {format: 'float32', values: scene.bandwidthCandidates}
            },
            operations: [
              new Operation({
                rowCount,
                predictorCount: 1,
                kernel: 'bisquare',
                maximumIterations: 16,
                positions: 'positions',
                predictors: 'predictors',
                response: 'response',
                mask: 'mask',
                parameters: 'parameters',
                bandwidthCandidates: 'bandwidthCandidates',
                bandwidthCandidateCount: 2,
                bandwidthSelection: {
                  criterion: 'AICc',
                  searchMethod,
                  maximumSearchIterations: searchMethod === 'golden-section' ? 4 : undefined
                },
                output: {
                  coefficients: 'coefficients',
                  fitted: 'fitted',
                  selectedBandwidths: 'selectedBandwidths',
                  status: {codes: 'codes', iterationCount: 'iterationCount'}
                }
              })
            ],
            outputs: ['coefficients', 'fitted', 'selectedBandwidths', 'codes', 'iterationCount']
          })
        );
      }
      for (const result of results) {
        expect(Array.from(result['codes'])).toEqual(Array.from(scene.mask, keep => (keep ? 0 : 1)));
        expect(Array.from(result['selectedBandwidths'])).toEqual([
          expectedBandwidth,
          expectedBandwidth
        ]);
        for (let row = 0; row < rowCount; row++) {
          if (row < 8) {
            expect(result['fitted'][row]).toBe(0);
            expect(result['coefficients'][2 * row]).toBe(0);
            expect(result['coefficients'][2 * row + 1]).toBe(0);
          } else {
            expect(result['fitted'][row]).toBeNaN();
            expect(result['coefficients'][2 * row]).toBeNaN();
            expect(result['coefficients'][2 * row + 1]).toBeNaN();
          }
        }
        expect(result['iterationCount'][0]).toBeGreaterThan(0);
        expect(result['iterationCount'][0]).toBeLessThanOrEqual(16);
      }
    });
  }

  it(`MGWR AICc score is about 729.089 with overflowing RSS and ${rowCount} capacity rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const scene = createOverflowingResidualScene(rowCount);
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format);
    };
    const positions = input('positions', 'float32x2', scene.positions);
    const predictors = input('predictors', 'float32', scene.predictors);
    const response = input('response', 'float32', scene.response);
    const parameters = input('parameters', 'float32', scene.parameters);
    const candidates = input('candidates', 'float32', scene.bandwidthCandidates);
    const state = input('state', 'uint32', Uint32Array.of(3, 0));
    const pairs = input(
      'pairs',
      'float32',
      Float32Array.from({length: 2 * rowCount}, (_, element) => {
        const row = Math.floor(element / 2);
        return element % 2 ? 1 : scene.mask[row] ? scene.response[row] : NaN;
      })
    );
    const selection = {criterion: 'AICc' as const};
    const initialWorkspace = createMultiscaleBandwidthWorkspace(
      graph,
      'scaled-score',
      rowCount,
      2,
      selection
    );
    const tableBuffer = createOutputBuffer(device, initialWorkspace.table.length);
    buffers.push(tableBuffer);
    const table = importGraphBuffer(
      graph,
      'score-table',
      tableBuffer,
      'float32',
      initialWorkspace.table.length
    );
    const workspace = {...initialWorkspace, table};
    const nodes = createMultiscaleBandwidthSelection(graph, {
      id: 'scaled-score-selection',
      positions,
      pairs,
      gridBindings: [],
      hasGrid: false,
      rows: rowCount,
      column: 0,
      common: `${getRegressionWGSL({positions, predictors, response, parameters, predictorCount: 1, kernel: 'bisquare'}, {neighborhoodOnly: true})}\nfn isActive(row: u32) -> bool { return isFiniteBits(pairs[pairsOffset + 2u * row]); }`,
      candidates,
      candidateCount: 2,
      selection,
      parameters,
      state,
      workspace
    });
    for (const node of nodes) graph.add(node);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const scores = await readFloat32(tableBuffer);
      // Eight included rows, a scalar constant smoother with trace 1, variance y^2.
      const expected =
        8 * Math.log(scene.response[0] ** 2) + 8 * Math.log(2 * Math.PI) + (8 * 9) / 5;
      expect(expected).toBeCloseTo(729.089, 3);
      for (const candidate of [0, 1]) {
        expect(Number.isFinite(scores[3 * candidate + 1])).toBe(true);
        expect(Math.abs(scores[3 * candidate + 1] - expected)).toBeLessThan(2e-3);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
