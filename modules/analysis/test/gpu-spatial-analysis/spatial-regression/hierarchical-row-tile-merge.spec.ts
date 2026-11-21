// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getRowTileSumNodes,
  getRowTileMergeNodes
} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import {getPairwiseSumWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/pairwise-sum-wgsl';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createInputBuffer, readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {createGWRBatchMergeFixture} from './tile-merge-review-fixtures';

// NumPy literals and the independent float64 bound: hierarchical-row-tile-merge.oracle.py.
// Every operation in these dyadic fixtures is exact in float32, including the group cancellation.
for (const [tileCount, total, bits] of [
  [4096, 1049429.125, 1233132201],
  [4097, 853.125, 1146439680],
  [8192, 1706.25, 1154828288],
  [8193, 1707.25, 1154836480]
] as const) {
  for (const tileRowCount of [1, 2]) {
    it(`sums and maximizes ${tileCount} tiles of ${tileRowCount} rows against the NumPy order`, async () => {
      const context = await createRegressionTestContext();
      const {graph, input, output} = context;
      try {
        const rowCount = tileCount * tileRowCount;
        const values = new Float32Array(rowCount);
        for (let tile = 0; tile < tileCount; tile++)
          values[tile * tileRowCount] =
            tile === 0
              ? 2 ** 20
              : tile === 4096
                ? -(2 ** 20)
                : tile === 8192
                  ? 1
                  : tile % 3 === 0
                    ? 0.125
                    : 0.25;
        const result = output('float32', 4);
        for (const node of getRowTileSumNodes(graph, {
          id: 'boundary',
          operation: 'TileMergeBoundaryTest',
          variant: 'sum-max',
          rowCount,
          layout: {tileRowCount, tileCount},
          sumCount: 1,
          maximumCount: 1,
          bindings: [{name: 'values', view: input('float32', values), type: 'f32', access: 'read'}],
          declarations: `fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
            (*terms)[0] += values[valuesOffset + row];
            (*terms)[1] = max((*terms)[1], select(0.0, 77.0, row == ROW_COUNT - 1u));
          }`,
          partial: createTransientView(graph, 'boundary-partial', 'float32', tileCount * 2),
          output: result.view,
          outputSlot: 1
        }))
          graph.add(node);
        context.run();
        const actual = await readFloat32(result.buffer, 4);
        expect(Number.isFinite(actual[1])).toBe(true);
        expect(actual[1]).toBe(total);
        expect(new Uint32Array(Float32Array.of(actual[1]).buffer)[0]).toBe(bits);
        expect(actual[2]).toBe(77); // A dropped last group loses this maximum too.
      } finally {
        context.destroy();
      }
    });
  }
}

for (const blockWidth of [1, 2, 6]) {
  it(`merges blocked partials with block width ${blockWidth}, including every tail channel`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const tileCount = 8193;
      const termCount = 6;
      const values = new Float32Array(tileCount * termCount);
      for (let term = 0; term < termCount; term++)
        for (let tile = 0; tile < tileCount; tile++) {
          const value =
            tile === 0
              ? 2 ** 20
              : tile === 4096
                ? -(2 ** 20)
                : tile === 8192
                  ? 1
                  : tile % 3 === 0
                    ? 0.125
                    : 0.25;
          values[
            Math.floor(term / blockWidth) * tileCount * blockWidth +
              tile * blockWidth +
              (term % blockWidth)
          ] = value * (term + 1);
        }
      const result = output('float32', termCount);
      for (const node of getRowTileMergeNodes(
        graph,
        {
          id: 'blocked',
          operation: 'TileMergeLayoutTest',
          variant: 'blocked',
          invocationCount: termCount,
          bindings: [
            {name: 'partial', view: input('float32', values), type: 'f32', access: 'read'},
            {name: 'result', view: result.view, type: 'f32', access: 'read_write'}
          ],
          declarations: `const TILE_COUNT: u32 = ${tileCount}u;`,
          body: `${getPairwiseSumWGSL({count: 'TILE_COUNT', term: `partial[partialOffset + (index / ${blockWidth}u) * TILE_COUNT * ${blockWidth}u + sumItem * ${blockWidth}u + index % ${blockWidth}u]`})}
          result[resultOffset + index] = sum;`
        },
        {tileCount, partials: [{name: 'partial', termCount, blockWidth}]}
      ))
        graph.add(node);
      context.run();
      const actual = await readFloat32(result.buffer, termCount);
      // NumPy float32 oracle for the six scaled channels (not a scaled rounded total).
      expect(Array.from(actual)).toEqual([1707.25, 3414.5, 5122, 6829, 8536, 10244]);
    } finally {
      context.destroy();
    }
  });
}

it('R7 uses a final cascade rather than a serial merge of four groups', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const values = new Float32Array(12289);
    values[0] = 2 ** 25;
    values[8192] = -(2 ** 25);
    values[12288] = 1;
    const total = output('float32', 1);
    for (const node of getRowTileSumNodes(graph, {
      id: 'four-groups',
      operation: 'FourGroupTest',
      variant: 'sum',
      rowCount: 12289,
      layout: {tileCount: 12289, tileRowCount: 1},
      sumCount: 1,
      bindings: [{name: 'source', view: input('float32', values), type: 'f32', access: 'read'}],
      declarations:
        'fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) { (*terms)[0] += source[sourceOffset + row]; }',
      partial: createTransientView(graph, 'four-partials', 'float32', 12289),
      output: total.view
    }))
      graph.add(node);
    context.run();
    // NumPy float32 cascade=0 exactly; a serial final merge and float64 sum both give 1.
    expect(await readFloat32(total.buffer, 1)).toEqual([0]);
  } finally {
    context.destroy();
  }
});

it('R3 OLS preserves masked first-group origins, counts and fitted estimates', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const coefficients = output('float32', 2),
      summary = output('float32', 22),
      code = output('uint32', 1);
    graph.add(
      new GPUOrdinaryLeastSquares({
        predictors: input(
          'float32',
          Float32Array.from({length: 8193}, (_, row) => 1000 + [-3, -1, 1, 3][row % 4])
        ),
        response: input(
          'float32',
          Float32Array.from(
            {length: 8193},
            (_, row) => 2005 + 2 * [-3, -1, 1, 3][row % 4] + [1, -1, -1, 1][row % 4]
          )
        ),
        mask: input(
          'uint32',
          Uint32Array.from({length: 8193}, (_, row) => Number(row >= 4096 && row < 8192))
        ),
        predictorCount: 1,
        tileRowCount: 1,
        output: {
          coefficients: coefficients.view,
          standardErrors: output('float32', 2).view,
          tStatistics: output('float32', 2).view,
          summary: summary.view,
          status: {code: code.view}
        }
      })
    );
    context.run();
    const actual = await readFloat32(coefficients.buffer, 2);
    // Four balanced row types, repeated 1024 times: residual noise is orthogonal to 1 and X.
    expect(actual.every(Number.isFinite)).toBe(true);
    expect(Math.abs(actual[0] - 5)).toBeLessThan(5e-4);
    expect(Math.abs(actual[1] - 2)).toBeLessThan(5e-7);
    expect((await readFloat32(summary.buffer, 22))[0]).toBe(4096);
    expect(await readUint32(code.buffer, 1)).toEqual([0]);
  } finally {
    context.destroy();
  }
});

function createCycleData() {
  const rowCount = 4097;
  const predictors = Float32Array.from(
    {length: rowCount},
    (_, row) => (((row * 17) % 101) - 50) / 16
  );
  const response = Float32Array.from(
    predictors,
    (value, row) =>
      1.25 +
      0.75 * value +
      0.2 * predictors[(row + rowCount - 1) % rowCount] +
      (((row * 29) % 67) - 33) / 64
  );
  return {
    rowCount,
    predictors,
    response,
    offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row),
    neighbors: Uint32Array.from({length: rowCount}, (_, row) => (row + rowCount - 1) % rowCount),
    weights: new Float32Array(rowCount).fill(1)
  };
}

for (const [kind, estimates, sigmaSquared] of [
  ['gm', [1.2492946260909914, 0.8015760707197507, -0.17513339631777944], 0.21370093245400762],
  ['two-stage', [0.9352273223442122, 0.7686951315481243, 0.2519345209879894], 0.12604706471098542],
  ['error-ml', [1.24930386947166, 0.8038976112926899, -0.19878703154547148], 0.21359596160762787],
  ['lag-ml', [0.9877028865430821, 0.7709065894681595, 0.20982985131793205], 0.12207259068582586]
] as const) {
  it(`${kind} estimates 4097 one-row tiles against a float64 NumPy estimator`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const scene = createCycleData();
      const table = output('float32', 12);
      const summary = output('float32', kind === 'gm' ? 13 : kind === 'two-stage' ? 16 : 22);
      const code = output('uint32', 1);
      const props = {
        id: kind,
        weights: {
          offsets: input('uint32', scene.offsets),
          neighbors: input('uint32', scene.neighbors),
          weights: input('float32', scene.weights)
        },
        predictors: input('float32', scene.predictors),
        response: input('float32', scene.response),
        predictorCount: 1,
        tileRowCount: 1,
        traceMode: 'exact' as const,
        maximumPower: 16,
        output: {table: table.view, summary: summary.view, status: {code: code.view}}
      };
      graph.add(
        kind === 'gm'
          ? new GPUSpatialErrorGM({...props, estimator: 'kelejian-prucha'})
          : kind === 'two-stage'
            ? new GPUSpatialTwoStageLeastSquares(props)
            : kind === 'error-ml'
              ? new GPUSpatialErrorMaximumLikelihood(props)
              : new GPUSpatialLagMaximumLikelihood(props)
      );
      context.run();
      expect(Array.from(await readUint32(code.buffer, 1))).toEqual([0]);
      const actual = await readFloat32(table.buffer, 12);
      // ~14 rounding levels plus the small equilibrated solve/search; 5e-4 absolute is
      // conservative for these order-one coefficients and rejects dropping/duplicating a group.
      if (kind === 'error-ml') {
        // Directed cycle: tr(W A)=tr((W A)^2)=0; tr((W A)' W A)=n/(1-lambda^2).
        expect(Number.isFinite(actual[9])).toBe(true);
        expect(Math.abs(actual[9] - 0.0153113)).toBeLessThan(5e-7);
      }
      estimates.forEach((expected, index) => {
        expect(Number.isFinite(actual[index * 4])).toBe(true);
        expect(Math.abs(actual[index * 4] - expected)).toBeLessThan(5e-4);
      });
      const actualSummary = await readFloat32(summary.buffer, summary.view.length);
      expect(actualSummary[0]).toBe(4097);
      const variance = actualSummary[kind === 'two-stage' ? 1 : 2];
      expect(Number.isFinite(variance)).toBe(true);
      expect(Math.abs(variance - sigmaSquared)).toBeLessThan(5e-5);
    } finally {
      context.destroy();
    }
  });
}

for (const route of ['trace-child'] as const) {
  it(`${route} reduces the real last-chunk norm with 4097 one-row chunks`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const rowCount = 4097;
      const chunkWeights = {
        offsets: input('uint32', [0, 1]),
        neighbors: input('uint32', [0]),
        weights: input('float32', [1])
      };
      const tailWeights = {...chunkWeights, weights: input('float32', [7])};
      const norm = output('float32', 2);
      const contributor = new GPUSpatialWeightsTraces({
        id: 'nested-tail',
        weights: {
          rowCount,
          chunks: Array.from({length: rowCount}, (_, rowStart) => ({
            rowStart,
            rowCount: 1,
            weights: rowStart === rowCount - 1 ? tailWeights : chunkWeights
          }))
        },
        maximumPower: 1,
        mode: 'stochastic',
        probeCount: 1,
        tileRowCount: 32,
        output: {
          powerTraces: output('float32', 1).view,
          powerSums: output('float32', 1).view
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      const groups = nodes.find(node => node.id === 'nested-tail-norm-merge-normPartial-groups')!;
      const merge = nodes.find(node => node.id === 'nested-tail-norm-merge')!;
      expect(groups).toBeDefined();
      const partial = groups.resources!.filter(resource => 'buffer' in resource)[0]
        .buffer as GraphDataView;
      // Both classes use an internal norm when no optional lag-only outputs are requested.
      // Error ML has no output.infinityNorm port: observe the actual nested merge destination.
      const mergedNorm = merge.resources!.filter(resource => 'buffer' in resource)[1]
        .buffer as GraphDataView;
      // Run the real tail producer. Earlier partial slots are zero-initialized and have norm 1
      // in the full CSR; neither value can mask the unique tail maximum 7.
      const tail = nodes.find(
        node => node.id.includes('norm-tiles') && node.id.endsWith('-chunk4096')
      )!;
      expect(tail).toBeDefined();
      graph.add(tail);
      graph.add(groups);
      graph.add(merge);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'observe-nested-norm',
          operation: 'NestedTraceMergeTest',
          invocationCount: 1,
          bindings: [
            {name: 'mergedNorm', view: mergedNorm, type: 'f32', access: 'read'},
            {name: 'partial', view: partial, type: 'f32', access: 'read'},
            {name: 'observedNorm', view: norm.view, type: 'f32', access: 'read_write'}
          ],
          body: 'observedNorm[observedNormOffset] = mergedNorm[mergedNormOffset]; observedNorm[observedNormOffset + 1u] = partial[partialOffset + 4096u];'
        })
      );
      context.run();
      // Independent closed form: max(1 repeated 4096 times, 7) = 7 exactly in float32.
      expect(await readFloat32(norm.buffer, 2)).toEqual([7, 7]);
    } finally {
      context.destroy();
    }
  });
}

it('weights traces writes every port above the old ceiling on a directed cycle', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const scene = createCycleData();
    const powerTraces = output('float32', 2);
    const powerSums = output('float32', 2);
    const gramTraces = output('float32', 4);
    const infinityNorm = output('float32', 1);
    const approximation = output('uint32', 1);
    const errors = output('float32', 2);
    graph.add(
      new GPUSpatialWeightsTraces({
        weights: {
          offsets: input('uint32', scene.offsets),
          neighbors: input('uint32', scene.neighbors),
          weights: input('float32', scene.weights)
        },
        tileRowCount: 1,
        mode: 'exact',
        maximumPower: 2,
        includeGramTraces: true,
        output: {
          powerTraces: powerTraces.view,
          powerSums: powerSums.view,
          gramTraces: gramTraces.view,
          infinityNorm: infinityNorm.view,
          powerTraceStandardErrors: errors.view,
          status: {approximation: approximation.view}
        }
      })
    );
    context.run();
    // W is a permutation with no cycles of length 1 or 2: W'W=I, and every power sends 1 to 1.
    expect(Array.from(await readFloat32(powerTraces.buffer, 2))).toEqual([0, 0]);
    expect(Array.from(await readFloat32(powerSums.buffer, 2))).toEqual([4097, 4097]);
    expect(Array.from(await readFloat32(gramTraces.buffer, 4))).toEqual([4097, 0, 4097, 4097]);
    expect(Array.from(await readFloat32(infinityNorm.buffer, 1))).toEqual([1]);
    expect(Array.from(await readFloat32(errors.buffer, 2))).toEqual([0, 0]);
    expect(Array.from(await readUint32(approximation.buffer, 1))).toEqual([0]);
  } finally {
    context.destroy();
  }
});

it('R8 GWR writes distinct second-batch spreads into runs 32 and 33', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const spreads = output('float32', 68);
    const fixture = createGWRBatchMergeFixture(
      graph,
      input('uint32', getGPUPermutationParameterValues({seed: 42, permutationCount: 33})),
      spreads.view
    );
    for (const node of fixture.nodes) graph.add(node);
    context.run();
    // Hand-derived balanced +/- a moments, with a=3 and a=5 in the two final runs.
    expect((await readFloat32(spreads.buffer, 68)).slice(62, 68)).toEqual([1, 2, 3, 6, 5, 10]);
  } finally {
    context.destroy();
  }
});

it('nonstationarity merges the supplied observed coefficients above 4096 tiles', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output, device} = context;
  const positionsBuffer = createInputBuffer(
    device,
    Float32Array.from({length: 4097 * 2}, (_, index) =>
      index % 2 === 0 ? Math.floor(index / 2) % 64 : Math.floor(index / 128)
    )
  );
  try {
    const scene = createCycleData();
    const table = output('float32', 12);
    const summary = output('float32', 4);
    const standardDeviations = output('float32', 4);
    const approximation = output('uint32', 1);
    graph.add(
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 4097),
        predictors: input('float32', scene.predictors),
        response: input('float32', scene.response),
        predictorCount: 1,
        coefficients: input(
          'float32',
          Float32Array.from({length: 4097 * 2}, (_, index) =>
            index % 2 === 0 ? (Math.floor(index / 2) % 17) / 8 : (Math.floor(index / 2) % 13) / 4
          )
        ),
        bandwidthParameters: input(
          'float32',
          getGPUGeographicallyWeightedRegressionParameterValues({
            kernel: 'bisquare',
            bandwidthMode: 'fixed',
            bandwidths: [3]
          })
        ),
        selectedBandwidth: input('float32', [0, 3]),
        parameters: input(
          'uint32',
          getGPUPermutationParameterValues({seed: 42, permutationCount: 1})
        ),
        tileRowCount: 1,
        maximumPermutations: 1,
        indexGridSize: [64, 64],
        output: {
          table: table.view,
          summary: summary.view,
          standardDeviations: standardDeviations.view,
          status: {approximation: approximation.view}
        }
      })
    );
    context.run();
    const actual = await readFloat32(table.buffer, 12);
    const spreads = await readFloat32(standardDeviations.buffer, 4);
    // NumPy std(ddof=0) over the caller's local coefficient columns, oracle.py.
    for (const [column, expected] of [
      [0, 0.6123724356957945],
      [1, 0.9356831550232827]
    ]) {
      expect(Number.isFinite(actual[column * 6])).toBe(true);
      expect(Math.abs(actual[column * 6] - expected)).toBeLessThan(2e-6);
      expect(Math.abs(spreads[column] - expected)).toBeLessThan(2e-6);
    }
    expect(Array.from(await readFloat32(summary.buffer, 4))).toEqual([1, 4097, 4097, 0]);
    expect(Array.from(await readUint32(approximation.buffer, 1))).toEqual([0]);
  } finally {
    context.destroy();
    positionsBuffer.destroy();
  }
});
