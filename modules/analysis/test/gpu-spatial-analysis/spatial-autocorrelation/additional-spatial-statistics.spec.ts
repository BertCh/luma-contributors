// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {Device} from '@luma.gl/core';
import {
  getGPUGammaPermutationAdapter,
  getGPUSpatialAutocorrelationParameterValues,
  GPUGammaStatistic,
  GPULocalGeary,
  GPU_LEES_L_PERMUTATION_ADAPTERS,
  GPU_LOCAL_BIVARIATE_JOIN_COUNT_PERMUTATION_ADAPTER,
  GPU_LOCAL_BIVARIATE_MORAN_PERMUTATION_ADAPTER,
  GPU_LOCAL_GEARY_PERMUTATION_ADAPTER,
  GPU_LOCAL_JOIN_COUNT_PERMUTATION_ADAPTER,
  GPUSpatialPearson
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import type {GPUIslandPolicy} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPUGlobalSpatialStatistics
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {
  getReferenceMask,
  getReferenceValues,
  getReferenceWeights,
  REFERENCE_ROW_COUNT
} from './spatial-autocorrelation-fixture';
import type {OracleWeights} from './spatial-autocorrelation-oracle';
import {isClose} from './spatial-autocorrelation-harness';
import {REFERENCE} from './spatial-autocorrelation-reference';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('additional spatial statistics match direct CPU edge oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const offsets = Uint32Array.from([0, 1, 3, 5, 6]);
  const neighbors = Uint32Array.from([1, 0, 2, 1, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const values = Float32Array.from([1, 2, 4, 8]);
  const secondValues = Float32Array.from([2, 0, 5, 9]);
  const parameters = getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05});
  const graph = new GPUCommandGraph(device, {id: 'additional-spatial-statistics'});
  const inputBuffers = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    secondValues: createInputBuffer(device, secondValues),
    parameters: createInputBuffer(device, parameters)
  };
  const outputBuffers = {
    localGeary: createOutputBuffer(device, values.length),
    pearson: createOutputBuffer(device, 1),
    gamma: createOutputBuffer(device, 1)
  };
  const weights = {
    offsets: importGraphBuffer(graph, 'offsets', inputBuffers.offsets, 'uint32', offsets.length),
    neighbors: importGraphBuffer(
      graph,
      'neighbors',
      inputBuffers.neighbors,
      'uint32',
      neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      'weights',
      inputBuffers.weights,
      'float32',
      edgeWeights.length
    )
  };
  const valueView = importGraphBuffer(
    graph,
    'values',
    inputBuffers.values,
    'float32',
    values.length
  );
  const secondView = importGraphBuffer(
    graph,
    'second-values',
    inputBuffers.secondValues,
    'float32',
    secondValues.length
  );
  graph.add(
    new GPULocalGeary({
      weights,
      values: valueView,
      parameters: importGraphBuffer(
        graph,
        'parameters',
        inputBuffers.parameters,
        'float32',
        parameters.length
      ),
      output: {
        statistic: importGraphBuffer(
          graph,
          'local-geary',
          outputBuffers.localGeary,
          'float32',
          values.length
        )
      }
    })
  );
  graph.add(
    new GPUSpatialPearson({
      weights,
      values: valueView,
      secondValues: secondView,
      output: {statistic: importGraphBuffer(graph, 'pearson', outputBuffers.pearson, 'float32', 1)}
    })
  );
  graph.add(
    new GPUGammaStatistic({
      weights,
      values: valueView,
      operation: 'squared-difference',
      output: {statistic: importGraphBuffer(graph, 'gamma', outputBuffers.gamma, 'float32', 1)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [localGeary, pearson, gamma] = await Promise.all([
    readFloat32(outputBuffers.localGeary, values.length),
    readFloat32(outputBuffers.pearson, 1),
    readFloat32(outputBuffers.gamma, 1)
  ]);

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const expectedLocal = values.map((value, row) => {
    let result = 0;
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++)
      result += (value - values[neighbors[slot]]) ** 2 / variance;
    return result;
  });
  const edgePairs = Array.from(neighbors, (neighbor, slot) => {
    let row = 0;
    while (offsets[row + 1] <= slot) row++;
    return [values[row], secondValues[neighbor]] as const;
  });
  const meanX = edgePairs.reduce((sum, pair) => sum + pair[0], 0) / edgePairs.length;
  const meanY = edgePairs.reduce((sum, pair) => sum + pair[1], 0) / edgePairs.length;
  const covariance = edgePairs.reduce(
    (sum, pair) => sum + (pair[0] - meanX) * (pair[1] - meanY),
    0
  );
  const squareX = edgePairs.reduce((sum, pair) => sum + (pair[0] - meanX) ** 2, 0);
  const squareY = edgePairs.reduce((sum, pair) => sum + (pair[1] - meanY) ** 2, 0);
  const expectedPearson = covariance / Math.sqrt(squareX * squareY);
  const expectedGamma =
    Array.from(neighbors, (neighbor, slot) => {
      let row = 0;
      while (offsets[row + 1] <= slot) row++;
      return (values[row] - values[neighbor]) ** 2;
    }).reduce((sum, value) => sum + value, 0) / neighbors.length;
  localGeary.forEach((value, row) => expect(value).toBeCloseTo(expectedLocal[row], 5));
  expect(pearson[0]).toBeCloseTo(expectedPearson, 5);
  expect(gamma[0]).toBeCloseTo(expectedGamma, 5);
  compiled.destroy();
  Object.values(inputBuffers).forEach(buffer => buffer.destroy());
  Object.values(outputBuffers).forEach(buffer => buffer.destroy());
});

/** Runs `GPULocalGeary` once and returns its values and global statistics. */
async function runLocalGeary(
  device: Device,
  options: {
    values: Float32Array;
    csr: OracleWeights;
    mask?: Uint32Array;
    islands?: GPUIslandPolicy;
    rowStandardized?: boolean;
  }
): Promise<{values: number[]; globalStatistics: number[]}> {
  const {values, csr} = options;
  const rows = values.length;
  const parameters = getGPUSpatialAutocorrelationParameterValues();
  const graph = new GPUCommandGraph(device, {id: 'local-geary-test'});
  const inputs = {
    offsets: createInputBuffer(device, csr.offsets),
    neighbors: createInputBuffer(device, csr.neighbors),
    weights: createInputBuffer(device, csr.weights),
    values: createInputBuffer(device, values),
    parameters: createInputBuffer(device, parameters),
    mask: options.mask && createInputBuffer(device, options.mask)
  };
  const outputs = {
    geary: createOutputBuffer(device, rows),
    globalStatistics: createOutputBuffer(device, 5)
  };
  graph.add(
    new GPULocalGeary({
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', inputs.offsets, 'uint32', rows + 1),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          csr.neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', csr.weights.length),
        information: options.rowStandardized
          ? {rowCount: rows, transform: 'row', symmetric: 'unknown'}
          : undefined
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', rows),
      parameters: importGraphBuffer(
        graph,
        'parameters',
        inputs.parameters,
        'float32',
        parameters.length
      ),
      mask: inputs.mask && importGraphBuffer(graph, 'mask', inputs.mask, 'uint32', rows),
      islands: options.islands,
      output: {
        statistic: importGraphBuffer(graph, 'geary', outputs.geary, 'float32', rows),
        globalStatistics: importGraphBuffer(
          graph,
          'global-statistics',
          outputs.globalStatistics,
          'float32',
          5
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: await readFloat32(outputs.geary, rows),
    globalStatistics: await readFloat32(outputs.globalStatistics, 5)
  };
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer?.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
  return result;
}

it('GPULocalGeary matches esda Geary_Local on a fixture with an island (keep, exclude, masked)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = getReferenceValues();
  // Oracle: test/oracles/statistics-local-autocorrelation.py (esda 2.10.0 Geary_Local, row-standardized).
  // The masked configuration is esda on the subset W re-row-standardized.
  for (const [policy, mask, reference, count, islandCount] of [
    ['keep', undefined, REFERENCE.keep, REFERENCE_ROW_COUNT, 1],
    ['exclude', undefined, REFERENCE.exclude, REFERENCE_ROW_COUNT - 1, 1],
    ['exclude', getReferenceMask(), REFERENCE.masked, REFERENCE.masked.count, 0]
  ] as const) {
    const result = await runLocalGeary(device, {
      values,
      csr: getReferenceWeights({rowStandardize: true}),
      mask,
      islands: policy,
      rowStandardized: true
    });
    const label = `${policy}${mask ? ' masked' : ''}`;
    for (let row = 0; row < REFERENCE_ROW_COUNT; row++) {
      if (!isClose(result.values[row], reference.gearyRow[row], 1e-4, 1e-3)) {
        throw new Error(
          `${label}: row ${row} c ${result.values[row]} != esda ${reference.gearyRow[row]}`
        );
      }
    }
    expect(result.globalStatistics[0], label).toBe(count);
    expect(result.globalStatistics[4], label).toBe(islandCount);
    // A kept island has c = 0 (esda); an excluded island is NaN.
    expect(result.values[REFERENCE.island], label).toSatisfy((value: number) =>
      policy === 'keep' ? value === 0 : Number.isNaN(value)
    );
  }
}, 60000);

it('local Geary permutation adapter reads clusters from the lower tail', () => {
  // Small c_i means similar neighbors, so the clustering tail is the lower one.
  expect(GPU_LOCAL_GEARY_PERMUTATION_ADAPTER.alternative).toBe('lesser');
});

it('permutation adapters of the local bivariate, join count and Lee statistics name the engine statistic', () => {
  expect(GPU_LOCAL_BIVARIATE_MORAN_PERMUTATION_ADAPTER).toMatchObject({
    scope: 'local',
    statistic: 'localBivariateMoran',
    permutedVariable: 'secondValues'
  });
  // Clustering is more joins than expected: the upper tail.
  expect(GPU_LOCAL_JOIN_COUNT_PERMUTATION_ADAPTER.alternative).toBe('greater');
  expect(GPU_LOCAL_BIVARIATE_JOIN_COUNT_PERMUTATION_ADAPTER.statistic).toBe(
    'localBivariateJoinCount'
  );
  expect(GPU_LEES_L_PERMUTATION_ADAPTERS.global).toMatchObject({
    scope: 'global',
    statistic: 'leesL'
  });
  expect(GPU_LEES_L_PERMUTATION_ADAPTERS.local).toMatchObject({
    scope: 'local',
    statistic: 'localLeeL'
  });
});

// spatial-statistics-04: the cross-product Gamma centers on the edge-weighted means and divides by
// the edge weight, so on a graph with unequal degrees it does not order permutations like the
// Moran pair sum. The adapter to the 'moran' permutation test is therefore a test of Moran's I
// there, not of this Gamma (it is the same ordering when every row, or every column, has the same
// total weight). Star 0-1, 0-2, 0-3 plus the edge 3-4, binary weights, values 1, 2, 4, 7, 11
// permuted two ways. Hand values, with the degree-weighted mean m = sum_i d_i x_i / 8:
// - p = (7, 2, 4, 1, 11): m = 5, Gamma = 2 (2 * -3 + 2 * -1 + 2 * -4 + -4 * 6) / 8 = -10;
//   Moran pair sum (global mean 5) = -80, I = (5 / 8) * -80 / 66 = -0.757576.
// - q = (11, 2, 1, 7, 4): m = 6.75,
//   Gamma = 2 (4.25 * -4.75 + 4.25 * -5.75 + 4.25 * 0.25 + 0.25 * -2.75) / 8 = -11.0625;
//   Moran pair sum = 2 (6 * -3 + 6 * -4 + 6 * 2 + 2 * -1) = -64, I = (5 / 8) * -64 / 66 = -0.606061.
// Gamma ranks p above q; Moran's I ranks q above p.
it('cross-product Gamma and Moran order two permutations of a star-and-tail graph differently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const offsets = Uint32Array.from([0, 3, 4, 5, 7, 8]);
  const neighbors = Uint32Array.from([1, 2, 3, 0, 0, 0, 4, 3]);
  const run = async (values: number[]) => {
    const graph = new GPUCommandGraph(device, {id: 'gamma-against-moran'});
    const inputs = [
      createInputBuffer(device, offsets),
      createInputBuffer(device, neighbors),
      createInputBuffer(device, new Float32Array(neighbors.length).fill(1)),
      createInputBuffer(device, Float32Array.from(values))
    ];
    const gammaBuffer = createOutputBuffer(device, 1);
    const summaryBuffer = createOutputBuffer(device, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
    const weights = {
      offsets: importGraphBuffer(graph, 'offsets', inputs[0], 'uint32', offsets.length),
      neighbors: importGraphBuffer(graph, 'neighbors', inputs[1], 'uint32', neighbors.length),
      weights: importGraphBuffer(graph, 'weights', inputs[2], 'float32', neighbors.length)
    };
    const valueView = importGraphBuffer(graph, 'values', inputs[3], 'float32', values.length);
    graph.add(
      new GPUGammaStatistic({
        weights,
        values: valueView,
        operation: 'cross-product',
        output: {statistic: importGraphBuffer(graph, 'gamma', gammaBuffer, 'float32', 1)}
      })
    );
    graph.add(
      new GPUGlobalSpatialStatistics({
        weights,
        values: valueView,
        statistics: ['moran'],
        output: {
          summary: importGraphBuffer(
            graph,
            'summary',
            summaryBuffer,
            'float32',
            GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
          )
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const gamma = (await readFloat32(gammaBuffer, 1))[0];
    const moran = (await readFloat32(summaryBuffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length))[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran
    ];
    compiled.destroy();
    for (const buffer of [...inputs, gammaBuffer, summaryBuffer]) {
      buffer.destroy();
    }
    return {gamma, moran};
  };
  const first = await run([7, 2, 4, 1, 11]);
  const second = await run([11, 2, 1, 7, 4]);
  expect(first.gamma).toBeCloseTo(-10, 4);
  expect(second.gamma).toBeCloseTo(-11.0625, 4);
  expect(first.moran).toBeCloseTo(-0.757576, 5);
  expect(second.moran).toBeCloseTo(-0.606061, 5);
  expect(first.gamma).toBeGreaterThan(second.gamma);
  expect(first.moran).toBeLessThan(second.moran);
  // The adapters still name the engine statistics; only the squared-difference one is exact.
  expect(getGPUGammaPermutationAdapter('cross-product').statistic).toBe('moran');
  expect(getGPUGammaPermutationAdapter('squared-difference').statistic).toBe('geary');
  expect(() => getGPUGammaPermutationAdapter('absolute-difference')).toThrow();
}, 60000);
