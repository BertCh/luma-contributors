// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  importGraphBuffer,
  GPUParameterBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUMultipleTestingAdjustment} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis,
  GPULocalMoran
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createInputBuffer,
  readUint32,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLocalPermutationHarness} from '../permutation-inference/permutation-harness';
import {
  computeLocalPermutationOracle,
  createWeights
} from '../permutation-inference/permutation-oracle';

// Above the old 2^24 - 1 cap, with an even count so the alternating +/-1 population has mean 0.
const LARGE_ROW_COUNT = 2 ** 24 + 2;

for (const Contributor of [GPUHotSpotAnalysis, GPULocalMoran]) {
  it(`${Contributor.name} exceeds the old float32 row ceiling and matches a closed-form population`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = LARGE_ROW_COUNT;
    const graph: GPUCommandGraph<undefined> = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', rows);
    const offsets = createTransientView(graph, 'offsets', 'uint32', rows + 1);
    const neighbors = createTransientView(graph, 'neighbors', 'uint32', rows);
    const weights = createTransientView(graph, 'weights', 'float32', rows);
    const zScores = createTransientView(graph, 'scores', 'float32', rows);
    const probes = [0, 1, 2 ** 24 - 1, 2 ** 24, rows - 1];
    const output = createOutputBuffer(device, probes.length);
    const counts = createOutputBuffer(device, 2);
    const moments = createOutputBuffer(device, 5);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8,
      values: getGPUSpatialAutocorrelationParameterValues()
    });
    let compiled: ReturnType<GPUCommandGraph<undefined>['compile']> | undefined;
    try {
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'population',
          operation: 'scale-fixture',
          invocationCount: rows + 1,
          bindings: [
            {name: 'values', view: values, type: 'f32', access: 'read_write'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
            {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
          ],
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `offsets[offsetsOffset + index] = index;
  if (index < ROWS) {
    values[valuesOffset + index] = select(-1.0, 1.0, (index & 1u) == 0u);
    neighbors[neighborsOffset + index] = (index + 2u) % ROWS;
    weights[weightsOffset + index] = 1.0;
  }`
        })
      );
      graph.add(
        new Contributor({
          values,
          weights: {offsets, neighbors, weights},
          parameters: parameters.importToGraph(graph),
          output: {
            zScores,
            globalCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', 2),
            globalStatistics: importGraphBuffer(graph, 'moments', moments, 'float32', 5)
          }
        })
      );
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'sample',
          operation: 'scale-sample',
          invocationCount: probes.length,
          bindings: [
            {name: 'scores', view: zScores, type: 'f32', access: 'read'},
            {
              name: 'result',
              view: importGraphBuffer(graph, 'result', output, 'float32', probes.length),
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: `const PROBES = array<u32, ${probes.length}>(${probes.map(row => `${row}u`).join(', ')});`,
          body: 'result[resultOffset + index] = scores[scoresOffset + PROBES[index]];'
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled);
      expect(await readUint32(counts)).toEqual([rows, 0]);
      const statistics = await readFloat32(moments);
      expect(statistics[0]).toBe(rows);
      expect(statistics[1]).toBeCloseTo(0, 6);
      expect(statistics[2]).toBeCloseTo(1, 5);
      expect(statistics[4]).toBe(0);
      const scores = await readFloat32(output);
      const expectedMagnitude =
        Contributor === GPUHotSpotAnalysis
          ? Math.sqrt((2 * (rows - 1)) / (rows - 2))
          : (1 + 1 / (rows - 1)) / Math.sqrt(rows / (rows - 1) - 1 / (rows - 1) ** 2);
      for (const [index, score] of scores.entries()) {
        // Local Moran's z is oriented by the focus value, so both signs give positive association.
        const sign = Contributor === GPUHotSpotAnalysis && probes[index] % 2 !== 0 ? -1 : 1;
        expect(score).toBeCloseTo(sign * expectedMagnitude, 4);
      }
    } finally {
      compiled?.destroy();
      parameters.destroy();
      output.destroy();
      counts.destroy();
      moments.destroy();
    }
  }, 120_000);
}

it('split-column local permutation moments and tiled draws match an independent CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = Float32Array.from([1, 3, 2, 9, 4, 12, 7, 6]);
  const secondValues = Float32Array.from([8, 2, 9, 1, 5, 3, 11, 4]);
  const weights = createWeights(
    Array.from({length: values.length}, (_, row) => [
      [(row + 1) % values.length, 1] as const,
      [(row + 3) % values.length, 2] as const
    ])
  );
  for (const statistic of ['localMoran', 'localBivariateMoran', 'localLeeL'] as const) {
    const settings = {seed: 7, permutationCount: 31, significanceLevel: 0.05};
    const scene = {values, weights, ...(statistic === 'localMoran' ? {} : {secondValues})};
    const harness = createLocalPermutationHarness(device, {
      scene,
      statistic,
      parameters: settings,
      rowTileSize: 3
    });
    try {
      const result = await harness.run(settings);
      const oracle = computeLocalPermutationOracle({
        ...scene,
        ...settings,
        statistic,
        maximumNeighbors: 32
      });
      expect(result.overflow).toBe(0);
      for (let row = 0; row < values.length; row++) {
        expect(result.observed[row]).toBeCloseTo(oracle.observed[row], 4);
        expect(result.exceedances[row]).toBe(oracle.exceedances[row]);
        expect(result.pseudoPValues[row]).toBeCloseTo(oracle.pseudoPValues[row], 6);
      }
    } finally {
      harness.destroy();
    }
  }
});

it('multiple-testing keeps a family count above 2^24 exact and matches closed-form corrections', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const familySize = 30_000_001;
  const pValues = Float32Array.of(1e-9, 2e-9, Number.NaN);
  for (const method of ['bonferroni', 'holm', 'benjamini-hochberg'] as const) {
    const graph: GPUCommandGraph<undefined> = new GPUCommandGraph(device);
    const input = createInputBuffer(device, pValues);
    const output = createOutputBuffer(device, 3);
    const count = createOutputBuffer(device, 1);
    let compiled: ReturnType<GPUCommandGraph<undefined>['compile']> | undefined;
    try {
      graph.add(
        new GPUMultipleTestingAdjustment({
          method,
          minimumFamilySize: familySize,
          pValues: importGraphBuffer(graph, 'input', input, 'float32', 3),
          output: {
            adjustedPValues: importGraphBuffer(graph, 'output', output, 'float32', 3),
            effectiveFamilySize: importGraphBuffer(graph, 'count', count, 'uint32', 1)
          }
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled);
      expect(await readUint32(count)).toEqual([familySize]);
      const adjusted = await readFloat32(output);
      const first = pValues[0] * familySize;
      const second = method === 'holm' ? pValues[1] * (familySize - 1) : pValues[1] * familySize;
      const expected =
        method === 'benjamini-hochberg'
          ? [Math.min(first, second / 2), second / 2]
          : [first, second];
      expect(adjusted[0]).toBeCloseTo(expected[0], 6);
      expect(adjusted[1]).toBeCloseTo(expected[1], 6);
      expect(adjusted[2]).toBeNaN();
    } finally {
      compiled?.destroy();
      input.destroy();
      output.destroy();
      count.destroy();
    }
  }
});
