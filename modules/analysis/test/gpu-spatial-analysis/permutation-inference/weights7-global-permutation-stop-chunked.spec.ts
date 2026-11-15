// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  planGPUSpatialWeightsChunks,
  type GPUChunkedSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createKnnWeights, createSeededRandom, type CPUSpatialWeights} from './permutation-oracle';

const R = GPU_GLOBAL_PERMUTATION_SUMMARY;

function sliceWeights(
  weights: CPUSpatialWeights,
  rowStart: number,
  rowCount: number
): CPUSpatialWeights {
  const begin = weights.offsets[rowStart];
  const end = weights.offsets[rowStart + rowCount];
  return {
    offsets: Uint32Array.from(
      weights.offsets.subarray(rowStart, rowStart + rowCount + 1),
      value => value - begin
    ),
    neighbors: weights.neighbors.slice(begin, Math.max(end, begin + 1)),
    weights: weights.weights.slice(begin, Math.max(end, begin + 1))
  };
}

async function runGlobal(
  device: Device,
  options: {
    values: Float32Array;
    weights: CPUSpatialWeights;
    chunkRowCounts?: readonly number[];
    rowStandardized: boolean;
    statistic: 'moran' | 'geary';
    stopAfterExceedances?: number;
    permutationBatchSize?: number;
  }
) {
  const rows = options.values.length;
  const maximumPermutations = 499;
  const graph = new GPUCommandGraph(device, {id: 'chunked-global-permutation'});
  const information = options.rowStandardized
    ? {rowCount: rows, transform: 'row' as const, symmetric: 'unknown' as const}
    : undefined;
  const importCsr = (csr: CPUSpatialWeights, name: string, rowCount: number) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      createInputBuffer(device, csr.offsets),
      'uint32',
      rowCount + 1
    ),
    neighbors: importGraphBuffer(
      graph,
      `${name}-neighbors`,
      createInputBuffer(device, csr.neighbors),
      'uint32',
      csr.neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      `${name}-weights`,
      createInputBuffer(device, csr.weights),
      'float32',
      csr.weights.length
    )
  });
  let weights: GPUChunkedSpatialWeights | ReturnType<typeof importCsr>;
  if (options.chunkRowCounts) {
    let rowStart = 0;
    weights = {
      rowCount: rows,
      information,
      chunks: options.chunkRowCounts.map((rowCount, index) => {
        const chunk = {
          rowStart,
          rowCount,
          weights: importCsr(
            sliceWeights(options.weights, rowStart, rowCount),
            `chunk${index}`,
            rowCount
          )
        };
        rowStart += rowCount;
        return chunk;
      })
    };
  } else {
    weights = {...importCsr(options.weights, 'whole', rows), information} as never;
  }
  const outputs = {
    summary: createOutputBuffer(device, R.length),
    referenceDistribution: createOutputBuffer(device, maximumPermutations)
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH,
    values: getGPUPermutationParameterValues({permutationCount: maximumPermutations, seed: 5})
  });
  graph.add(
    new GPUGlobalPermutationTest({
      weights: weights as never,
      values: importGraphBuffer(
        graph,
        'values',
        createInputBuffer(device, options.values),
        'float32',
        rows
      ),
      parameters: parameters.importToGraph(graph),
      statistic: options.statistic,
      maximumPermutations,
      stopAfterExceedances: options.stopAfterExceedances,
      permutationBatchSize: options.permutationBatchSize,
      output: {
        summary: importGraphBuffer(graph, 'summary', outputs.summary, 'float32', R.length),
        referenceDistribution: importGraphBuffer(
          graph,
          'reference',
          outputs.referenceDistribution,
          'float32',
          maximumPermutations
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    summary: await readFloat32(outputs.summary, R.length),
    reference: await readFloat32(outputs.referenceDistribution, maximumPermutations)
  };
  compiled.destroy();
  return result;
}

it('stopAfterExceedances is invariant to the chunk (tile) size of the weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(23);
  const count = 260;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  // Spatially random values: exceedances are frequent, so the run stops well before 499.
  const values = Float32Array.from({length: count}, () => random());
  for (const statistic of ['moran', 'geary'] as const) {
    for (const rowStandardized of [false, true]) {
      const weights = createKnnWeights(positions, 5, rowStandardized);
      const counts = Array.from(
        {length: count},
        (_, row) => weights.offsets[row + 1] - weights.offsets[row]
      );
      const whole = await runGlobal(device, {
        values,
        weights,
        rowStandardized,
        statistic,
        stopAfterExceedances: 12
      });
      // A real stop: fewer permutations than the maximum, exactly the target of exceedances.
      expect(whole.summary[R.permutationCount]).toBeLessThan(499);
      for (const slotLimit of [900, 350, 120]) {
        const plan = planGPUSpatialWeightsChunks(counts, slotLimit);
        expect(plan.length).toBeGreaterThan(1);
        const chunked = await runGlobal(device, {
          values,
          weights,
          rowStandardized,
          statistic,
          stopAfterExceedances: 12,
          chunkRowCounts: plan.map(chunk => chunk.rowCount)
        });
        const materialized = await runGlobal(device, {
          values,
          weights,
          rowStandardized,
          statistic,
          stopAfterExceedances: 12,
          permutationBatchSize: 3,
          chunkRowCounts: plan.map(chunk => chunk.rowCount)
        });
        expect(materialized.summary).toEqual(chunked.summary);
        expect(materialized.reference).toEqual(chunked.reference);
        const label = `${statistic} ${rowStandardized} chunks=${plan.length}`;
        expect(chunked.summary[R.permutationCount], label).toBe(whole.summary[R.permutationCount]);
        expect(chunked.summary[R.exceedances], label).toBe(whole.summary[R.exceedances]);
        expect(chunked.summary[R.pseudoPValue], label).toBeCloseTo(
          whole.summary[R.pseudoPValue],
          6
        );
        expect(chunked.summary[R.observed], label).toBeCloseTo(whole.summary[R.observed], 4);
        const used = whole.summary[R.permutationCount];
        for (let index = 0; index < used; index++) {
          expect(chunked.reference[index], `${label} permutation ${index + 1}`).toBeCloseTo(
            whole.reference[index],
            4
          );
        }
      }
    }
  }
});
