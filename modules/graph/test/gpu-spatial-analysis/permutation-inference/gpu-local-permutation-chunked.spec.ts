// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPULocalPermutationTest,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  planGPUSpatialWeightsChunks,
  type GPUChunkedSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {createKnnWeights, createSeededRandom, type CPUSpatialWeights} from './permutation-oracle';

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

async function runLocalMoran(
  device: Device,
  options: {
    values: Float32Array;
    weights: CPUSpatialWeights;
    chunkRowCounts?: readonly number[];
    rowStandardized: boolean;
  }
) {
  const rows = options.values.length;
  const graph = new GPUCommandGraph(device, {id: 'chunked-local-permutation'});
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
    exceedances: createOutputBuffer(device, rows),
    pseudoPValues: createOutputBuffer(device, rows),
    observed: createOutputBuffer(device, rows),
    overflow: createOutputBuffer(device, 1)
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH,
    values: getGPUPermutationParameterValues({permutationCount: 99, seed: 7})
  });
  graph.add(
    new GPULocalPermutationTest({
      weights: weights as never,
      values: importGraphBuffer(
        graph,
        'values',
        createInputBuffer(device, options.values),
        'float32',
        rows
      ),
      parameters: parameters.importToGraph(graph),
      statistic: 'localMoran',
      maximumPermutations: 99,
      output: {
        exceedances: importGraphBuffer(graph, 'exceedances', outputs.exceedances, 'uint32', rows),
        pseudoPValues: importGraphBuffer(graph, 'p', outputs.pseudoPValues, 'float32', rows),
        observed: importGraphBuffer(graph, 'observed', outputs.observed, 'float32', rows),
        status: {overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    exceedances: await readUint32(outputs.exceedances, rows),
    pseudoPValues: await readFloat32(outputs.pseudoPValues, rows),
    observed: await readFloat32(outputs.observed, rows),
    overflow: (await readUint32(outputs.overflow, 1))[0]
  };
  compiled.destroy();
  return result;
}

it('chunked CSR matches whole CSR for local Moran (binary and row-standardized)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(11);
  const count = 300;
  const positions = Float64Array.from({length: count * 2}, () => random() * 100);
  const values = Float32Array.from(
    {length: count},
    (_, index) => Math.sin(positions[index * 2] / 9) + random()
  );
  for (const rowStandardized of [false, true]) {
    const weights = createKnnWeights(positions, 6, rowStandardized);
    const whole = await runLocalMoran(device, {values, weights, rowStandardized});
    const counts = Array.from(
      {length: count},
      (_, row) => weights.offsets[row + 1] - weights.offsets[row]
    );
    const plan = planGPUSpatialWeightsChunks(counts, 400);
    expect(plan.length).toBeGreaterThan(2);
    const chunked = await runLocalMoran(device, {
      values,
      weights,
      rowStandardized,
      chunkRowCounts: plan.map(chunk => chunk.rowCount)
    });
    expect(chunked.overflow).toBe(whole.overflow);
    expect(Array.from(chunked.exceedances)).toEqual(Array.from(whole.exceedances));
    expect(Array.from(chunked.observed)).toEqual(Array.from(whole.observed));
    expect(Array.from(chunked.pseudoPValues)).toEqual(Array.from(whole.pseudoPValues));
  }
});
