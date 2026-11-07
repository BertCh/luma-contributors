// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUGammaStatistic,
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary,
  GPUSpatialPearson
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import type {
  GPUChunkedSpatialWeights,
  GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createKnnWeights,
  createSeededRandom,
  type CPUSpatialWeights
} from '../permutation-inference/permutation-oracle';

/**
 * Tile-size invariance of the weights6 chunk-aware consumers: LOSH, multivariate Local Geary,
 * Spatial Pearson and Gamma run with one CSR and with several chunkings (including one-row chunks)
 * and must agree. Per-row consumers are exact; the edge reductions differ only by f32 association.
 */

const ROW_COUNT = 150;

const CHUNKINGS: number[][] = [
  [75, 75],
  [40, 40, 40, 30],
  [7, ...Array.from({length: 13}, () => 11)],
  [1, 1, 1, 1, 1, 145]
];

function sliceWeights(weights: CPUSpatialWeights, rowStart: number, rowCount: number) {
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

const random = createSeededRandom(11);
const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
const values = Float32Array.from(
  {length: ROW_COUNT},
  (_, index) => Math.sin(positions[index * 2] / 9) + random()
);
const secondValues = Float32Array.from({length: ROW_COUNT}, () => random());
const stackedValues = Float32Array.from({length: ROW_COUNT * 2}, (_, index) =>
  index % 2 === 0 ? values[index / 2] : secondValues[(index - 1) / 2]
);
const mask = Uint32Array.from({length: ROW_COUNT}, (_, index) => (index % 11 === 4 ? 0 : 1));
const csr = createKnnWeights(positions, 5, false);

type Context = {
  graph: GPUCommandGraph;
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  output: <Format extends 'float32' | 'uint32'>(
    name: string,
    kind: Format,
    length: number
  ) => GraphDataView<Format>;
  input: {
    (name: string, data: Float32Array): GraphDataView<'float32'>;
    (name: string, data: Uint32Array): GraphDataView<'uint32'>;
  };
};

async function run(
  device: Device,
  chunkRowCounts: number[] | undefined,
  build: (context: Context) => void
): Promise<Record<string, number[]>> {
  const graph = new GPUCommandGraph(device, {id: 'weights6-parity'});
  const importCsr = (slice: CPUSpatialWeights, name: string, rowCount: number) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      createInputBuffer(device, slice.offsets),
      'uint32',
      rowCount + 1
    ),
    neighbors: importGraphBuffer(
      graph,
      `${name}-neighbors`,
      createInputBuffer(device, slice.neighbors),
      'uint32',
      slice.neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      `${name}-weights`,
      createInputBuffer(device, slice.weights),
      'float32',
      slice.weights.length
    )
  });
  const information = {
    rowCount: ROW_COUNT,
    transform: 'binary' as const,
    symmetric: 'unknown' as const
  };
  let weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  if (chunkRowCounts) {
    let rowStart = 0;
    weights = {
      rowCount: ROW_COUNT,
      information,
      chunks: chunkRowCounts.map((rowCount, index) => {
        const chunk = {
          rowStart,
          rowCount,
          weights: importCsr(sliceWeights(csr, rowStart, rowCount), `chunk${index}`, rowCount)
        };
        rowStart += rowCount;
        return chunk;
      })
    };
  } else {
    weights = {...importCsr(csr, 'whole', ROW_COUNT), information};
  }
  const outputs: {
    name: string;
    kind: string;
    length: number;
    buffer: ReturnType<typeof createOutputBuffer>;
  }[] = [];
  build({
    graph,
    weights,
    output: (name, kind, length) => {
      const buffer = createOutputBuffer(device, length);
      outputs.push({name, kind, length, buffer});
      return importGraphBuffer(graph, name, buffer, kind, length);
    },
    input: ((name: string, data: Float32Array | Uint32Array) =>
      importGraphBuffer(
        graph,
        name,
        createInputBuffer(device, data),
        data instanceof Float32Array ? 'float32' : 'uint32',
        data.length
      )) as never
  });
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: Record<string, number[]> = {};
  for (const output of outputs) {
    result[output.name] = Array.from(
      output.kind === 'float32'
        ? await readFloat32(output.buffer, output.length)
        : await readUint32(output.buffer, output.length)
    );
  }
  compiled.destroy();
  return result;
}

async function expectTileInvariant(
  build: (context: Context) => void,
  tolerance: number
): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const whole = await run(device, undefined, build);
  for (const chunking of CHUNKINGS) {
    const chunked = await run(device, chunking, build);
    for (const key of Object.keys(whole)) {
      expect(chunked[key].length, key).toBe(whole[key].length);
      whole[key].forEach((reference, index) => {
        const value = chunked[key][index];
        if (Number.isNaN(reference)) expect(Number.isNaN(value), `${key}[${index}]`).toBe(true);
        else
          expect(Math.abs(value - reference), `${key}[${index}] ${chunking}`).toBeLessThanOrEqual(
            tolerance * Math.max(1, Math.abs(reference))
          );
      });
    }
  }
}

it('chunked LOSH matches the single CSR for every tile size', async () => {
  for (const residualPower of [1, 2] as const) {
    await expectTileInvariant(({graph, weights, output, input}) => {
      graph.add(
        new GPULocalSpatialHeteroscedasticity({
          weights,
          values: input('values', values),
          mask: input('mask', mask),
          residualPower,
          output: {
            statistic: output('statistic', 'float32', ROW_COUNT),
            status: output('status', 'uint32', ROW_COUNT),
            normalizedResiduals: output('residuals', 'float32', ROW_COUNT)
          }
        })
      );
    }, 1e-6);
  }
});

it('chunked multivariate Local Geary matches the single CSR for every tile size', async () => {
  await expectTileInvariant(({graph, weights, output, input}) => {
    graph.add(
      new GPUMultivariateLocalGeary({
        weights,
        values: input('values', stackedValues),
        variableCount: 2,
        mask: input('mask', mask),
        output: {
          statistic: output('statistic', 'float32', ROW_COUNT),
          status: output('status', 'uint32', ROW_COUNT)
        }
      })
    );
  }, 1e-6);
});

it('chunked Spatial Pearson and Gamma match the single CSR for every tile size', async () => {
  await expectTileInvariant(({graph, weights, output, input}) => {
    graph.add(
      new GPUSpatialPearson({
        id: 'pearson',
        weights,
        values: input('values', values),
        secondValues: input('secondValues', secondValues),
        mask: input('mask', mask),
        output: {statistic: output('pearson', 'float32', 1)}
      })
    );
    for (const operation of [
      'cross-product',
      'squared-difference',
      'absolute-difference'
    ] as const) {
      graph.add(
        new GPUGammaStatistic({
          id: `gamma-${operation}`,
          weights,
          values: input(`values-${operation}`, values),
          mask: input(`mask-${operation}`, mask),
          operation,
          output: {statistic: output(`gamma-${operation}`, 'float32', 1)}
        })
      );
    }
  }, 1e-4);
});
