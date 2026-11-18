// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUChunkedSpatialWeightsAlgebra,
  GPUChunkedSpatialWeightsTranspose,
  GPUChunkedSpatialWeightsTransform,
  GPUChunkedSpatialWeightsValidation,
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsTranspose,
  GPUSpatialWeightsTransform,
  GPUSpatialWeightsValidation,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsTransformOperation
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom} from '../permutation-inference/permutation-oracle';
import {WeightsRig} from './spatial-weights-harness';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';

it('final2 rejects shared per-chunk loss words and preserves an earlier overflow with disjoint words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const input = {
      rowCount: 4,
      chunks: [0, 2].map((rowStart, index) => ({
        rowStart,
        rowCount: 2,
        weights: rig.uploadWeights({
          offsets: [0, 2, 4],
          neighbors: index ? [0, 1, 0, 1] : [2, 3, 2, 3],
          weights: [1, 1, 1, 1],
          distances: []
        })
      }))
    };
    const firstLoss = rig.output('uint32', 1);
    const secondLoss = rig.output('uint32', 1);
    const chunks = [1, 4].map((capacity, index) => ({
      rowStart: index * 2,
      rowCount: 2,
      weights: {
        ...rig.weightsOutput(2, capacity).spatialWeights,
        status: {overflow: firstLoss.view}
      }
    }));
    const props = {
      operation: 'subgraph' as const,
      mask: rig.input(Uint32Array.of(1, 1, 1, 1), 'uint32', 4),
      weights: input,
      output: {chunks}
    };
    expect(() => new GPUChunkedSpatialWeightsAlgebra(props)).toThrow(/overlap|share/);
    chunks[1].weights.status.overflow = secondLoss.view;
    const algebra = new GPUChunkedSpatialWeightsAlgebra(props);
    const aggregate = rig.output('uint32', 1);
    const summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
    rig.run(
      algebra,
      new GPUGlobalSpatialStatistics({
        weights: algebra.outputWeights,
        values: rig.input(Float32Array.of(1, 2, 4, 8), 'float32', 4),
        statistics: ['moran'],
        output: {summary: summary.view, status: {overflow: aggregate.view}}
      })
    );
    expect(await readUint32(firstLoss.buffer, 1)).toEqual([1]);
    expect(await readUint32(secondLoss.buffer, 1)).toEqual([0]);
    expect(await readUint32(aggregate.buffer, 1)).toEqual([1]);
    expect(
      Number.isNaN(
        (await readFloat32(summary.buffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length))[
          GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran
        ]
      )
    ).toBe(true);
  } finally {
    rig.destroy();
  }
});

it('final1 chunked algebra feeds aggregate completeness directly to global Moran for fitting and overflowing chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [6, 1]) {
    const rig = new WeightsRig(device);
    try {
      const input = rig.uploadWeights({
        offsets: [0, 1, 3, 5, 6],
        neighbors: [1, 0, 2, 1, 3, 2],
        weights: [1, 1, 1, 1, 1, 1],
        distances: []
      });
      const overflow = rig.output('uint32', 1);
      const algebra = new GPUChunkedSpatialWeightsAlgebra({
        operation: 'subgraph',
        mask: rig.input(new Uint32Array([1, 1, 1, 1]), 'uint32', 4),
        weights: {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights: input}]},
        output: {
          chunks: [
            {
              rowStart: 0,
              rowCount: 4,
              weights: {
                ...rig.weightsOutput(4, capacity).spatialWeights,
                status: {overflow: overflow.view}
              }
            }
          ]
        }
      });
      rig.graph.add(algebra);
      const summary = rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
      const matrixOverflow = rig.output('uint32', 1);
      rig.run(
        new GPUGlobalSpatialStatistics({
          weights: algebra.outputWeights,
          values: rig.input(new Float32Array([1, 2, 4, 8]), 'float32', 4),
          statistics: ['moran'],
          output: {summary: summary.view, status: {overflow: matrixOverflow.view}}
        })
      );
      expect(await readUint32(overflow.buffer, 1)).toEqual([Number(capacity < 6)]);
      expect(await readUint32(matrixOverflow.buffer, 1)).toEqual([Number(capacity < 6)]);
      const result = await readFloat32(summary.buffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
      if (capacity < 6)
        expect(Number.isNaN(result[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran])).toBe(true);
      else {
        expect(Number.isFinite(result[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran])).toBe(true);
        expect(result[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran]).toBeCloseTo(29 / 115, 5);
      }
    } finally {
      rig.destroy();
    }
  }
});

/**
 * WEIGHTS7: chunk-aware transpose, algebra, transform and validation produce the same CSR as the
 * whole-CSR classes for every chunking, including one-row chunks.
 */

const ROW_COUNT = 90;
const CHUNKINGS: number[][] = [
  [45, 45],
  [30, 30, 30],
  [7, ...Array.from({length: 11}, () => 7), 6],
  [1, 1, 1, 1, 86]
];

type CSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  distances: Float32Array;
};

/** Random CSR with k nearest neighbors (ascending IDs) and positive random weights. */
function createCSR(seed: number, k: number): CSR {
  const random = createSeededRandom(seed);
  const positions = Float64Array.from({length: ROW_COUNT * 2}, () => random() * 100);
  const offsets = new Uint32Array(ROW_COUNT + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  const distances: number[] = [];
  for (let row = 0; row < ROW_COUNT; row++) {
    const candidates: [number, number][] = [];
    for (let other = 0; other < ROW_COUNT; other++) {
      if (other !== row) {
        candidates.push([
          other,
          Math.hypot(
            positions[other * 2] - positions[row * 2],
            positions[other * 2 + 1] - positions[row * 2 + 1]
          )
        ]);
      }
    }
    candidates.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    // Rows 13 and 40 are empty to exercise empty rows at chunk edges.
    const picked = row === 13 || row === 40 ? [] : candidates.slice(0, k);
    for (const [other, distance] of picked.sort((a, b) => a[0] - b[0])) {
      neighbors.push(other);
      weights.push(0.5 + random());
      distances.push(distance);
    }
    offsets[row + 1] = neighbors.length;
  }
  return {
    offsets,
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights),
    distances: Float32Array.from(distances)
  };
}

const first = createCSR(5, 5);
const second = createCSR(9, 4);
// A corrupted copy: a self slot, an out-of-range ID, a duplicate and a negative weight.
const corrupted: CSR = (() => {
  const copy = {
    offsets: first.offsets.slice(),
    neighbors: first.neighbors.slice(),
    weights: first.weights.slice(),
    distances: first.distances.slice()
  };
  copy.neighbors[first.offsets[20]] = 20;
  copy.neighbors[first.offsets[50]] = 9999;
  copy.neighbors[first.offsets[70] + 1] = copy.neighbors[first.offsets[70]];
  copy.weights[first.offsets[85]] = -1;
  return copy;
})();
const mask = Uint32Array.from({length: ROW_COUNT}, (_, index) => (index % 7 === 3 ? 0 : 1));

function slice(csr: CSR, rowStart: number, rowCount: number): CSR {
  const begin = csr.offsets[rowStart];
  const end = csr.offsets[rowStart + rowCount];
  const stop = Math.max(end, begin + 1);
  return {
    offsets: Uint32Array.from(
      csr.offsets.subarray(rowStart, rowStart + rowCount + 1),
      value => value - begin
    ),
    neighbors: csr.neighbors.slice(begin, stop),
    weights: csr.weights.slice(begin, stop),
    distances: csr.distances.slice(begin, stop)
  };
}

type Output = {name: string; kind: 'float32' | 'uint32'; length: number; buffer: Buffer};

class Rig {
  readonly graph: GPUCommandGraph;
  readonly outputs: Output[] = [];
  constructor(readonly device: Device) {
    this.graph = new GPUCommandGraph(device, {id: 'weights7'});
  }
  input<Format extends 'float32' | 'uint32'>(
    name: string,
    data: Float32Array | Uint32Array,
    format: Format
  ): GraphDataView<Format> {
    return importGraphBuffer(
      this.graph,
      name,
      createInputBuffer(this.device, data),
      format,
      data.length
    );
  }
  output<Format extends 'float32' | 'uint32'>(
    name: string,
    kind: Format,
    length: number
  ): GraphDataView<Format> {
    const buffer = createOutputBuffer(this.device, length);
    this.outputs.push({name, kind, length, buffer});
    return importGraphBuffer(this.graph, name, buffer, kind, length);
  }
  importCSR(csr: CSR, name: string, withDistances = true): GPUSpatialWeights {
    return {
      offsets: this.input(`${name}-offsets`, csr.offsets, 'uint32'),
      neighbors: this.input(`${name}-neighbors`, csr.neighbors, 'uint32'),
      weights: this.input(`${name}-weights`, csr.weights, 'float32'),
      distances: withDistances
        ? this.input(`${name}-distances`, csr.distances, 'float32')
        : undefined
    };
  }
  /** Whole weights, or chunked weights when `chunking` is given. */
  weights(
    csr: CSR,
    name: string,
    chunking: number[] | undefined,
    information: GPUSpatialWeights['information'] = {
      rowCount: ROW_COUNT,
      transform: 'raw',
      symmetric: 'unknown'
    }
  ): GPUSpatialWeights | GPUChunkedSpatialWeights {
    if (!chunking) {
      return {...this.importCSR(csr, name), information};
    }
    let rowStart = 0;
    return {
      rowCount: ROW_COUNT,
      information,
      chunks: chunking.map((rowCount, index) => {
        const chunk = {
          rowStart,
          rowCount,
          weights: {
            ...this.importCSR(slice(csr, rowStart, rowCount), `${name}-chunk${index}`),
            information: {...information!, rowCount}
          }
        };
        rowStart += rowCount;
        return chunk;
      })
    };
  }
  async run(): Promise<Record<string, number[]>> {
    const compiled = this.graph.compile();
    submitGraph(this.device, compiled, undefined);
    const result: Record<string, number[]> = {};
    for (const output of this.outputs) {
      result[output.name] = Array.from(
        output.kind === 'float32'
          ? await readFloat32(output.buffer, output.length)
          : await readUint32(output.buffer, output.length)
      );
    }
    compiled.destroy();
    return result;
  }
}

/** Row lists `neighbor:weight` of a (possibly per-chunk) CSR read back from GPU arrays. */
function readRows(
  result: Record<string, number[]>,
  prefix: string,
  chunkRowCounts: number[] | undefined,
  hasDistances = false
): string[] {
  const rows: string[] = [];
  const pieces = chunkRowCounts ?? [ROW_COUNT];
  pieces.forEach((rowCount, index) => {
    const tag = chunkRowCounts ? `${prefix}-chunk${index}` : prefix;
    const offsets = result[`${tag}-offsets`];
    for (let row = 0; row < rowCount; row++) {
      const entries: string[] = [];
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        entries.push(
          `${result[`${tag}-neighbors`][slot]}:${result[`${tag}-weights`][slot].toFixed(6)}` +
            (hasDistances ? `:${result[`${tag}-distances`][slot].toFixed(4)}` : '')
        );
      }
      rows.push(entries.join(','));
    }
  });
  return rows;
}

async function compareToWhole<Value>(
  build: (rig: Rig, chunking: number[] | undefined) => void,
  read: (result: Record<string, number[]>, chunking: number[] | undefined) => Value
): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const wholeRig = new Rig(device);
  build(wholeRig, undefined);
  const whole = read(await wholeRig.run(), undefined);
  for (const chunking of CHUNKINGS) {
    const rig = new Rig(device);
    build(rig, chunking);
    expect(read(await rig.run(), chunking), `chunking ${chunking}`).toEqual(whole);
  }
}

function outputCSR(rig: Rig, name: string, rowCount: number, capacity: number) {
  return {
    offsets: rig.output(`${name}-offsets`, 'uint32', rowCount + 1),
    neighbors: rig.output(`${name}-neighbors`, 'uint32', capacity),
    weights: rig.output(`${name}-weights`, 'float32', capacity),
    distances: rig.output(`${name}-distances`, 'float32', capacity),
    status: {
      overflow: rig.output(`${name}-overflow`, 'uint32', 1),
      requiredCount: rig.output(`${name}-required`, 'uint32', 1)
    }
  };
}

it('chunked validation reports the same flags and invalid count at every chunking', async () => {
  await compareToWhole(
    (rig, chunking) => {
      const weights = rig.weights(corrupted, 'csr', chunking);
      const output = {
        flags: rig.output('flags', 'uint32', ROW_COUNT),
        status: {invalidCount: rig.output('invalid', 'uint32', 1)}
      };
      rig.graph.add(
        chunking
          ? new GPUChunkedSpatialWeightsValidation({
              weights: weights as GPUChunkedSpatialWeights,
              output
            })
          : new GPUSpatialWeightsValidation({weights: weights as GPUSpatialWeights, output})
      );
    },
    result => ({flags: result['flags'], invalid: result['invalid']})
  );
});

it('chunked transpose equals the whole transpose and detects asymmetry', async () => {
  await compareToWhole(
    (rig, chunking) => {
      const weights = rig.weights(first, 'csr', chunking);
      const capacity = first.neighbors.length + (chunking ? chunking.length : 0);
      const output = {
        ...outputCSR(rig, 'transpose', ROW_COUNT, capacity),
        asymmetricSlots: rig.output('asymmetric', 'uint32', 1)
      };
      rig.graph.add(
        chunking
          ? new GPUChunkedSpatialWeightsTranspose({
              weights: weights as GPUChunkedSpatialWeights,
              output
            })
          : new GPUSpatialWeightsTranspose({weights: weights as GPUSpatialWeights, output})
      );
    },
    result => {
      const used = result['transpose-offsets'][ROW_COUNT];
      return {
        offsets: result['transpose-offsets'],
        neighbors: result['transpose-neighbors'].slice(0, used),
        weights: result['transpose-weights'].slice(0, used),
        distances: result['transpose-distances'].slice(0, used),
        asymmetric: result['asymmetric']
      };
    }
  );
});

const TRANSFORMS: GPUSpatialWeightsTransformOperation[] = [
  'row',
  'binary',
  'kernel',
  'double',
  'variance',
  'symmetrize'
];

for (const operation of TRANSFORMS) {
  it(`chunked ${operation} transform equals the whole transform`, async () => {
    await compareToWhole(
      (rig, chunking) => {
        const weights = rig.weights(first, 'csr', chunking);
        const options = {operation, kernel: 'epanechnikov' as const, bandwidth: 60};
        if (chunking) {
          const chunked = weights as GPUChunkedSpatialWeights;
          const outputs = chunked.chunks.map((chunk, index) =>
            rig.output(`result-chunk${index}-weights`, 'float32', chunk.weights.neighbors.length)
          );
          rig.graph.add(
            new GPUChunkedSpatialWeightsTransform({...options, weights: chunked, output: outputs})
          );
        } else {
          const whole = weights as GPUSpatialWeights;
          rig.graph.add(
            new GPUSpatialWeightsTransform({
              ...options,
              weights: whole,
              output: rig.output('result-weights', 'float32', whole.neighbors.length)
            })
          );
        }
      },
      (result, chunking) => {
        // Transformed weights in global slot order; compare with a float32 association tolerance.
        const values = chunking
          ? chunking.flatMap((_, index) => result[`result-chunk${index}-weights`])
          : result['result-weights'];
        return {
          values: values.map(value => Math.round(value * 1e5) / 1e5)
        };
      }
    );
  });
}

it('chunked algebra equals the whole algebra for every row-local operation', async () => {
  const binaryOperations = ['union', 'intersection', 'difference', 'symmetricDifference'] as const;
  for (const operation of binaryOperations) {
    for (const weightRule of ['left', 'sum'] as const) {
      await compareToWhole(
        (rig, chunking) => {
          const left = rig.weights(first, 'left', chunking);
          const right = rig.weights(second, 'right', chunking);
          if (chunking) {
            const chunks = (left as GPUChunkedSpatialWeights).chunks.map((chunk, index) => {
              const capacity =
                chunk.weights.neighbors.length +
                (right as GPUChunkedSpatialWeights).chunks[index].weights.neighbors.length;
              return {
                rowStart: chunk.rowStart,
                rowCount: chunk.rowCount,
                weights: outputCSR(rig, `result-chunk${index}`, chunk.rowCount, capacity)
              };
            });
            rig.graph.add(
              new GPUChunkedSpatialWeightsAlgebra({
                operation,
                weightRule,
                left: left as GPUChunkedSpatialWeights,
                right: right as GPUChunkedSpatialWeights,
                output: {chunks}
              })
            );
          } else {
            rig.graph.add(
              new GPUSpatialWeightsAlgebra({
                operation,
                weightRule,
                left: left as GPUSpatialWeights,
                right: right as GPUSpatialWeights,
                output: outputCSR(
                  rig,
                  'result',
                  ROW_COUNT,
                  first.neighbors.length + second.neighbors.length
                )
              })
            );
          }
        },
        (result, chunking) => readRows(result, 'result', chunking, true)
      );
    }
  }
});

it('chunked self-weight and subgraph equal the whole operations', async () => {
  for (const operation of ['selfWeight', 'subgraph'] as const) {
    await compareToWhole(
      (rig, chunking) => {
        const weights = rig.weights(first, 'csr', chunking);
        const maskView = rig.input('mask', mask, 'uint32');
        const selfWeights = rig.input(
          'self',
          Float32Array.from({length: ROW_COUNT}, (_, index) => 1 + index / 100),
          'float32'
        );
        const extra = operation === 'selfWeight' ? {selfWeight: selfWeights} : {mask: maskView};
        if (chunking) {
          const chunks = (weights as GPUChunkedSpatialWeights).chunks.map((chunk, index) => ({
            rowStart: chunk.rowStart,
            rowCount: chunk.rowCount,
            weights: outputCSR(
              rig,
              `result-chunk${index}`,
              chunk.rowCount,
              chunk.weights.neighbors.length + chunk.rowCount
            )
          }));
          rig.graph.add(
            new GPUChunkedSpatialWeightsAlgebra({
              operation,
              weights: weights as GPUChunkedSpatialWeights,
              output: {chunks},
              ...extra
            } as never)
          );
        } else {
          rig.graph.add(
            new GPUSpatialWeightsAlgebra({
              operation,
              weights: weights as GPUSpatialWeights,
              output: outputCSR(rig, 'result', ROW_COUNT, first.neighbors.length + ROW_COUNT),
              ...extra
            } as never)
          );
        }
      },
      (result, chunking) => readRows(result, 'result', chunking, true)
    );
  }
});
