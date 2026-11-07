// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
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
import type {CPUSpatialWeights} from '../permutation-inference/permutation-oracle';

/** Chunk row counts that exercise equal, uneven and one-row chunks for `rowCount` rows. */
export function getWeights7Chunkings(rowCount: number): number[][] {
  const half = Math.floor(rowCount / 2);
  const chunkings = [
    [half, rowCount - half],
    [7, ...Array.from({length: Math.floor((rowCount - 7) / 11)}, () => 11)],
    [1, 1, 1, rowCount - 3]
  ];
  chunkings[1].push(rowCount - chunkings[1].reduce((total, count) => total + count, 0));
  return chunkings.map(chunking => chunking.filter(count => count > 0));
}

/** Slices rows `[rowStart, rowStart + rowCount)` of a CSR into a chunk CSR with local offsets. */
export function sliceWeights7Csr(weights: CPUSpatialWeights, rowStart: number, rowCount: number) {
  const begin = weights.offsets[rowStart];
  const end = weights.offsets[rowStart + rowCount];
  return {
    offsets: Uint32Array.from(
      weights.offsets.subarray(rowStart, rowStart + rowCount + 1),
      value => value - begin
    ),
    // A zero-slot chunk still needs a non-empty buffer.
    neighbors: weights.neighbors.slice(begin, Math.max(end, begin + 1)),
    weights: weights.weights.slice(begin, Math.max(end, begin + 1))
  };
}

export type Weights7Context = {
  graph: GPUCommandGraph;
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  output: <Format extends 'float32' | 'uint32' | 'sint32'>(
    name: string,
    kind: Format,
    length: number
  ) => GraphDataView<Format>;
  input: {
    (name: string, data: Float32Array): GraphDataView<'float32'>;
    (name: string, data: Uint32Array): GraphDataView<'uint32'>;
  };
};

export async function runWeights7(
  device: Device,
  csr: CPUSpatialWeights,
  chunkRowCounts: number[] | undefined,
  build: (context: Weights7Context) => void
): Promise<Record<string, number[]>> {
  const rowCount = csr.offsets.length - 1;
  const graph = new GPUCommandGraph(device, {id: 'weights7-parity'});
  const importCsr = (slice: CPUSpatialWeights, name: string, chunkRows: number) => ({
    offsets: importGraphBuffer(
      graph,
      `${name}-offsets`,
      createInputBuffer(device, slice.offsets),
      'uint32',
      chunkRows + 1
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
  const information = {rowCount, transform: 'binary' as const, symmetric: 'unknown' as const};
  let weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  if (chunkRowCounts) {
    let rowStart = 0;
    weights = {
      rowCount,
      information,
      chunks: chunkRowCounts.map((chunkRows, index) => {
        const chunk = {
          rowStart,
          rowCount: chunkRows,
          weights: importCsr(sliceWeights7Csr(csr, rowStart, chunkRows), `chunk${index}`, chunkRows)
        };
        rowStart += chunkRows;
        return chunk;
      })
    };
  } else {
    weights = {...importCsr(csr, 'whole', rowCount), information};
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

/**
 * Runs `build` with the whole CSR and with every chunking and requires the outputs to agree within
 * `tolerance` (relative to max(1, |reference|)); NaN must match NaN.
 */
export async function expectWeights7ChunkInvariant(
  csr: CPUSpatialWeights,
  build: (context: Weights7Context) => void,
  tolerance: number,
  chunkings = getWeights7Chunkings(csr.offsets.length - 1)
): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const whole = await runWeights7(device, csr, undefined, build);
  for (const chunking of chunkings) {
    const chunked = await runWeights7(device, csr, chunking, build);
    for (const key of Object.keys(whole)) {
      expect(chunked[key].length, key).toBe(whole[key].length);
      whole[key].forEach((reference, index) => {
        const value = chunked[key][index];
        if (Number.isNaN(reference)) {
          expect(Number.isNaN(value), `${key}[${index}]`).toBe(true);
        } else {
          expect(Math.abs(value - reference), `${key}[${index}] ${chunking}`).toBeLessThanOrEqual(
            tolerance * Math.max(1, Math.abs(reference))
          );
        }
      });
    }
  }
}

/** Placeholder chunked weights over transient views for CPU-only graph-structure specs. */
export function createWeights7TransientChunks(
  graph: GPUCommandGraph,
  chunkRowCounts: number[],
  slotsPerRow = 4
): GPUChunkedSpatialWeights {
  let rowStart = 0;
  const rowCount = chunkRowCounts.reduce((total, count) => total + count, 0);
  return {
    rowCount,
    information: {rowCount, transform: 'binary', symmetric: 'unknown'},
    chunks: chunkRowCounts.map((chunkRows, index) => {
      const slots = chunkRows * slotsPerRow;
      const chunk = {
        rowStart,
        rowCount: chunkRows,
        weights: {
          offsets: createTransientView(graph, `offsets-${index}`, 'uint32', chunkRows + 1),
          neighbors: createTransientView(graph, `neighbors-${index}`, 'uint32', slots),
          weights: createTransientView(graph, `weights-${index}`, 'float32', slots)
        }
      };
      rowStart += chunkRows;
      return chunk;
    })
  };
}
