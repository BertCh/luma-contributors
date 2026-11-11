// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it, expect} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';

const ROWS = 3000;
const NONE = 0xffffffff;

/** Deterministic linear congruential stream in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

/**
 * Random sorted CSR over integer positions (exact float32 distances, so ties are real ties):
 * about a sixth of the rows are islands, and some edges are one-directional so islands can
 * already appear in other rows.
 */
function createInput() {
  const random = createRandom(2026);
  const positions: number[] = [];
  for (let row = 0; row < ROWS; row++)
    positions.push(Math.floor(random() * 200), Math.floor(random() * 200));
  const rowsNeighbors: Map<number, number>[] = Array.from({length: ROWS}, () => new Map());
  for (let row = 0; row < ROWS; row++) {
    if (random() < 1 / 6) continue;
    const count = 1 + Math.floor(random() * 4);
    for (let edge = 0; edge < count; edge++) {
      const other = Math.floor(random() * ROWS);
      if (other === row) continue;
      const weight = 0.25 + Math.floor(random() * 8) / 8;
      rowsNeighbors[row].set(other, weight);
      if (random() < 0.8 && rowsNeighbors[other].size > 0) rowsNeighbors[other].set(row, weight);
    }
  }
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (const row of rowsNeighbors) {
    for (const other of [...row.keys()].sort((left, right) => left - right)) {
      neighbors.push(other);
      weights.push(row.get(other) as number);
    }
    offsets.push(neighbors.length);
  }
  return {positions, offsets, neighbors, weights};
}

/** Brute-force reference of the documented semantics, scanning every row pair. */
function getReference(input: ReturnType<typeof createInput>, reciprocal: boolean) {
  const {positions, offsets, neighbors, weights} = input;
  const attachment = new Array(ROWS).fill(NONE);
  for (let row = 0; row < ROWS; row++) {
    if (offsets[row] !== offsets[row + 1]) continue;
    let best = Infinity;
    for (let other = 0; other < ROWS; other++) {
      if (other === row) continue;
      const distance =
        (positions[2 * row] - positions[2 * other]) ** 2 +
        (positions[2 * row + 1] - positions[2 * other + 1]) ** 2;
      if (distance < best) {
        best = distance;
        attachment[row] = other;
      }
    }
  }
  const rows: {neighbor: number; weight: number; added: number}[][] = [];
  for (let row = 0; row < ROWS; row++) {
    const edges: {neighbor: number; weight: number; added: number}[] = [];
    for (let other = 0; other < ROWS; other++) {
      let original = -1;
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++)
        if (neighbors[slot] === other) original = slot;
      const added =
        other !== row && (attachment[row] === other || (reciprocal && attachment[other] === row));
      if (original >= 0) edges.push({neighbor: other, weight: weights[original], added: 0});
      else if (added) edges.push({neighbor: other, weight: 1, added: 1});
    }
    rows.push(edges);
  }
  return {attachment, rows};
}

for (const reciprocal of [true, false])
  for (const truncated of [false, true])
    it(`island attachment merge matches a brute-force reference (reciprocal ${reciprocal}, truncated ${truncated})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        const input = createInput();
        const reference = getReference(input, reciprocal);
        const flat = reference.rows.flat();
        const required = flat.length;
        const capacity = truncated ? Math.floor(required / 2) : required;
        let serial = 0;
        const upload = <Format extends 'float32' | 'float32x2' | 'uint32'>(
          format: Format,
          data: number[]
        ) => {
          const buffer = createInputBuffer(
            device,
            format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
          );
          buffers.push(buffer);
          const length = format === 'float32x2' ? data.length / 2 : data.length;
          return importGraphBuffer(graph, `input-${serial++}`, buffer, format, length);
        };
        const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
          const buffer = createOutputBuffer(device, length);
          buffers.push(buffer);
          return {
            buffer,
            view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)
          };
        };
        const offsets = output('uint32', ROWS + 1);
        const neighbors = output('uint32', capacity);
        const weights = output('float32', capacity);
        const attachment = output('uint32', ROWS);
        const added = output('uint32', capacity);
        const overflow = output('uint32', 1);
        const requiredCount = output('uint32', 1);
        graph.add(
          new GPUIslandAttachment({
            weights: {
              offsets: upload('uint32', input.offsets),
              neighbors: upload('uint32', input.neighbors),
              weights: upload('float32', input.weights)
            },
            positions: upload('float32x2', input.positions),
            reciprocal,
            output: {
              offsets: offsets.view,
              neighbors: neighbors.view,
              weights: weights.view,
              attachment: attachment.view,
              added: added.view,
              status: {overflow: overflow.view, requiredCount: requiredCount.view}
            }
          })
        );
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        expect(await readUint32(attachment.buffer, ROWS)).toEqual(reference.attachment);
        expect(await readUint32(requiredCount.buffer, 1)).toEqual([required]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([truncated ? 1 : 0]);
        const expectedOffsets = [0];
        for (const row of reference.rows)
          expectedOffsets.push(expectedOffsets[expectedOffsets.length - 1] + row.length);
        expect(await readUint32(offsets.buffer, ROWS + 1)).toEqual(
          expectedOffsets.map(offset => Math.min(offset, capacity))
        );
        const written = flat.slice(0, capacity);
        expect(await readUint32(neighbors.buffer, capacity)).toEqual(
          written.map(edge => edge.neighbor)
        );
        expect(await readFloat32(weights.buffer, capacity)).toEqual(
          written.map(edge => edge.weight)
        );
        expect(await readUint32(added.buffer, capacity)).toEqual(written.map(edge => edge.added));
        expect(reference.attachment.filter(value => value !== NONE).length).toBeGreaterThan(100);
      } finally {
        compiled?.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
