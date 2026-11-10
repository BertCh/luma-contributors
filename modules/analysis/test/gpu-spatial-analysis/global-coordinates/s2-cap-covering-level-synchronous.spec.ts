// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import type {GPUS2CapCoveringRefinement} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Result = {cells: number[]; counts: number[]; status: number[]};

async function executeCaps(
  device: Device,
  caps: number[],
  options: {
    level: number;
    capacity: number;
    mode: 'exterior' | 'interior';
    refinement: GPUS2CapCoveringRefinement;
    frontierCapacity?: number;
  }
): Promise<Result> {
  const capCount = caps.length / 4;
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const allocate = (bytes: number) => {
    const buffer = device.createBuffer({
      byteLength: bytes,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    buffers.push(buffer);
    return buffer;
  };
  const capsBuffer = allocate(caps.length * 4);
  capsBuffer.write(new Float32Array(caps));
  const cells = allocate(capCount * options.capacity * 8);
  const counts = allocate(capCount * 8);
  const status = allocate(capCount * 4);
  graph.add(
    new GPUS2CapCovering({
      caps: importGraphBuffer(graph, 'caps', capsBuffer, 'float32x4', capCount),
      level: options.level,
      mode: options.mode,
      capacity: options.capacity,
      refinement: options.refinement,
      frontierCapacity: options.frontierCapacity,
      output: {
        cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', capCount * options.capacity),
        counts: importGraphBuffer(graph, 'counts', counts, 'uint32x2', capCount),
        status: importGraphBuffer(graph, 'status', status, 'uint32', capCount)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      cells: await readUint32(cells, capCount * options.capacity * 2),
      counts: await readUint32(counts, capCount * 2),
      status: await readUint32(status, capCount)
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

const CAPS = [
  // Centers on a face center, an edge, a corner, and in general position.
  ...[1, 0, 0, 0.02],
  ...[1, 0, 0, 0.3],
  ...[0.7071068, 0.7071068, 0, 0.25],
  ...[0.5773503, 0.5773503, 0.5773503, 0.41],
  ...[0, 0, -1, 0.9],
  ...[0.3, -0.5, 0.8124038, 0.0001],
  ...[-0.8001536, 0.3600691, 0.4796921, 1.3],
  ...[0.3711337, 0.9127883, 0.1705209, 0],
  ...[0, 0, 1, Math.PI],
  ...[0, 0, 1, Math.PI - 0.0001],
  // Invalid: center is not a unit vector.
  ...[2, 0, 0, 0.1]
];
const CAP_COUNT = CAPS.length / 4;

it('level-synchronous S2 cap covering equals the uniform enumeration for levels 0 to 5', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['exterior', 'interior'] as const)
    for (const level of [0, 1, 3, 5]) {
      const capacity = Math.min(6 * 4 ** level, 1500);
      const uniform = await executeCaps(device, CAPS, {
        level,
        capacity,
        mode,
        refinement: 'uniform'
      });
      const refined = await executeCaps(device, CAPS, {
        level,
        capacity,
        mode,
        refinement: 'level-synchronous',
        frontierCapacity: 6 * 4 ** level
      });
      const note = `${mode} level ${level}`;
      expect(refined.status, note).toEqual(uniform.status);
      expect(refined.counts, note).toEqual(uniform.counts);
      expect(refined.cells, note).toEqual(uniform.cells);
      expect(uniform.status[CAP_COUNT - 1], note).toBe(1);
    }
});

it('level-synchronous S2 cap covering reports required counts past capacity like the uniform enumeration', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const level = 4;
  const capacity = 5;
  const uniform = await executeCaps(device, CAPS, {
    level,
    capacity,
    mode: 'exterior',
    refinement: 'uniform'
  });
  // The wide caps need more than the default frontier holds at capacity 5, so give the
  // frontier room for the largest cover: the required count then stays exact.
  const refined = await executeCaps(device, CAPS, {
    level,
    capacity,
    mode: 'exterior',
    refinement: 'level-synchronous',
    frontierCapacity: 6 * 4 ** level
  });
  expect(refined).toEqual(uniform);
  expect(uniform.status.some(word => (word & 2) !== 0)).toBe(true);
});

it('level-synchronous S2 cap covering flags a frontier overflow instead of returning a partial cover', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const capacity = 64;
  const refined = await executeCaps(device, [1, 0, 0, 0.4, 1, 0, 0, 0.001], {
    level: 6,
    capacity,
    mode: 'exterior',
    refinement: 'level-synchronous',
    frontierCapacity: 64
  });
  // The wide cap needs far more than 64 cells; the tiny one fits.
  expect(refined.status[0]! & 4).toBe(4);
  expect(refined.counts.slice(0, 2)).toEqual([0, 0]);
  expect(refined.cells.slice(0, capacity * 2).every(word => word === 0)).toBe(true);
  expect(refined.status[1]).toBe(0);
  expect(refined.counts[2]).toBeGreaterThan(0);
});

it('level-synchronous S2 cap covering reaches level 11, beyond the uniform tier, and refines the level 8 cover', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cap = [0.3, -0.5, 0.8124038, 0.02];
  const coarse = await executeCaps(device, cap, {
    level: 8,
    capacity: 4096,
    mode: 'exterior',
    refinement: 'uniform'
  });
  const fine = await executeCaps(device, cap, {
    level: 11,
    capacity: 65536,
    mode: 'exterior',
    refinement: 'level-synchronous',
    frontierCapacity: 70000
  });
  expect(coarse.status).toEqual([0]);
  expect(fine.status).toEqual([0]);
  const toBigInt = (words: number[], index: number) =>
    BigInt(words[index * 2]!) + (BigInt(words[index * 2 + 1]!) << 32n);
  const coarseCells = new Set(
    Array.from({length: coarse.counts[0]!}, (_, index) => toBigInt(coarse.cells, index))
  );
  const fineIdentifiers = Array.from({length: fine.counts[0]!}, (_, index) =>
    toBigInt(fine.cells, index)
  );
  expect(fineIdentifiers.length).toBeGreaterThan(coarseCells.size);
  const shift = BigInt(2 * (30 - 8) + 1);
  for (const identifier of fineIdentifiers) {
    const ancestor = ((identifier >> shift) << shift) | (1n << (shift - 1n));
    expect(coarseCells.has(ancestor)).toBe(true);
  }
  // Ascending ID order, no duplicates.
  for (let index = 1; index < fineIdentifiers.length; index++) {
    expect(fineIdentifiers[index]! > fineIdentifiers[index - 1]!).toBe(true);
  }
});

it('a zero-radius cap at a face center covers the four cells that touch it (uniform enumeration)', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const refinement of ['uniform', 'level-synchronous'] as const) {
    const result = await executeCaps(device, [0, 1, 0, 0], {
      level: 1,
      capacity: 24,
      mode: 'exterior',
      refinement,
      frontierCapacity: refinement === 'uniform' ? undefined : 24
    });
    expect(result.status, refinement).toEqual([0]);
    expect(result.counts, refinement).toEqual([4, 4]);
  }
});

it('level-synchronous S2 cap covering with the default frontier reports the required count when the cover exceeds capacity', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cap = [1, 0, 0, 0.3];
  const level = 4;
  for (const mode of ['exterior', 'interior'] as const) {
    const wide = await executeCaps(device, cap, {
      level,
      capacity: 6 * 4 ** level,
      mode,
      refinement: 'uniform'
    });
    const required = wide.counts[1]!;
    expect(required, mode).toBeGreaterThan(8);
    // A capacity at half the cover: the default frontier (no frontierCapacity prop) holds it.
    const capacity = Math.ceil(required / 2);
    const uniform = await executeCaps(device, cap, {
      level,
      capacity,
      mode,
      refinement: 'uniform'
    });
    const refined = await executeCaps(device, cap, {
      level,
      capacity,
      mode,
      refinement: 'level-synchronous'
    });
    expect(uniform.status, mode).toEqual([2]);
    expect(refined.status, mode).toEqual([2]);
    expect(refined.counts, mode).toEqual([capacity, required]);
    expect(refined.cells, mode).toEqual(uniform.cells);
  }
});

it('level-synchronous S2 cap covering with the default frontier flags an overflow that cannot report a required count', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Capacity 2 gives a default frontier of 68 cells, far below the cover of this wide cap.
  const refined = await executeCaps(device, [1, 0, 0, 0.9], {
    level: 5,
    capacity: 2,
    mode: 'exterior',
    refinement: 'level-synchronous'
  });
  expect(refined.status[0]! & 4).toBe(4);
  expect(refined.counts).toEqual([0, 0]);
  expect(refined.cells.every(word => word === 0)).toBe(true);
});
