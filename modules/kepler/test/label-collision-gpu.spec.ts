// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {_GlobeViewport} from '@deck.gl/core';
import {WebMercatorViewport} from '@math.gl/web-mercator';
import {expect, test} from 'vitest';
import {
  createGpuLabelCollision,
  type GpuLabelCollisionUpdate
} from '../../../examples/deck/kepler-compute/src/label-collision-gpu';
import {computeLabelVisibilityWords} from '../../../examples/deck/kepler-compute/src/label-collision-runtime';

const STREAMED_CELL_SIZE = 4;
const VIEWPORT = {longitude: -122.4, latitude: 37.8, zoom: 11, width: 1000, height: 800};
const STYLE: Omit<GpuLabelCollisionUpdate, 'viewport'> = {
  size: 64,
  fontSize: 64,
  anchor: 'start',
  alignment: 'center',
  radius: 0,
  radiusPixelScale: 0,
  radiusPadding: 0,
  verticalPadding: 0
};

async function getDevice() {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for GPU label collision verification');
  return device;
}

function makeFixture(
  device: Awaited<ReturnType<typeof getDevice>>,
  positionRows: Float32Array,
  boundsRows: Float32Array,
  chunkRowCounts: number[],
  priorityRows?: Float32Array
) {
  const buffers: Buffer[] = [];
  const makeBuffer = (data: Float32Array) => {
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  };
  const positionChunks: GPUData<'float32x2'>[] = [];
  const boundsChunks: GPUData<'float32x4'>[] = [];
  const priorityChunks: GPUData<'float32'>[] = [];
  let rowBase = 0;
  for (const length of chunkRowCounts) {
    positionChunks.push(
      new GPUData({
        format: 'float32x2',
        buffer: makeBuffer(positionRows.slice(rowBase * 2, (rowBase + length) * 2)),
        length,
        ownsBuffer: false
      })
    );
    boundsChunks.push(
      new GPUData({
        format: 'float32x4',
        buffer: makeBuffer(boundsRows.slice(rowBase * 4, (rowBase + length) * 4)),
        length,
        ownsBuffer: false
      })
    );
    if (priorityRows)
      priorityChunks.push(
        new GPUData({
          format: 'float32',
          buffer: makeBuffer(priorityRows.slice(rowBase, rowBase + length)),
          length,
          ownsBuffer: false
        })
      );
    rowBase += length;
  }
  const positions = new GPUVector({
    type: 'data',
    name: 'positions',
    format: 'float32x2',
    ownsData: false,
    data: positionChunks
  });
  const priorities = priorityRows
    ? new GPUVector({
        type: 'data',
        name: 'priorities',
        format: 'float32',
        ownsData: false,
        data: priorityChunks
      })
    : undefined;
  return {
    positions,
    rowBounds: boundsChunks,
    priorities,
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

function unprojectPixels(pixels: number[][]): Float32Array {
  const projector = new WebMercatorViewport({...VIEWPORT, pitch: 0, bearing: 0});
  const rows = new Float32Array(pixels.length * 2);
  pixels.forEach(([x, y], index) => {
    const [longitude, latitude] = projector.unproject([x, y]);
    rows[index * 2] = longitude;
    rows[index * 2 + 1] = latitude;
  });
  return rows;
}

function listAccepted(words: Uint32Array, rowCount: number): number[] {
  const rows: number[] = [];
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++)
    if (words[rowIndex >>> 5] & (1 << (rowIndex & 31))) rows.push(rowIndex);
  return rows;
}

test('GPU label collision keeps the highest priority label of each overlapping group', async () => {
  const device = await getDevice();
  // Anchor pixels (label starts at the anchor, vertically centered): A and B overlap, C is clear,
  // D is far outside the viewport margin and stays hidden.
  const positions = unprojectPixels([
    [100, 100],
    [120, 105],
    [500, 400],
    [5000, 5000]
  ]);
  const bounds = new Float32Array([60, 20, 0, 0, 60, 20, 0, 0, 60, 20, 0, 0, 60, 20, 0, 0]);
  const run = async (priorities?: Float32Array) => {
    const fixture = makeFixture(device, positions, bounds, [2, 2], priorities);
    const collision = createGpuLabelCollision({
      device,
      positions: fixture.positions,
      rowBounds: fixture.rowBounds,
      priorities: fixture.priorities,
      priorityRange: [0, 10]
    });
    try {
      collision.update({viewport: VIEWPORT, ...STYLE});
      return listAccepted(await collision.readVisibilityWords(), 4);
    } finally {
      collision.destroy();
      fixture.destroy();
    }
  };
  // Without priorities the lower row index wins ties.
  expect(await run()).toEqual([0, 2]);
  // A higher priority on the later overlapping label flips the winner.
  expect(await run(new Float32Array([1, 9, 1, 1]))).toEqual([1, 2]);
  device.destroy();
});

test('GPU label collision above 100k rows never accepts overlapping labels', async () => {
  const device = await getDevice();
  const rowCount = 150000;
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const pixels: number[][] = [];
  const bounds = new Float32Array(rowCount * 4);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    pixels.push([random() * VIEWPORT.width, random() * VIEWPORT.height]);
    bounds[rowIndex * 4] = 30 + random() * 50;
    bounds[rowIndex * 4 + 1] = 14 + random() * 8;
  }
  const positions = unprojectPixels(pixels);
  const fixture = makeFixture(device, positions, bounds, [100000, 50000]);
  const collision = createGpuLabelCollision({
    device,
    positions: fixture.positions,
    rowBounds: fixture.rowBounds
  });
  try {
    collision.update({viewport: VIEWPORT, ...STYLE});
    const accepted = listAccepted(await collision.readVisibilityWords(), rowCount);
    expect(accepted.length).toBeGreaterThan(100);
    expect(accepted.length).toBeLessThan(rowCount);
    // CPU oracle over the readback: project independently and compare every accepted pair.
    const projector = new WebMercatorViewport({...VIEWPORT, pitch: 0, bearing: 0});
    const rects = accepted.map(rowIndex => {
      const [x, y] = projector.project([positions[rowIndex * 2], positions[rowIndex * 2 + 1]]);
      const height = bounds[rowIndex * 4 + 1];
      return {
        left: x,
        top: y - height / 2,
        right: x + bounds[rowIndex * 4],
        bottom: y + height / 2
      };
    });
    const grid = new Map<string, number[]>();
    let overlaps = 0;
    rects.forEach((rect, index) => {
      const cellKeys: string[] = [];
      for (let row = Math.floor(rect.top / 64); row <= Math.floor(rect.bottom / 64); row++)
        for (
          let column = Math.floor(rect.left / 64);
          column <= Math.floor(rect.right / 64);
          column++
        )
          cellKeys.push(`${column},${row}`);
      const checked = new Set<number>();
      for (const key of cellKeys)
        for (const otherIndex of grid.get(key) ?? []) {
          if (checked.has(otherIndex)) continue;
          checked.add(otherIndex);
          const other = rects[otherIndex];
          if (
            rect.left < other.right - 0.01 &&
            rect.right > other.left + 0.01 &&
            rect.top < other.bottom - 0.01 &&
            rect.bottom > other.top + 0.01
          )
            overlaps++;
        }
      for (const key of cellKeys) grid.set(key, [...(grid.get(key) ?? []), index]);
    });
    expect(overlaps).toBe(0);
  } finally {
    collision.destroy();
    fixture.destroy();
    device.destroy();
  }
}, 120000);

test('converged GPU collision keeps nearly as many labels as the CPU greedy result', async () => {
  const device = await getDevice();
  const rowCount = 12000;
  let seed = 777;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const pixels: number[][] = [];
  const bounds = new Float32Array(rowCount * 4);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    pixels.push([random() * VIEWPORT.width, random() * VIEWPORT.height]);
    bounds[rowIndex * 4] = 30 + random() * 50;
    bounds[rowIndex * 4 + 1] = 14 + random() * 8;
  }
  const positions = unprojectPixels(pixels);
  const fixture = makeFixture(device, positions, bounds, [8000, 4000]);
  const update = {viewport: VIEWPORT, ...STYLE};
  const cpu = computeLabelVisibilityWords({
    rowCount,
    anchors: {positions},
    rowBounds: bounds,
    viewport: VIEWPORT,
    size: STYLE.size,
    fontSize: STYLE.fontSize,
    anchor: STYLE.anchor,
    alignment: STYLE.alignment,
    radius: 0,
    radiusPixelScale: 0,
    radiusPadding: 0,
    verticalPadding: 0,
    maximumCandidates: rowCount
  });
  const cpuCount = listAccepted(cpu, rowCount).length;
  try {
    for (const cellSize of [4, 8, 16]) {
      const collision = createGpuLabelCollision({
        device,
        positions: fixture.positions,
        rowBounds: fixture.rowBounds,
        cellSize
      });
      collision.update(update);
      const fixedCount = listAccepted(await collision.readVisibilityWords(), rowCount).length;
      const result = await collision.updateUntilStable(update);
      const stableCount = listAccepted(await collision.readVisibilityWords(), rowCount).length;
      console.log(
        `label collision cell=${cellSize}: cpu=${cpuCount} gpu6=${fixedCount} gpuStable=${stableCount} rounds=${result.rounds}`
      );
      expect(result.accepted).toBe(stableCount);
      expect(stableCount).toBeGreaterThanOrEqual(fixedCount);
      if (cellSize === STREAMED_CELL_SIZE)
        expect(stableCount).toBeGreaterThanOrEqual(cpuCount * 0.9);
      collision.destroy();
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
}, 120000);

test('GPU label collision projects on the globe and hides the far hemisphere', async () => {
  const device = await getDevice();
  const globeViewport = {
    longitude: 0,
    latitude: 0,
    zoom: 1,
    width: 1000,
    height: 800,
    pitch: 0,
    bearing: 0
  };
  const projector = new _GlobeViewport(globeViewport);
  // A and B are 6 pixels apart on the near side (overlap), C is clear, D is on the far side.
  const nearPixels = [
    [300, 400],
    [306, 402],
    [700, 300]
  ];
  const near = nearPixels.flatMap(([x, y]) => projector.unproject([x, y]).slice(0, 2));
  const positions = new Float32Array([...near, 180, 0]);
  const bounds = new Float32Array(16).map((_value, index) => [60, 20, 0, 0][index % 4]);
  const fixture = makeFixture(device, positions, bounds, [4]);
  const collision = createGpuLabelCollision({
    device,
    positions: fixture.positions,
    rowBounds: fixture.rowBounds
  });
  try {
    collision.update({viewport: {...globeViewport, globe: true}, ...STYLE});
    expect(listAccepted(await collision.readVisibilityWords(), 4)).toEqual([0, 2]);
  } finally {
    collision.destroy();
    fixture.destroy();
    device.destroy();
  }
}, 120000);
