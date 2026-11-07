// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {GPUKeplerSpatialSelection, type KeplerSpatialSelection} from '../src/selection';

const POLYGON: KeplerSpatialSelection = {
  kind: 'polygon',
  polygons: [
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ],
      [
        [1, 1],
        [3, 1],
        [3, 3],
        [1, 3]
      ]
    ],
    [
      [
        [8, 8],
        [10, 8],
        [10, 10],
        [8, 10]
      ]
    ]
  ]
};

function makePositions(device: Device, chunks: number[][]): GPUVector<'float32x2'> {
  return new GPUVector({
    type: 'data',
    name: 'source-points',
    format: 'float32x2',
    data: chunks.map(
      values =>
        new GPUData({
          buffer: device.createBuffer({
            ...(values.length ? {data: Float32Array.from(values)} : {byteLength: 8}),
            usage: Buffer.STORAGE | Buffer.COPY_DST
          }),
          format: 'float32x2',
          length: values.length / 2,
          ownsBuffer: true
        })
    )
  });
}

async function readMask(selection: GPUKeplerSpatialSelection): Promise<number[]> {
  const chunks = await Promise.all(
    selection.mask.data.map(async chunk => {
      if (chunk.length === 0) return [];
      const bytes = await chunk.buffer.readAsync(0, chunk.length * 4);
      return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
    })
  );
  return chunks.flat();
}

function submit(device: Device, selection: GPUKeplerSpatialSelection): void {
  const encoder = device.createCommandEncoder();
  selection.encode(encoder);
  device.submit(encoder.finish());
}

it('GPU Kepler selection preserves point order across chunks, holes, boundaries and union parts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makePositions(device, [[0.5, 0.5, 2, 2, 0, 2], [], [1, 2, 5, 5, 9, 9, NaN, 0]]);
  const closed = new GPUKeplerSpatialSelection({device, positions, selection: POLYGON});
  const open = new GPUKeplerSpatialSelection({
    device,
    positions,
    selection: POLYGON,
    boundaryRule: 'open'
  });
  try {
    submit(device, closed);
    expect(await readMask(closed)).toEqual([1, 0, 1, 1, 0, 1, 0]);
    expect(await closed.readStatus()).toMatchObject({
      revision: 0,
      complete: true,
      overflow: 0,
      uncertainCount: 0
    });
    submit(device, open);
    expect(await readMask(open)).toEqual([1, 0, 0, 0, 0, 1, 0]);
    expect(positions.data.map(chunk => chunk.length)).toEqual([3, 0, 4]);
  } finally {
    closed.destroy();
    open.destroy();
    positions.destroy();
  }
});

it('GPU Kepler rectangle updates reuse mask storage and update selections without source upload', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makePositions(device, [
    [0, 0, 2, 2],
    [4, 4, 6, 6]
  ]);
  const selection = new GPUKeplerSpatialSelection({
    device,
    positions,
    selection: {kind: 'rectangle', bounds: [-1, -1, 3, 3]}
  });
  const originalMasks = selection.mask.data.map(chunk => chunk.buffer);
  try {
    submit(device, selection);
    expect(await readMask(selection)).toEqual([1, 1, 0, 0]);
    selection.updateSelection({kind: 'rectangle', bounds: [3, 3, 7, 7]});
    expect(await selection.readStatus()).toMatchObject({stale: true, complete: false});
    submit(device, selection);
    expect(await readMask(selection)).toEqual([0, 0, 1, 1]);
    expect(await selection.readStatus()).toMatchObject({revision: 1, stale: false, complete: true});
    expect(selection.mask.data.map(chunk => chunk.buffer)).toEqual(originalMasks);
  } finally {
    selection.destroy();
    positions.destroy();
  }
});

it('GPU Kepler empty selection inputs produce empty masks with complete diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makePositions(device, [[]]);
  const selection = new GPUKeplerSpatialSelection({
    device,
    positions,
    selection: {kind: 'rectangle', bounds: [0, 0, 1, 1]}
  });
  try {
    submit(device, selection);
    expect(await readMask(selection)).toEqual([]);
    expect(await selection.readStatus()).toMatchObject({complete: true, candidateCount: 0});
  } finally {
    selection.destroy();
    positions.destroy();
  }
});

it('GPU Kepler selection handles sliced shared source buffers across multiple workgroups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = new Float32Array((513 + 2) * 2);
  values.set([-100, -100]);
  const expected: number[] = [];
  for (let rowIndex = 0; rowIndex < 513; rowIndex++) {
    const coordinateX = Math.fround((rowIndex % 27) / 3 - 1);
    const coordinateY = Math.fround(Math.floor(rowIndex / 27) / 3 - 1);
    values.set([coordinateX, coordinateY], (rowIndex + 1) * 2);
    const insideShell =
      coordinateX >= 0 && coordinateX <= 4 && coordinateY >= 0 && coordinateY <= 4;
    const insideHole = coordinateX > 1 && coordinateX < 3 && coordinateY > 1 && coordinateY < 3;
    expected.push(Number(insideShell && !insideHole));
  }
  const source = device.createBuffer({data: values, usage: Buffer.STORAGE | Buffer.COPY_DST});
  const positions = new GPUVector({
    type: 'data',
    name: 'sliced-points',
    format: 'float32x2',
    data: [
      new GPUData({
        buffer: source,
        format: 'float32x2',
        length: 129,
        byteOffset: 8,
        ownsBuffer: false
      }),
      new GPUData({
        buffer: source,
        format: 'float32x2',
        length: 384,
        byteOffset: 130 * 8,
        ownsBuffer: false
      })
    ]
  });
  const selection = new GPUKeplerSpatialSelection({device, positions, selection: POLYGON});
  try {
    submit(device, selection);
    expect(await readMask(selection)).toEqual(expected);
    expect(await selection.readStatus()).toMatchObject({complete: true, overflow: 0});
  } finally {
    selection.destroy();
    positions.destroy();
    source.destroy();
  }
});

it('GPU Kepler selection rebuilds polygon topology into a stable caller-owned mask', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makePositions(device, [[0, 0, 2, 2, 4, 4]]);
  const outputBuffer = device.createBuffer({
    byteLength: 12,
    usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const outputMask = new GPUVector({
    type: 'buffer',
    name: 'stable-selection',
    buffer: outputBuffer,
    format: 'uint32',
    length: 3,
    ownsBuffer: false
  });
  const rectangle = new GPUKeplerSpatialSelection({
    device,
    positions,
    outputMask,
    selection: {kind: 'rectangle', bounds: [1, 1, 5, 5]}
  });
  let triangle: GPUKeplerSpatialSelection | undefined;
  try {
    const ownedMaskSelection = new GPUKeplerSpatialSelection({
      device,
      positions,
      selection: {kind: 'rectangle', bounds: [1, 1, 5, 5]}
    });
    try {
      expect(ownedMaskSelection.workingMemoryByteLength - rectangle.workingMemoryByteLength).toBe(
        outputBuffer.byteLength
      );
    } finally {
      ownedMaskSelection.destroy();
    }
    const admittedBytes = rectangle.workingMemoryByteLength;
    expect(admittedBytes).toBeGreaterThan(0);
    submit(device, rectangle);
    // Compiled graph scratch is counted from the first encode, so the total can only grow once.
    const encodedBytes = rectangle.workingMemoryByteLength;
    expect(encodedBytes).toBeGreaterThanOrEqual(admittedBytes);
    submit(device, rectangle);
    expect(rectangle.workingMemoryByteLength).toBe(encodedBytes);
    expect(await readMask(rectangle)).toEqual([0, 1, 1]);
    rectangle.destroy();
    expect(rectangle.workingMemoryByteLength).toBe(0);
    expect(outputBuffer.destroyed).toBe(false);
    triangle = new GPUKeplerSpatialSelection({
      device,
      positions,
      outputMask,
      selection: {
        kind: 'polygon',
        polygons: [
          [
            [
              [-1, -1],
              [7, -1],
              [-1, 7]
            ]
          ]
        ]
      }
    });
    submit(device, triangle);
    expect(triangle.mask).toBe(outputMask);
    expect(await readMask(triangle)).toEqual([1, 1, 0]);
  } finally {
    rectangle.destroy();
    triangle?.destroy();
    outputBuffer.destroy();
    positions.destroy();
  }
});

it('GPU spatial selection snapshots queued geometry and retires diagnostics after reads', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makePositions(device, [
    [0, 0, 2, 2],
    [4, 4, 6, 6]
  ]);
  const selection = new GPUKeplerSpatialSelection({
    device,
    positions,
    selection: {kind: 'rectangle', bounds: [-1, -1, 3, 3]}
  });
  const snapshots = device.createBuffer({byteLength: 32, usage: Buffer.COPY_SRC | Buffer.COPY_DST});
  const maskBuffers = selection.mask.data.map(chunk => chunk.buffer as Buffer);
  try {
    const encoder = device.createCommandEncoder();
    for (const [snapshotIndex, bounds] of [
      [-1, -1, 3, 3],
      [3, 3, 7, 7]
    ].entries()) {
      selection.updateSelection({
        kind: 'rectangle',
        bounds: [bounds[0], bounds[1], bounds[2], bounds[3]]
      });
      selection.encode(encoder);
      for (const [chunkIndex, buffer] of maskBuffers.entries()) {
        encoder.copyBufferToBuffer({
          sourceBuffer: buffer,
          destinationBuffer: snapshots,
          destinationOffset: snapshotIndex * 16 + chunkIndex * 8,
          size: 8
        });
      }
    }
    device.submit(encoder.finish());
    const bytes = await snapshots.readAsync();
    expect([...new Uint32Array(bytes.buffer, bytes.byteOffset, 8)]).toEqual([
      1, 1, 0, 0, 0, 0, 1, 1
    ]);
    const previousStatus = selection.readStatus();
    // Same revision, new encoding: the earlier asynchronous diagnostics must still be stale.
    submit(device, selection);
    expect(await previousStatus).toMatchObject({stale: true, complete: false});
    const finalStatus = selection.readStatus();
    const admittedBytes = selection.workingMemoryByteLength;
    selection.destroy();
    expect(selection.workingMemoryByteLength).toBe(admittedBytes);
    for (const buffer of maskBuffers) expect(buffer.destroyed).toBe(false);
    expect(await finalStatus).toMatchObject({stale: true, complete: false});
    expect(selection.workingMemoryByteLength).toBe(0);
    for (const buffer of maskBuffers) expect(buffer.destroyed).toBe(true);
  } finally {
    selection.destroy();
    snapshots.destroy();
    positions.destroy();
  }
});
