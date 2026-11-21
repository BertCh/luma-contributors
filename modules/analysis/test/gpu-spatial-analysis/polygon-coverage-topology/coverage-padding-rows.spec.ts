// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Rows before the first ring offset and after the last are padding, including for snap origins.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {requireTestDevice} from '../../utils/require-test-device';

type TestDevice = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

/** Two unit squares side by side: one feature each, both labelled 7. */
const SQUARES = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1];

async function runDissolve(
  device: TestDevice,
  trailing: number,
  value: number,
  leading = 0,
  coordinates = SQUARES,
  snapTolerance = 0.001,
  vertexTolerance?: number
) {
  const vertexCount = coordinates.length / 2 + trailing + leading;
  const positions = new Float32Array(2 * vertexCount).fill(value);
  positions.set(coordinates, 2 * leading);
  const ringCapacity = 2;
  const vertexCapacity = 16;
  const rig = new WeightsRig(device);
  const out = {
    ringOffsets: rig.output('uint32', ringCapacity + 1),
    positions: rig.output('float32x2', vertexCapacity),
    count: rig.output('uint32', 1),
    overflow: rig.output('uint32', 1),
    uncertainCount: rig.output('uint32', 1),
    boundarySegmentCount: rig.output('uint32', 1)
  };
  rig.run(
    new GPUCoverageDissolve({
      positions: rig.input(positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(Uint32Array.of(leading, leading + 4, leading + 8), 'uint32', 3),
      polygonOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
      featureOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
      labels: rig.input(Uint32Array.of(7, 7), 'uint32', 2),
      snapGridSize: snapTolerance,
      vertexTolerance,
      output: {
        ringOffsets: out.ringOffsets.view,
        positions: out.positions.view,
        status: {
          count: out.count.view,
          overflow: out.overflow.view,
          uncertainCount: out.uncertainCount.view,
          boundarySegmentCount: out.boundarySegmentCount.view
        }
      }
    })
  );
  const offsets = await readUint32(out.ringOffsets.buffer, ringCapacity + 1);
  const result = {
    count: (await readUint32(out.count.buffer, 1))[0],
    overflow: (await readUint32(out.overflow.buffer, 1))[0],
    uncertainCount: (await readUint32(out.uncertainCount.buffer, 1))[0],
    boundarySegmentCount: (await readUint32(out.boundarySegmentCount.buffer, 1))[0],
    offsets,
    positions: await readFloat32(out.positions.buffer, 2 * offsets[ringCapacity])
  };
  rig.destroy();
  return result;
}

it('GPUCoverageDissolve does not count padding rows as uncertain snap keys', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const reference = await runDissolve(device, 0, 0);
  expect(reference.uncertainCount).toBe(0);
  expect(reference.overflow).toBe(0);
  expect(reference.count).toBe(1);
  for (const value of [Number.NaN, 3e6, -3e6]) {
    expect(await runDissolve(device, 3, value)).toEqual(reference);
  }
});

it('GPUCoverageDissolve anchors snapping at the first ring-owned vertex after leading padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A zero-anchored grid merges the two near boundaries; a grid anchored at the real first
  // vertex keeps them separate (cells 4 and 5). All coordinates are dyadic.
  const coordinates = [
    10.0625, 20.0625, 11.125, 20.0625, 11.125, 21.0625, 10.0625, 21.0625, 11.1875, 20.0625, 12.0625,
    20.0625, 12.0625, 21.0625, 11.1875, 21.0625
  ];
  // Assembly compares the stored canonical coordinates, not snap-cell indices. Its default
  // tolerance is 0.25 / 4 = 1/16, exactly the gap: use a smaller matching distance so this
  // fixture isolates the snap origin rather than connecting its two distinct boundaries.
  const vertexTolerance = 1 / 1024;
  const reference = await runDissolve(device, 0, 0, 0, coordinates, 0.25, vertexTolerance);
  expect(reference.count).toBe(2);
  expect(reference.boundarySegmentCount).toBe(8);
  expect(reference.uncertainCount).toBe(0);
  for (const value of [Number.NaN, 3e20, -3e20]) {
    expect(await runDissolve(device, 3, value, 2, coordinates, 0.25, vertexTolerance)).toEqual(
      reference
    );
  }
  const projected = SQUARES.map(coordinate => coordinate + 1e6);
  const projectedReference = await runDissolve(device, 0, 0, 0, projected);
  expect(projectedReference.count).toBe(1);
  expect(projectedReference.uncertainCount).toBe(0);
  for (const value of [Number.NaN, 3e20, -3e20]) {
    expect(await runDissolve(device, 3, value, 2, projected)).toEqual(projectedReference);
  }
});
