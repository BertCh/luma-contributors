// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';

import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterRegistration, GPURasterRegionGraph} from '../../../src/gpu-raster/remote-sensing';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import reference from '../../oracles/fixtures/raster-registration-regions.json';

for (const fixture of reference.registration) {
  it(`RS07 masked linear translation ${fixture.translation} matches independent SciPy interpolation/NCC`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegistration({
        mode: 'masked-linear',
        grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
        reference: harness.input(fixture.reference),
        moving: harness.input(fixture.moving),
        referenceValidity: harness.mask(fixture.referenceMask),
        movingValidity: harness.mask(fixture.movingMask),
        maximumShift: [4, 3],
        upsampleFactor: 20,
        minimumOverlapFraction: 0.3,
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          overlap: harness.output('overlap', 'uint32', 1)
        }
      })
    );
    expect(result.status).toEqual([0]);
    fixture.expectedShift.forEach((value, index) => {
      expect(result.shift[index]).toBeCloseTo(value, 4);
    });
    expect(result.peak[0]).toBeCloseTo(fixture.expectedPeak, 4);
    expect(result.overlap).toEqual([fixture.expectedOverlap]);
    // The independently maximized bilinear objective is also near the known physical shift.
    fixture.translation.forEach((value, index) => {
      expect(Math.abs(result.shift[index] + value)).toBeLessThanOrEqual(0.10001);
    });
  });
}

it('RS07 masked mode rejects insufficient overlap and constant local support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const constant of [false, true]) {
    const harness = createRemoteSensingHarness(device);
    const values = Array.from({length: 48}, (_, index) => (constant ? 2 : Math.sin(index)));
    const result = await harness.run(
      new GPURasterRegistration({
        mode: 'masked-linear',
        grid: makeGPURasterMetadata({width: 8, height: 6}),
        reference: harness.input(values),
        moving: harness.input(values),
        referenceValidity: harness.mask(values.map((_, index) => (constant || index < 2 ? 1 : 0))),
        movingValidity: harness.mask(values.map(() => 1)),
        maximumShift: [2, 2],
        minimumOverlapCount: 3,
        upsampleFactor: 1,
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1),
          overlap: harness.output('overlap', 'uint32', 1)
        }
      })
    );
    expect(result.status).toEqual([3]);
    expect(result.overlap).toEqual([0]);
    expect(result.shift.every(Number.isNaN)).toBe(true);
  }
});

it('RS07 masked overlap is linear, applies QA before interpolation, and preserves integer offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const fixture = reference.registration[0];
  const result = await harness.run(
    new GPURasterRegistration({
      mode: 'masked-linear',
      grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
      reference: harness.input(
        fixture.reference.map((value, index) => (fixture.referenceMask[index] ? value : NaN))
      ),
      moving: harness.input(
        fixture.moving.map((value, index) => (fixture.movingMask[index] ? value : NaN))
      ),
      maximumShift: [4, 3],
      upsampleFactor: 1,
      output: {
        shift: harness.output('shift', 'float32x2', 1),
        peak: harness.output('peak', 'float32', 1),
        status: harness.output('status', 'uint32', 1),
        overlap: harness.output('overlap', 'uint32', 1)
      }
    })
  );
  expect(result.status).toEqual([0]);
  expect(result.shift).toEqual([-2, 1]);
  expect(result.peak[0]).toBeCloseTo(1, 5);
  expect(result.overlap).toEqual([fixture.expectedOverlap]);
});

for (const [index, fixture] of reference.regions.entries()) {
  it(`RS08 adaptive merge ${index} matches independent NetworkX quotient-graph mean updates`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const regions = fixture.expectedMapping.length;
    const result = await harness.run(
      new GPURasterRegionGraph({
        grid: makeGPURasterMetadata({width: fixture.values.length, height: 1}),
        labels: harness.mask(fixture.labels),
        values: harness.input(fixture.values),
        regionCount: regions,
        mergeThreshold: fixture.threshold,
        mergeMode: 'adaptive',
        maximumIterations: fixture.rounds,
        output: {
          adjacency: harness.output('adjacency', 'uint32', regions ** 2),
          weights: harness.output('weights', 'float32', regions ** 2),
          mapping: harness.output('mapping', 'uint32', regions),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result.mapping).toEqual(fixture.expectedMapping);
    expect(result.status).toEqual([fixture.expectedStatus]);
    if (index === 1) {
      expect(result.weights[5]).toBeCloseTo(3.5, 6);
      expect(result.weights[1]).toBeNaN();
    }
  });
}

it('RS07 compiled masked graph resets FFT support after QA and radiometric updates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const importView = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: `masked-repeat-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length});
  };
  const values = Float32Array.from(
    {length: 64},
    (_, index) => Math.sin(index * 12.7) + Math.cos(index * 0.37)
  );
  const referenceBuffer = createInputBuffer(device, values);
  const movingBuffer = createInputBuffer(device, values);
  const maskBuffer = createInputBuffer(device, new Uint32Array(64).fill(1));
  const shiftBuffer = createOutputBuffer(device, 2);
  const peakBuffer = createOutputBuffer(device, 1);
  const statusBuffer = createOutputBuffer(device, 1);
  const overlapBuffer = createOutputBuffer(device, 1);
  graph.add(
    new GPURasterRegistration({
      mode: 'masked-linear',
      grid: makeGPURasterMetadata({width: 8, height: 8}),
      reference: importView(referenceBuffer, 'float32', 64),
      moving: importView(movingBuffer, 'float32', 64),
      movingValidity: importView(maskBuffer, 'uint32', 64),
      maximumShift: [2, 2],
      upsampleFactor: 1,
      output: {
        shift: importView(shiftBuffer, 'float32x2', 1),
        peak: importView(peakBuffer, 'float32', 1),
        status: importView(statusBuffer, 'uint32', 1),
        overlap: importView(overlapBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(shiftBuffer, 2)).toEqual([0, 0]);
    expect(await readUint32(overlapBuffer, 1)).toEqual([64]);
    maskBuffer.write(new Uint32Array(64));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(statusBuffer, 1)).toEqual([3]);
    expect(await readUint32(overlapBuffer, 1)).toEqual([0]);
    maskBuffer.write(new Uint32Array(64).fill(1));
    movingBuffer.write(Float32Array.from(values, value => value * 3 + 20));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(statusBuffer, 1)).toEqual([0]);
    expect(await readFloat32(shiftBuffer, 2)).toEqual([0, 0]);
    expect((await readFloat32(peakBuffer, 1))[0]).toBeCloseTo(1, 5);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
