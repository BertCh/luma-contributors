// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkKernelDensity} from '../../src/gpu-network/gpu-network-kernel-density';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createOracleContext} from './adjacency-matrix/network-output-oracle-test-utils';

it('GPUNetworkKernelDensity publishes the invalid query count beside per-query validity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'kernel-density-invalid-count'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32'>(name: string, values: number[], format: Format) => {
    const buffer = createInputBuffer(device, Float32Array.from(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const intensity = output('intensity', 'float32', 4);
  const mass = output('mass', 'float32', 2);
  const validity = output('validity', 'uint32', 4);
  const invalidCount = output('invalid-count', 'uint32', 1);
  // Query 1 has a negative (invalid) distance to event 0, so the mass of event 0 is invalid and
  // every query that reads it is invalid too. All four validity words must be zero.
  graph.add(
    new GPUNetworkKernelDensity({
      distances: input(
        'distances',
        [0, Infinity, -1, Infinity, 1, Infinity, Infinity, 0],
        'float32'
      ),
      eventCount: 2,
      quadratureLengths: input('lengths', [1, 2, 1, 3], 'float32'),
      bandwidth: 2,
      output: {
        intensity: intensity.view,
        kernelMass: mass.view,
        validity: validity.view,
        status: {invalidCount: invalidCount.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(validity.buffer, 4)).toEqual([0, 0, 0, 0]);
  expect(await readUint32(invalidCount.buffer, 1)).toEqual([4]);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('network KDE counts one invalid row in a mixed validity mask', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  try {
    const validity = output('validity', 'uint32', 3);
    const invalidCount = output('invalid-count', 'uint32');
    // Every event has triangular shape mass 2^-126, so each coincident contribution is 2^126.
    // Four contributions overflow float32 in the first row. Excluding one leaves a finite
    // 3 * 2^126 in the second row. The third row is at the support boundary and contributes zero.
    graph.add(
      new GPUNetworkKernelDensity({
        distances: input('distances', 'float32', [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3]),
        eventCount: 4,
        quadratureLengths: input('lengths', 'float32', [2 ** -126, 0, 1]),
        bandwidth: 3,
        excludedEventIds: input('excluded-events', 'uint32', [0xffffffff, 0, 0xffffffff]),
        output: {
          intensity: output('intensity', 'float32', 3),
          kernelMass: output('mass', 'float32', 4),
          validity,
          status: {invalidCount}
        }
      })
    );
    context.run();
    expect(await read(validity)).toEqual([0, 1, 1]);
    expect(await read(invalidCount)).toEqual([1]);
  } finally {
    context.destroy();
  }
});

for (const [bandwidth, quadratureLength] of [
  [3e38, 1e20],
  [3e-38, 1e-20]
]) {
  it(`network KDE retains normalized density at bandwidth ${bandwidth} with packed outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const distances = createInputBuffer(device, Float32Array.of(bandwidth / 2, bandwidth * 0.75));
    const lengths = createInputBuffer(device, Float32Array.of(quadratureLength, quadratureLength));
    const packed = createOutputBuffer(device, 66);
    // One physical import. Mass is adjacent to intensity in the same alignment window; their
    // writers are separate dispatches. Intensity and validity are co-bound writable disjoint ranges.
    const storage = importGraphBuffer(graph, 'packed', packed, 'uint32', 66).buffer;
    const intensity = graph.createDataView(storage, {format: 'float32', length: 2});
    const mass = graph.createDataView(storage, {format: 'float32', length: 1, byteOffset: 8});
    const validity = graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 256});
    graph.add(
      new GPUNetworkKernelDensity({
        distances: importGraphBuffer(graph, 'distances', distances, 'float32', 2),
        eventCount: 1,
        quadratureLengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', 2),
        bandwidth,
        output: {intensity, kernelMass: mass, validity}
      })
    );
    const compiled = graph.compile();
    try {
      await submitGraph(device, compiled, undefined);
      const values = await readFloat32(packed, 66);
      // Triangular shapes 1/2 and 1/4 have total mass (3/4) * length / bandwidth.
      const expected = [
        2 / (3 * quadratureLength),
        1 / (3 * quadratureLength),
        (0.75 * quadratureLength) / bandwidth
      ];
      for (let index = 0; index < expected.length; index++) {
        expect(Number.isFinite(values[index])).toBe(true);
        expect(Math.abs(values[index] / expected[index] - 1)).toBeLessThan(2e-5);
      }
      expect(Array.from(await readUint32(packed, 66)).slice(64)).toEqual([1, 1]);
    } finally {
      compiled.destroy();
      distances.destroy();
      lengths.destroy();
      packed.destroy();
    }
  });
}

it('CPL-11 suppresses incomplete matrix KDE and recovers on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const distances = createInputBuffer(device, Float32Array.of(0, 1));
  const lengths = createInputBuffer(device, Float32Array.of(1, 1));
  const evidence = createInputBuffer(device, Uint32Array.of(0));
  const horizon = createInputBuffer(device, Float32Array.of(2));
  const intensity = createOutputBuffer(device, 2);
  const mass = createOutputBuffer(device, 1);
  const validity = createOutputBuffer(device, 2);
  const invalidCount = createOutputBuffer(device, 1);
  graph.add(
    new GPUNetworkKernelDensity({
      distances: importGraphBuffer(graph, 'distances', distances, 'float32', 2),
      quadratureLengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', 2),
      costsComplete: importGraphBuffer(graph, 'evidence', evidence, 'uint32', 1),
      costLimit: importGraphBuffer(graph, 'horizon', horizon, 'float32', 1),
      bandwidth: 2,
      eventCount: 1,
      output: {
        intensity: importGraphBuffer(graph, 'intensity', intensity, 'float32', 2),
        kernelMass: importGraphBuffer(graph, 'mass', mass, 'float32', 1),
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 2),
        status: {invalidCount: importGraphBuffer(graph, 'invalidCount', invalidCount, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const [complete, limit, valid] of [
      [0, 2, false],
      [1, 1, false],
      [1, 2, true],
      [1, Number.NaN, false],
      [1, -1, false],
      [1, Infinity, true],
      [0, 2, false],
      [1, 2, true]
    ] as const) {
      evidence.write(Uint32Array.of(complete));
      horizon.write(Float32Array.of(limit));
      await submitGraph(device, compiled, undefined);
      expect(await readUint32(validity, 2)).toEqual(valid ? [1, 1] : [0, 0]);
      expect(await readUint32(invalidCount, 1)).toEqual(valid ? [0] : [2]);
      const values = await readFloat32(intensity, 2);
      expect(values).toHaveLength(2);
      if (valid) {
        // Shapes [1, 1/2] normalize to [2/3, 1/3]; mass = (3/2)/2.
        expect(values.every(Number.isFinite)).toBe(true);
        expect(values[0]).toBeCloseTo(2 / 3);
        expect(values[1]).toBeCloseTo(1 / 3);
        expect(await readFloat32(mass, 1)).toEqual([0.75]);
      } else {
        expect(values).toEqual([NaN, NaN]);
        expect(await readFloat32(mass, 1)).toEqual([NaN]);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of [
      distances,
      lengths,
      evidence,
      horizon,
      intensity,
      mass,
      validity,
      invalidCount
    ])
      buffer.destroy();
  }
});

it('S1 certifies subnormal horizons by their unsigned words on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const distances = createInputBuffer(device, Float32Array.of(0));
  const lengths = createInputBuffer(device, Float32Array.of(1));
  const evidence = createInputBuffer(device, Uint32Array.of(7));
  const horizon = createInputBuffer(device, Uint32Array.of(8));
  const intensity = createOutputBuffer(device, 1);
  const mass = createOutputBuffer(device, 1);
  const validity = createOutputBuffer(device, 1);
  const complete = createOutputBuffer(device, 1);
  graph.add(
    new GPUNetworkKernelDensity({
      distances: importGraphBuffer(graph, 'distances', distances, 'float32', 1),
      quadratureLengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', 1),
      costsComplete: importGraphBuffer(graph, 'evidence', evidence, 'uint32', 1),
      costLimit: importGraphBuffer(graph, 'horizon', horizon, 'float32', 1),
      bandwidth: new Float32Array(Uint32Array.of(16).buffer)[0],
      eventCount: 1,
      output: {
        intensity: importGraphBuffer(graph, 'intensity', intensity, 'float32', 1),
        kernelMass: importGraphBuffer(graph, 'mass', mass, 'float32', 1),
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 1),
        status: {complete: importGraphBuffer(graph, 'complete', complete, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    // IEEE ordering: 8 < bandwidth word 16; equality and +Infinity cover the kernel support.
    for (const [bits, expected] of [
      [8, 0],
      [16, 1],
      [0x7f800000, 1],
      [0x7fc00000, 0],
      [0xbf800000, 0],
      [8, 0]
    ] as const) {
      horizon.write(Uint32Array.of(bits));
      await submitGraph(device, compiled, undefined);
      expect(await readUint32(complete, 1)).toEqual([expected]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [
      distances,
      lengths,
      evidence,
      horizon,
      intensity,
      mass,
      validity,
      complete
    ])
      buffer.destroy();
  }
});
