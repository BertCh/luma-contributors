// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkInhomogeneousK} from '../../../src/gpu-network/gpu-network-inhomogeneous-k';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './network-point-process-fixtures.json';

for (const fixture of fixtures.cases) {
  it(`inhomogeneous network K matches independently enumerated metric spheres: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
      format: Format,
      values: number[]
    ) => {
      const buffer = createInputBuffer(
        device,
        format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
      );
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        `input-${buffers.length}`,
        buffer,
        format,
        format === 'uint32x2' ? values.length / 2 : values.length
      );
    };
    const output = <Format extends 'float32' | 'uint32'>(format: Format, count: number) => {
      const buffer = createOutputBuffer(device, count);
      buffers.push(buffer);
      return {
        view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, count),
        buffer
      };
    };
    const eventNodeIds = input('uint32', fixture.events);
    const intensity = input('float32', fixture.intensity);
    const nodeDistances = input(
      'float32',
      fixture.nodeDistances.map(value => value ?? Infinity)
    );
    const edgeNodes = input(
      'uint32x2',
      fixture.edges.flatMap(edge => edge.slice(0, 2))
    );
    const edgeLengths = input(
      'float32',
      fixture.edges.map(edge => edge[2])
    );
    const radii = input('float32', fixture.radii);
    const results: {
      buffer: Buffer;
      numerator: Buffer;
      status: Buffer;
      expected: number[];
      expectedNumerator: number[];
    }[] = [];
    for (const correction of ['none', 'ang'] as const) {
      for (const normalization of [
        'network-length',
        'inverse-intensity',
        'inverse-intensity-squared'
      ] as const) {
        const normalizationIndex = [
          'network-length',
          'inverse-intensity',
          'inverse-intensity-squared'
        ].indexOf(normalization);
        const values = output('float32', fixture.radii.length);
        const numerator = output('float32', fixture.radii.length);
        const status = output('uint32', fixture.radii.length);
        graph.add(
          new GPUNetworkInhomogeneousK({
            id: `${correction}-${normalization}`,
            eventNodeIds,
            intensity,
            nodeDistances,
            nodeCount: fixture.nodeCount,
            edgeNodes,
            edgeLengths,
            radii,
            correction,
            normalization,
            output: {
              k: values.view,
              numerator: numerator.view,
              normalization: output('float32', 3).view,
              status: status.view
            }
          })
        );
        const expectedNumerator =
          correction === 'ang' ? fixture.numeratorAng : fixture.numeratorNone;
        results.push({
          buffer: values.buffer,
          numerator: numerator.buffer,
          status: status.buffer,
          expected: expectedNumerator.map(
            value => value / fixture.normalizations[normalizationIndex]
          ),
          expectedNumerator
        });
      }
    }
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    for (const result of results) {
      const values = await readFloat32(result.buffer, fixture.radii.length);
      const numerator = await readFloat32(result.numerator, fixture.radii.length);
      expect(await readUint32(result.status, fixture.radii.length)).toEqual(
        fixture.radii.map(() => 0)
      );
      for (let radius = 0; radius < values.length; radius++) {
        expect(values[radius]).toBeCloseTo(result.expected[radius], 5);
        expect(numerator[radius]).toBeCloseTo(result.expectedNumerator[radius], 5);
      }
    }
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  });
}

it('inhomogeneous network K invalidates bad intensities and recovers after a resident update', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
    format: Format,
    values: number[]
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return {
      view: importGraphBuffer(
        graph,
        `input-${buffers.length}`,
        buffer,
        format,
        format === 'uint32x2' ? values.length / 2 : values.length
      ),
      buffer
    };
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, count: number) => {
    const buffer = createOutputBuffer(device, count);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, count),
      buffer
    };
  };
  const intensity = input('float32', [0, 1]);
  const values = output('float32', 1);
  const status = output('uint32', 1);
  graph.add(
    new GPUNetworkInhomogeneousK({
      eventNodeIds: input('uint32', [0, 1]).view,
      intensity: intensity.view,
      nodeDistances: input('float32', [0, 2, 2, 0]).view,
      nodeCount: 2,
      edgeNodes: input('uint32x2', [0, 1]).view,
      edgeLengths: input('float32', [2]).view,
      radii: input('float32', [2]).view,
      output: {
        k: values.view,
        numerator: output('float32', 1).view,
        normalization: output('float32', 3).view,
        status: status.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect((await readUint32(status.buffer, 1))[0] & 1).toBe(1);
  expect(Number.isNaN((await readFloat32(values.buffer, 1))[0])).toBe(true);
  intensity.buffer.write(Float32Array.from([1, 1]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(status.buffer, 1)).toEqual([0]);
  expect(await readFloat32(values.buffer, 1)).toEqual([1]);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
