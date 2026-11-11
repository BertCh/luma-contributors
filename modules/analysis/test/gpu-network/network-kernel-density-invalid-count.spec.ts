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
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

it('GPUNetworkKernelDensity publishes the invalid query count beside the per-query status', async () => {
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
  const status = output('status', 'uint32', 4);
  const invalidCount = output('invalid-count', 'uint32', 1);
  // Query 1 has a negative (invalid) distance to event 0, so the mass of event 0 is invalid and
  // every query that reads it is invalid too: the count is the number of status-1 rows.
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
        status: status.view,
        ports: {status: {invalidCount: invalidCount.view}}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const statuses = await readUint32(status.buffer, 4);
  expect(statuses.some(word => word === 1)).toBe(true);
  expect(await readUint32(invalidCount.buffer, 1)).toEqual([
    statuses.filter(word => word === 1).length
  ]);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
});
