// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createGroupedPruningCase,
  createGroupedRingCase,
  createNodingParameterCase,
  createRingBucketCase,
  createRingExtentCase,
  NODING_PARAMETER_CASES,
  type CreateGeometryFixture
} from './b7-geometry-fixtures';
import {getOracleView, type OracleAllocate} from './noding-polygonize-output-oracles-fixtures';

async function runFixture(createFixture: CreateGeometryFixture) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  try {
    const allocate: OracleAllocate = (name, format, length, values) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const buffer = values
        ? createInputBuffer(
            device,
            format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values)
          )
        : createOutputBuffer(device, length * width);
      const view = importGraphBuffer(graph, name, buffer, format, length);
      buffers.set(view, buffer);
      return view;
    };
    const fixture = createFixture(graph, allocate);
    for (const node of fixture.nodes) graph.add(node);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const parameters: Record<string, number[]> = {};
      const directParameters: Record<string, number[]> = {};
      for (const [path, values] of Object.entries(fixture.expected)) {
        const view = getOracleView(fixture.output, path);
        expect((view.length * view.rowByteLength) / 4, path).toBe(values.length);
        const buffer = buffers.get(view)!;
        const actual = view.format.startsWith('uint')
          ? await readUint32(buffer, values.length)
          : await readFloat32(buffer, values.length);
        if (fixture.reciprocalExpected?.[path]) {
          parameters[path] = actual;
          directParameters[path] = values;
        } else {
          expect(actual, path).toEqual(values);
        }
      }
      if (fixture.reciprocalExpected) {
        // Accept only one complete pre-stream bit pattern. Some exact fractions
        // are preserved by only one of the permitted division lowerings.
        expect([directParameters, fixture.reciprocalExpected]).toContainEqual(parameters);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers.values()) buffer.destroy();
    device.destroy();
  }
}

it.each(NODING_PARAMETER_CASES)('line noding preserves %s split parameters', async geometry => {
  await runFixture((graph, allocate) => createNodingParameterCase(graph, allocate, geometry));
});

it('polygonize candidate visits stay local for 32 squares with a distant group', async () => {
  await runFixture((graph, allocate) => createGroupedPruningCase(graph, allocate, 32, true));
});

it.each([
  'explicit',
  'derived'
] as const)('ring assembly matches across a cell boundary with %s tolerance', async tolerance => {
  await runFixture((graph, allocate) => createGroupedRingCase(graph, allocate, tolerance));
});

it.each([
  false,
  true
])('ring assembly local hash buckets stay sparse with distant group=%s', async distant => {
  await runFixture((graph, allocate) => createRingBucketCase(graph, allocate, distant));
});

it('ring assembly reports overflow on the rectangle spanning opposite float32 extremes', async () => {
  await runFixture(createRingExtentCase);
});
