// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {NEAREST_RANK_BOUNDARY_CASES} from './nearest-rank-boundary-test-utils';

it.each(NEAREST_RANK_BOUNDARY_CASES)('GPU nearest $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
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
  const ids = output('ids', 'uint32', 2);
  const distances = output('distances', 'float32', 2);
  const counts = output('counts', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  graph.add(
    new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions: input('query', 'float32x2', Float32Array.of(0, 0))},
      features: {
        kind: 'points',
        positions: input('features', 'float32x2', Float32Array.from(fixture.positions)),
        sourceIds: input('source-ids', 'uint32', Uint32Array.of(900, 100))
      },
      neighborCount: 2,
      output: {
        neighborIds: ids.view,
        neighborDistances: distances.view,
        neighborCounts: counts.view,
        status: {overflow: overflow.view}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(ids.buffer, 2)).toEqual(fixture.order);
    expect(await readUint32(counts.buffer, 1)).toEqual([2]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    // Both represented distances round to 1; the rank ordering above still distinguishes them.
    expect(await readFloat32(distances.buffer, 2)).toEqual([1, 1]);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {
  createNearestFinalWaveCases,
  expectNearestResultValues
} from './nearest-rank-boundary-test-utils';

for (const entry of createNearestFinalWaveCases())
  it(`WebGPU ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: entry.inputs,
      outputs: entry.outputs,
      create: ({inputs, outputs}) => {
        if (outputs['foot']) outputs['foot'].buffer.defaultBuffer!.write(Float32Array.of(123, 456));
        return {getCommandNodes: graph => entry.declare(graph, inputs, outputs)};
      }
    });
    try {
      const actual = await fixture.run();
      for (const [name, expected] of Object.entries(entry.expected)) {
        expectNearestResultValues(
          actual[name],
          expected,
          entry.outputs[name].format === 'uint32' || name === 'foot' || name === 'queryPoint',
          entry.tolerance ?? 0
        );
      }
      if (actual['foot'])
        expect(
          Math.hypot(
            actual['queryPoint'][0] - actual['foot'][0],
            actual['queryPoint'][1] - actual['foot'][1]
          )
        ).toBeCloseTo(entry.expected['distances'][0], -1);
    } finally {
      fixture.destroy();
    }
  });

import {
  NEAREST_REVIEW_CASES,
  createReviewNearestFixture,
  assertReviewNearestFixture
} from './multipart-nearest-review-fixture';
for (const definition of NEAREST_REVIEW_CASES.filter(entry =>
  entry.name.startsWith('R2 reversed parallel edges')
))
  it(`WebGPU KI-rr-joins-R4 paired-witness control ${definition.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const fixture = createReviewNearestFixture(
      graph,
      (name, format, length, values) => {
        // Outputs are read back, so every fixture buffer needs COPY_SRC (createOutputBuffer).
        const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
        buffer.write(values);
        buffers.push(buffer);
        return importGraphBuffer(graph, name, buffer, format, length);
      },
      definition
    );
    graph.add(fixture.join);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      await assertReviewNearestFixture(fixture, async view => {
        const buffer = view.buffer.defaultBuffer as Buffer;
        const length = view.length * (view.format === 'float32x2' ? 2 : 1);
        return view.format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length);
      });
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
