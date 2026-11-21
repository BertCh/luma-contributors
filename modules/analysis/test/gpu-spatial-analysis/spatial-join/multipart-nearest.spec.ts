// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {it, expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  MULTIPART_NEAREST_CASES,
  createMultipartNearestFixture,
  assertMultipartNearestFixture
} from './multipart-nearest-fixture';
import {
  NEAREST_REVIEW_CASES,
  createReviewNearestFixture,
  assertReviewNearestFixture
} from './multipart-nearest-review-fixture';

for (const definition of NEAREST_REVIEW_CASES) {
  it(`${definition.name} WebGPU regression`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers = new Map<object, Buffer>();
    try {
      const fixture = createReviewNearestFixture(
        graph,
        (name, format, length, values) => {
          const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
          buffer.write(values);
          const view = importGraphBuffer(graph, name, buffer, format, length);
          buffers.set(view, buffer);
          return view;
        },
        definition
      );
      graph.add(fixture.join);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        await assertReviewNearestFixture(fixture, view =>
          view.format === 'uint32'
            ? readUint32(buffers.get(view)!, view.length)
            : readFloat32(buffers.get(view)!, view.length * (view.format === 'float32x2' ? 2 : 1))
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      for (const buffer of buffers.values()) buffer.destroy();
    }
  });
}

for (const [caseIndex, definition] of MULTIPART_NEAREST_CASES.entries()) {
  for (const options of [
    {mode: 'nearest' as const},
    {mode: 'nearest' as const, querySort: true},
    {neighborCount: 4},
    {neighborCount: 32},
    {neighborCount: 32, radius: 2.25, querySort: true},
    {neighborCount: 1, neighborCapacity: 2, ties: 'all' as const},
    {neighborCount: 1, ties: 'all' as const}
  ]) {
    it(`Shapely multipart ${definition.queryKind}/${definition.featureKind} ${JSON.stringify(options)}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const byView = new Map<object, Buffer>();
      try {
        const fixture = createMultipartNearestFixture(
          graph,
          (name, format, length, values) => {
            const words = length * (format === 'float32x2' ? 2 : 1);
            const buffer = createOutputBuffer(device, words);
            if (values) buffer.write(values);
            buffers.push(buffer);
            const view = importGraphBuffer(graph, name, buffer, format, length);
            byView.set(view, buffer);
            return view;
          },
          caseIndex,
          options
        );
        graph.add(fixture.join);
        if (fixture.weights) graph.add(fixture.weights);
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          await assertMultipartNearestFixture(fixture, view =>
            view.format === 'uint32'
              ? readUint32(byView.get(view)!, view.length)
              : readFloat32(byView.get(view)!, view.length * (view.format === 'float32x2' ? 2 : 1))
          );
        } finally {
          compiled.destroy();
        }
      } finally {
        buffers.forEach(buffer => buffer.destroy());
      }
    });
  }
}

it('multipart BVH loss propagates to weights and every affected query', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const byView = new Map<object, Buffer>();
  try {
    const fixture = createMultipartNearestFixture(
      graph,
      (name, format, length, values) => {
        const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
        if (values) buffer.write(values);
        buffers.push(buffer);
        const view = importGraphBuffer(graph, name, buffer, format, length);
        byView.set(view, buffer);
        return view;
      },
      0,
      {leafCapacity: 16}
    );
    graph.add(fixture.join);
    graph.add(fixture.weights!);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const name of ['overflow', 'candidateOverflow', 'weightCandidateOverflow']) {
        expect(await readUint32(byView.get(fixture.results[name])!, 1)).toEqual([1]);
      }
      expect(await readUint32(byView.get(fixture.results['weightOverflow'])!, 1)).toEqual([0]);
      expect(await readUint32(byView.get(fixture.results['candidateTruncated'])!, 30)).toEqual(
        Array(30).fill(1)
      );
    } finally {
      compiled.destroy();
    }
  } finally {
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('R5 reused WebGPU graph replaces occupied point slots with NaN padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<object, Buffer>();
  let radius: Buffer | undefined;
  try {
    const fixture = createMultipartNearestFixture(
      graph,
      (name, format, length, values) => {
        const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
        if (values) buffer.write(values);
        if (name === 'radius') radius = buffer;
        const view = importGraphBuffer(graph, name, buffer, format, length);
        buffers.set(view, buffer);
        return view;
      },
      0,
      {neighborCount: 4, radius: 10000}
    );
    graph.add(fixture.join);
    graph.add(fixture.weights!);
    const compiled = graph.compile();
    const read = (view: import('@luma.gl/graph/gpu-core').GraphDataView) =>
      view.format === 'uint32'
        ? readUint32(buffers.get(view)!, view.length)
        : readFloat32(buffers.get(view)!, view.length * (view.format === 'float32x2' ? 2 : 1));
    try {
      submitGraph(device, compiled, undefined);
      await assertMultipartNearestFixture(fixture, read);
      for (const port of ['neighborFootPoints', 'neighborQueryPoints'])
        expect((await read(fixture.results[port])).slice(0, 8).every(Number.isFinite)).toBe(true);
      radius!.write(Float32Array.of(0));
      fixture.radiusValue = 0;
      submitGraph(device, compiled, undefined);
      await assertMultipartNearestFixture(fixture, read);
      for (const port of ['neighborFootPoints', 'neighborQueryPoints'])
        expect((await read(fixture.results[port])).slice(0, 8).every(Number.isNaN)).toBe(true);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers.values()) buffer.destroy();
  }
});
