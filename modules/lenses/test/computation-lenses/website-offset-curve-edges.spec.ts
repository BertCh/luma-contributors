// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createOutputBuffer,
  importGraphBuffer,
  readUint32,
  requireTestDevice,
  submitGraph
} from '@luma-contributors/analysis/testing';
import {expect, test} from 'vitest';
import {addOffsetCurveComponentEdges} from '../../../../website/src/gpu/runs/entries/geometry-processing/offset-curve-components';
import type {RunBuffer, RunBuilder} from '../../../../website/src/gpu/runs/define-run';

test('website offset edges ignore stale components on loss and past the live prefix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph<void>(device);
  const buffers: Buffer[] = [];
  const allocate = <Format extends 'uint32' | 'uint32x2' | 'uint32x4'>(
    id: string,
    format: Format,
    length: number,
    words: number
  ): RunBuffer<Format> => {
    const buffer = createOutputBuffer(device, words);
    buffers.push(buffer);
    return {
      id,
      format,
      length,
      buffer,
      view: importGraphBuffer(graph, id, buffer, format, length),
      write: data => buffer.write(data)
    };
  };
  const unexpectedAllocation = () => {
    throw new Error('Fixture supplies the edge adapter buffers');
  };
  const builder: RunBuilder = {
    device,
    graph,
    upload: unexpectedAllocation,
    allocate: unexpectedAllocation,
    own: unexpectedAllocation
  };
  const components = allocate('components', 'uint32x4', 2, 8);
  const status = allocate('status', 'uint32x4', 2, 8);
  const edges = allocate('edges', 'uint32x2', 4, 8);
  const categories = allocate('categories', 'uint32', 4, 4);
  // Stale component row 1 would link vertices 2 and 3 if the shader scanned all slots.
  components.write(Uint32Array.of(0, 2, 0, 0, 2, 2, 0, 0));
  addOffsetCurveComponentEdges(builder, components, status, edges, categories);
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    const plan = graph.compile();
    compiled = plan;
    const check = async (
      statusWords: Uint32Array,
      expectedEdges: number[],
      expectedIds: number[]
    ) => {
      status.write(statusWords);
      submitGraph(device, plan);
      expect(await readUint32(edges.buffer)).toEqual(expectedEdges);
      expect(await readUint32(categories.buffer)).toEqual(expectedIds);
    };
    const missing = 0xffffffff;
    const liveEdges = [0, 1, 1, 1, 2, 2, 3, 3];
    const liveIds = [0, 0, missing, missing];
    const suppressedEdges = [0, 0, 1, 1, 2, 2, 3, 3];
    const suppressedIds = [missing, missing, missing, missing];
    await check(Uint32Array.of(1, 2, 1, 2, 0, 0, 0, 0), liveEdges, liveIds);
    // Keep the stale component's vertices within the readable prefix to isolate the component bound.
    await check(Uint32Array.of(1, 4, 1, 4, 0, 0, 0, 0), liveEdges, liveIds);
    for (const lossWord of [4, 5, 6]) {
      const statusWords = Uint32Array.of(1, 2, 1, 2, 0, 0, 0, 0);
      statusWords[lossWord] = 1;
      await check(statusWords, suppressedEdges, suppressedIds);
    }
    await check(Uint32Array.of(0, 0, 1, 2, 0, 0, 1, 0), suppressedEdges, suppressedIds);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
