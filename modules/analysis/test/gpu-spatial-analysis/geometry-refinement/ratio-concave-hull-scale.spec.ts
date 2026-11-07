// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURatioConcaveHull} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  function output<Format extends 'uint32' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) {
    const components = format.endsWith('x2') ? 2 : format.endsWith('x3') ? 3 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length)
    };
  }
  return {
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

function createRandomPositions(count: number, extent: number): number[] {
  let state = 12345;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  return Array.from({length: count * 2}, () => Math.fround(next() * extent));
}

async function runRatioHull(pointCount: number, ratio: number, allowHoles: boolean) {
  const {graph, input, output, run} = await setup();
  const capacity = getDelaunayMaximumTriangleCount(pointCount);
  const triangles = output(capacity, 'uint32x3');
  const triangleValidity = output(capacity, 'uint32');
  const edges = output(capacity * 3, 'uint32x2');
  const edgeValidity = output(capacity * 3, 'uint32');
  const converged = output(1, 'uint32');
  graph.add(
    new GPURatioConcaveHull({
      positions: input(createRandomPositions(pointCount, 100), 'float32x2'),
      ratio,
      allowHoles,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: edges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: converged.view}
      }
    })
  );
  const destroy = run();
  const result = {
    converged: (await readUint32(converged.buffer, 1))[0],
    triangleValidity: await readUint32(triangleValidity.buffer, capacity),
    edges: await readUint32(edges.buffer, capacity * 6),
    edgeValidity: await readUint32(edgeValidity.buffer, capacity * 3)
  };
  destroy();
  return result;
}

for (const allowHoles of [false, true]) {
  it(`ratio hull erosion converges on 2000 random points with a closed boundary (holes: ${allowHoles})`, async () => {
    const start = performance.now();
    const result = await runRatioHull(2000, 0.15, allowHoles);
    const elapsed = performance.now() - start;
    expect(result.converged).toBe(1);
    const retained = result.triangleValidity.reduce((sum, value) => sum + value, 0);
    expect(retained).toBeGreaterThan(100);
    expect(retained).toBeLessThan(result.triangleValidity.length);
    // Every boundary vertex has equal in and out degree: the retained region is closed.
    const balance = new Map<number, number>();
    result.edgeValidity.forEach((valid, index) => {
      if (!valid) return;
      const from = result.edges[index * 2];
      const to = result.edges[index * 2 + 1];
      balance.set(from, (balance.get(from) ?? 0) + 1);
      balance.set(to, (balance.get(to) ?? 0) - 1);
    });
    expect(Array.from(balance.values()).every(value => value === 0)).toBe(true);
    // The former single-thread O(T^3) erosion took tens of seconds at this size.
    expect(elapsed).toBeLessThan(20000);
  }, 60000);
}

it('ratio hull reports the removal budget as not converged', async () => {
  const {graph, input, output, run} = await setup();
  const capacity = getDelaunayMaximumTriangleCount(200);
  const triangles = output(capacity, 'uint32x3');
  const triangleValidity = output(capacity, 'uint32');
  const edges = output(capacity * 3, 'uint32x2');
  const edgeValidity = output(capacity * 3, 'uint32');
  const converged = output(1, 'uint32');
  graph.add(
    new GPURatioConcaveHull({
      positions: input(createRandomPositions(200, 100), 'float32x2'),
      ratio: 0.05,
      maximumIterations: 1,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: edges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: converged.view}
      }
    })
  );
  const destroy = run();
  expect((await readUint32(converged.buffer, 1))[0]).toBe(0);
  destroy();
});
