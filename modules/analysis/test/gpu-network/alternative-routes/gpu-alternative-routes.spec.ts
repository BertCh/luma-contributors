// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUAlternativeRoutes} from '../../../src/gpu-network/alternative-routes/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from '../../oracles/fixtures/alternative-routes.json';

async function setup(
  caseIndex: number,
  routeCount: number,
  maximumFrontierSize = 1024,
  maximumExpansions = 100000
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const fixture = reference.cases[caseIndex];
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let next = 0;
  function input<Format extends 'uint32' | 'float32'>(values: number[], format: Format) {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `input-${next++}`, buffer, format, values.length)
    };
  }
  function output<Format extends 'uint32' | 'float32'>(length: number, format: Format) {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length)};
  }
  const weights = input(fixture.weights, 'float32');
  const nodeCount = fixture.offsets.length - 1;
  const vertices = output(routeCount * nodeCount, 'uint32');
  const edges = output(routeCount * nodeCount, 'uint32');
  const lengths = output(routeCount, 'uint32');
  const costs = output(routeCount, 'float32');
  const counts = output(1, 'uint32');
  const complete = output(1, 'uint32');
  graph.add(
    new GPUAlternativeRoutes({
      topology: {
        offsets: input(fixture.offsets, 'uint32').view,
        neighbors: input(fixture.neighbors, 'uint32').view,
        weights: weights.view
      },
      sources: input([fixture.source], 'uint32').view,
      targets: input([fixture.target], 'uint32').view,
      routeCount,
      maximumFrontierSize,
      maximumExpansions,
      output: {
        vertices: vertices.view,
        edges: edges.view,
        lengths: lengths.view,
        costs: costs.view,
        counts: counts.view,
        complete: complete.view
      }
    })
  );
  const compiled = graph.compile();
  return {
    weights: weights.buffer,
    run: () => submitGraph(device, compiled, undefined),
    destroy: () => {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    },
    async read() {
      const count = (await readUint32(counts.buffer, 1))[0];
      const routeLengths = await readUint32(lengths.buffer, routeCount);
      const routeCosts = await readFloat32(costs.buffer, routeCount);
      const routeVertices = await readUint32(vertices.buffer, routeCount * nodeCount);
      const routeEdges = await readUint32(edges.buffer, routeCount * nodeCount);
      return {
        complete: (await readUint32(complete.buffer, 1))[0],
        count,
        routes: Array.from({length: count}, (_, route) => ({
          cost: routeCosts[route],
          vertices: Array.from(
            routeVertices.slice(route * nodeCount, route * nodeCount + routeLengths[route])
          ),
          edges: Array.from(
            routeEdges.slice(route * nodeCount, route * nodeCount + routeLengths[route] - 1)
          )
        })),
        routeLengths: Array.from(routeLengths),
        routeCosts: Array.from(routeCosts),
        routeVertices: Array.from(routeVertices),
        routeEdges: Array.from(routeEdges)
      };
    }
  };
}

for (const [caseIndex, fixture] of reference.cases.entries()) {
  for (const routeCount of [1, 3, 16]) {
    it(`GPUAlternativeRoutes matches NetworkX ${fixture.name}, k=${routeCount}`, async () => {
      const test = await setup(caseIndex, routeCount);
      try {
        test.run();
        const result = await test.read();
        expect(result.complete).toBe(1);
        expect(result.routes).toEqual(fixture.routes.slice(0, routeCount));
        const nodeCount = fixture.offsets.length - 1;
        for (let route = 0; route < routeCount; route++) {
          const length = result.routeLengths[route];
          expect(
            result.routeVertices.slice(route * nodeCount + length, (route + 1) * nodeCount)
          ).toEqual(Array(nodeCount - length).fill(0xffffffff));
          expect(
            result.routeEdges.slice(
              route * nodeCount + Math.max(0, length - 1),
              (route + 1) * nodeCount
            )
          ).toEqual(Array(nodeCount - Math.max(0, length - 1)).fill(0xffffffff));
          if (route >= result.count) {
            expect(length).toBe(0);
            expect(result.routeCosts[route]).toBe(Infinity);
          }
        }
      } finally {
        test.destroy();
      }
    });
  }
}

it('GPUAlternativeRoutes stops explicitly on frontier overflow and edge budget exhaustion', async () => {
  for (const [capacity, budget] of [
    [1, 1000],
    [1024, 1]
  ]) {
    const test = await setup(0, 3, capacity, budget);
    try {
      test.run();
      const result = await test.read();
      expect(result.complete).toBe(0);
      expect(result.routes).toEqual(reference.cases[0].routes.slice(0, result.count));
    } finally {
      test.destroy();
    }
  }
});

it('GPUAlternativeRoutes reinitializes heap and outputs on GPU input updates', async () => {
  const test = await setup(0, 16);
  try {
    test.run();
    expect((await test.read()).routes).toEqual(reference.cases[0].routes);
    test.weights.write(Float32Array.from(reference.cases[1].weights));
    test.run();
    expect((await test.read()).routes).toEqual(reference.cases[1].routes);
    test.weights.write(
      Float32Array.from([Infinity, NaN, -1, Infinity, Infinity, Infinity, Infinity, Infinity])
    );
    test.run();
    const empty = await test.read();
    expect(empty.complete).toBe(1);
    expect(empty.count).toBe(0);
    expect(empty.routeCosts.every(value => value === Infinity)).toBe(true);
    test.weights.write(Float32Array.from([3e38, 3e38, 3e38, 0, 3e38, 3e38, 0, 3e38]));
    test.run();
    expect((await test.read()).count).toBe(0);
  } finally {
    test.destroy();
  }
});

it('GPUAlternativeRoutes certifies k=1 after three examinations despite expensive earlier CSR branches', async () => {
  const test = await setup(4, 1, 8, 3);
  try {
    test.run();
    const result = await test.read();
    expect(result.complete).toBe(1);
    expect(result.routes).toEqual(reference.cases[4].routes.slice(0, 1));
  } finally {
    test.destroy();
  }
});
