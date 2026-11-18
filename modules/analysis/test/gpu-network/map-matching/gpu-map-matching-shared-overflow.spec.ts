// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUMapMatching,
  getGPUMapMatchingParameterValues,
  type GPUMapMatchingProps,
  type GPUMapMatchingSettings
} from '../../../src/gpu-network/map-matching/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes shared overflow from many tracks and clears it on the next encoding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let nextView = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `overflow-input-${nextView++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `overflow-output-${nextView++}`, buffer, 'uint32', length)
    };
  };
  const trackCount = 512;
  const pointCount = trackCount * 2;
  const coordinates = new Float32Array(pointCount * 2);
  for (let track = 0; track < trackCount; track++) {
    coordinates[track * 4] = 0.25;
    coordinates[track * 4 + 2] = 1.75;
  }
  const positions = input(coordinates, 'float32x2');
  const matchedEdges = output(pointCount);
  const traversedEdges = output(trackCount);
  const counts = output(trackCount);
  const overflow = output(1);
  graph.add(
    new GPUMapMatching({
      positions: positions.view,
      trackOffsets: input(
        Uint32Array.from({length: trackCount + 1}, (_, track) => track * 2),
        'uint32'
      ).view,
      topology: {
        offsets: input(Uint32Array.of(0, 1, 2, 2), 'uint32').view,
        neighbors: input(Uint32Array.of(1, 2), 'uint32').view,
        nodePositions: input(Float32Array.of(0, 0, 1, 0, 2, 0), 'float32x2').view
      },
      parameters: input(
        getGPUMapMatchingParameterValues({sigma: 0.1, beta: 1, searchRadius: 0.2}),
        'float32'
      ).view,
      cellSize: 1,
      bounds: {minimum: [-1, -1], maximum: [3, 1]},
      candidateCount: 1,
      traversedEdgeCapacity: 1,
      output: {
        matchedEdges: matchedEdges.view,
        traversedEdges: {
          edges: traversedEdges.view,
          counts: counts.view,
          status: {overflow: overflow.view}
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    await submitGraph(device, compiled, undefined);
    // Every track traverses both directed chain edges; only its first edge fits capacity one.
    expect(Array.from(await readUint32(matchedEdges.buffer, pointCount))).toEqual(
      Array.from({length: pointCount}, (_, point) => point % 2)
    );
    expect(Array.from(await readUint32(traversedEdges.buffer, trackCount))).toEqual(
      new Array(trackCount).fill(0)
    );
    expect(Array.from(await readUint32(counts.buffer, trackCount))).toEqual(
      new Array(trackCount).fill(2)
    );
    expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([1]);
    // Same-edge forward fixes need only one route entry: no writer may leave a stale flag.
    for (let track = 0; track < trackCount; track++) coordinates[track * 4 + 2] = 0.75;
    positions.buffer.write(coordinates);
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(counts.buffer, trackCount))).toEqual(
      new Array(trackCount).fill(1)
    );
    expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([0]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

type ReviewFixture = {
  nodes: Float32Array;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights?: Float32Array;
  positions: Float32Array;
  tracks: Uint32Array;
};

function createReviewGraph(
  device: Device,
  fixture: ReviewFixture,
  options: Pick<GPUMapMatchingProps, 'bounds' | 'cellSize' | 'spatialContext'> & {
    budget?: number;
    entryCapacity?: number;
    parameters?: GPUMapMatchingSettings;
    edgeCapacity?: number;
  }
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let nextView = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `review-input-${nextView++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `review-output-${nextView++}`, buffer, format, length)
    };
  };
  const pointCount = fixture.positions.length / 2;
  const trackCount = fixture.tracks.length - 1;
  const edgeCapacity = options.edgeCapacity ?? 128;
  const positions = input(fixture.positions, 'float32x2');
  const neighbors = input(fixture.neighbors, 'uint32');
  const matched = output(pointCount, 'uint32');
  const edges = output(trackCount * edgeCapacity, 'uint32');
  const counts = output(trackCount, 'uint32');
  const overflow = output(1, 'uint32');
  const candidateOverflow = output(1, 'uint32');
  const routeOverflow = output(1, 'uint32');
  const traversedOverflow = output(1, 'uint32');
  const likelihood = output(trackCount, 'float32');
  graph.add(
    new GPUMapMatching({
      positions: positions.view,
      trackOffsets: input(fixture.tracks, 'uint32').view,
      topology: {
        offsets: input(fixture.offsets, 'uint32').view,
        neighbors: neighbors.view,
        weights: fixture.weights ? input(fixture.weights, 'float32').view : undefined,
        nodePositions: input(fixture.nodes, 'float32x2').view
      },
      parameters: input(
        getGPUMapMatchingParameterValues(
          options.parameters ?? {sigma: 1, beta: 1, searchRadius: 0.1}
        ),
        'float32'
      ).view,
      bounds: options.bounds,
      cellSize: options.cellSize,
      spatialContext: options.spatialContext,
      candidateCount: 1,
      routeNodeBudget: options.budget ?? 64,
      entryCapacity: options.entryCapacity,
      traversedEdgeCapacity: edgeCapacity,
      output: {
        matchedEdges: matched.view,
        trackLogLikelihoods: likelihood.view,
        status: {
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view,
          routeOverflow: routeOverflow.view
        },
        traversedEdges: {
          edges: edges.view,
          counts: counts.view,
          status: {overflow: traversedOverflow.view}
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    positions: positions.buffer,
    neighbors: neighbors.buffer,
    run: () => submitGraph(device, compiled, undefined),
    read: async () => ({
      matched: Array.from(await readUint32(matched.buffer, pointCount)),
      edges: Array.from(await readUint32(edges.buffer, trackCount * edgeCapacity)),
      counts: Array.from(await readUint32(counts.buffer, trackCount)),
      overflow: (await readUint32(overflow.buffer, 1))[0],
      candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
      routeOverflow: (await readUint32(routeOverflow.buffer, 1))[0],
      traversedOverflow: (await readUint32(traversedOverflow.buffer, 1))[0],
      likelihood: Array.from(await readFloat32(likelihood.buffer, trackCount))
    }),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it('three distant identical fixes have a negative 2.4-billion log-likelihood', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createReviewGraph(
    device,
    {
      nodes: Float32Array.of(0, 0, 1, 0),
      offsets: Uint32Array.of(0, 1, 1),
      neighbors: Uint32Array.of(1),
      positions: Float32Array.of(0.5, 40000, 0.5, 40000, 0.5, 40000),
      tracks: Uint32Array.of(0, 3)
    },
    {
      bounds: {minimum: [-1, -1], maximum: [2, 80001]},
      cellSize: 50000,
      parameters: {sigma: 1, beta: 1, searchRadius: 40001}
    }
  );
  try {
    await test.run();
    const result = await test.read();
    expect(result.matched).toEqual([0, 0, 0]);
    // Three Gaussian emissions -0.5 * 40000^2; log constants round away at this f32 magnitude.
    expect(result.likelihood[0]).toBeCloseTo(Math.fround(-2400000000), -3);
    expect(result.overflow).toBe(0);
  } finally {
    test.destroy();
  }
});

for (const budget of [32, 64, 128]) {
  it(`clears hashes between nontrivial reconstructed cycle searches and encodings, budget ${budget}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const nodes = Float32Array.from(
      Array.from({length: 12}, (_, node) => [
        100 * Math.cos((node * Math.PI) / 6),
        100 * Math.sin((node * Math.PI) / 6)
      ]).flat()
    );
    const points = (firstEdge: number) =>
      Float32Array.from(
        Array.from({length: 4}, (_, point) => {
          const edge = (firstEdge + point * 6) % 12;
          const next = (edge + 1) % 12;
          return [
            (nodes[edge * 2] + nodes[next * 2]) / 2,
            (nodes[edge * 2 + 1] + nodes[next * 2 + 1]) / 2
          ];
        }).flat()
      );
    const test = createReviewGraph(
      device,
      {
        nodes,
        offsets: Uint32Array.from({length: 13}, (_, node) => node),
        neighbors: Uint32Array.from({length: 12}, (_, node) => (node + 1) % 12),
        positions: points(0),
        tracks: Uint32Array.of(0, 4)
      },
      {budget, bounds: {minimum: [-110, -110], maximum: [110, 110]}, cellSize: 50}
    );
    try {
      for (const firstEdge of [0, 3, 0]) {
        test.positions.write(points(firstEdge));
        await test.run();
        const result = await test.read();
        expect(result.matched).toEqual([
          firstEdge,
          (firstEdge + 6) % 12,
          firstEdge,
          (firstEdge + 6) % 12
        ]);
        expect(result.counts).toEqual([19]);
        expect(result.edges.slice(0, 19)).toEqual(
          Array.from({length: 19}, (_, edge) => (firstEdge + edge) % 12)
        );
        expect([result.overflow, result.routeOverflow, result.traversedOverflow]).toEqual([
          0, 0, 0
        ]);
      }
    } finally {
      test.destroy();
    }
  });
}

it('reports concurrent grid and route-budget overflow and clears flags and counts on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const trackCount = 512;
  const nodes = Float32Array.from([
    ...Array.from({length: 41}, (_, node) => [node, 0]).flat(),
    80,
    10,
    100,
    10
  ]);
  const neighbors = Uint32Array.from([...Array.from({length: 40}, (_, node) => node + 1), 42]);
  const positions = Float32Array.from(
    Array.from({length: trackCount}, () => [0.25, 0, 35.75, 0]).flat()
  );
  const test = createReviewGraph(
    device,
    {
      nodes,
      offsets: Uint32Array.from({length: 44}, (_, node) =>
        node <= 40 ? node : node === 41 ? 40 : 41
      ),
      neighbors,
      positions,
      tracks: Uint32Array.from({length: trackCount + 1}, (_, track) => track * 2)
    },
    {
      budget: 8,
      entryCapacity: 60,
      edgeCapacity: 1,
      cellSize: 2,
      bounds: {minimum: [-1, -1], maximum: [101, 21]}
    }
  );
  try {
    await test.run();
    const dropped = await test.read();
    expect(dropped.matched).toEqual(
      Array.from({length: trackCount * 2}, (_, point) => (point % 2 === 0 ? 0 : 35))
    );
    expect(dropped.counts).toEqual(new Array(trackCount).fill(3));
    expect([dropped.overflow, dropped.routeOverflow, dropped.traversedOverflow]).toEqual([
      0,
      trackCount,
      1
    ]);
    expect(dropped.candidateOverflow).toBe(1);
    // The chain contributes exactly 60 cell entries. Disabling the later, distant edge clears
    // grid overflow; same-edge fixes avoid route searches and need one traversed slot per track.
    neighbors[40] = 0xffffffff;
    test.neighbors.write(neighbors);
    for (let track = 0; track < trackCount; track++) positions[track * 4 + 2] = 0.75;
    test.positions.write(positions);
    await test.run();
    const clean = await test.read();
    expect(clean.matched).toEqual(new Array(trackCount * 2).fill(0));
    expect(clean.counts).toEqual(new Array(trackCount).fill(1));
    expect([clean.overflow, clean.routeOverflow, clean.traversedOverflow]).toEqual([0, 0, 0]);
    expect(clean.candidateOverflow).toBe(0);
  } finally {
    test.destroy();
  }
});

it('map matching ignores routing weights when measuring a same-edge transition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const beta = Math.fround(1e38);
  const test = createReviewGraph(
    device,
    {
      nodes: Float32Array.of(0, 0, 1, 0),
      offsets: Uint32Array.of(0, 1, 1),
      neighbors: Uint32Array.of(1),
      weights: Float32Array.of(1e38),
      positions: Float32Array.of(0.25, 0, 0.75, 0),
      tracks: Uint32Array.of(0, 2)
    },
    {
      bounds: {minimum: [-1, -1], maximum: [2, 1]},
      cellSize: 1,
      parameters: {sigma: 1, beta, searchRadius: 0.1, routeSlack: beta}
    }
  );
  try {
    await test.run();
    const result = await test.read();
    expect(result.matched).toEqual([0, 0]);
    // Weights are unused by contract: both the geometric route and straight distance are 0.5,
    // so their difference is zero despite the supplied routing weight of 1e38.
    expect(Math.abs(result.likelihood[0] - (-Math.log(2 * Math.PI) - Math.log(beta)))).toBeLessThan(
      2e-4
    );
    expect(result.overflow).toBe(0);
  } finally {
    test.destroy();
  }
});

it('map matching reconstructs a route exactly at a large inclusive search bound', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const test = createReviewGraph(
    device,
    {
      // Order CSR rows so endpoint ties select the short origin/destination edges.
      nodes: Float32Array.of(16777218, 0, 0, 0, 1, 0, 16777220, 0),
      offsets: Uint32Array.of(0, 1, 2, 3, 3),
      neighbors: Uint32Array.of(3, 2, 0),
      positions: Float32Array.of(1, 0, 16777218, 0),
      tracks: Uint32Array.of(0, 2)
    },
    {
      bounds: {minimum: [-1, -1], maximum: [16777222, 1]},
      cellSize: 16777222,
      parameters: {sigma: 1, beta: 1, searchRadius: 1, routeFactor: 0, routeSlack: 16777216}
    }
  );
  try {
    await test.run();
    const result = await test.read();
    expect(result.matched).toEqual([1, 0]);
    // The middle edge has float32 length 2^24. It must be found, not reported as a break.
    expect(result.counts).toEqual([3]);
    expect(result.edges.slice(0, 3)).toEqual([1, 2, 0]);
    expect([result.overflow, result.routeOverflow, result.traversedOverflow]).toEqual([0, 0, 0]);
  } finally {
    test.destroy();
  }
});

it('map matching preserves a nonzero transition penalty and cross-cell search at near-maximum scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sphereRadius = Math.fround(1e38);
  const beta = sphereRadius;
  const sigma = Math.fround(1e35);
  const firstLatitude = Math.fround(0.02);
  const lastLatitude = Math.fround(9.98);
  const test = createReviewGraph(
    device,
    {
      // Directed three-edge detour: east at the equator, north, then west at latitude ten.
      nodes: Float32Array.of(0, 0, 90, 0, 90, 10, 0, 10),
      offsets: Uint32Array.of(0, 1, 2, 3, 3),
      neighbors: Uint32Array.of(1, 2, 3),
      positions: Float32Array.of(45, firstLatitude, 45, lastLatitude),
      tracks: Uint32Array.of(0, 2)
    },
    {
      // The first fix and its edge lie on opposite sides of the grid boundary at latitude .01.
      // A reciprocal-flushed angular search extent misses that edge altogether.
      bounds: {minimum: [-1, -44.99], maximum: [91, 11]},
      cellSize: 45,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', sphereRadius},
      parameters: {sigma, beta, searchRadius: 3e35, routeFactor: 1, routeSlack: 2e38}
    }
  );
  try {
    await test.run();
    const result = await test.read();
    expect(result.matched).toEqual([0, 2]);
    expect(result.counts).toEqual([3]);
    expect(result.edges.slice(0, 3)).toEqual([0, 1, 2]);
    const radians = Math.PI / 180;
    // Both projections are midpoints. Spherical edge angles are pi/2, ten degrees,
    // and acos(sin(ten degrees)^2); the straight hop is the latitude difference.
    const routeAngle = Math.PI / 4 + 10 * radians + Math.acos(Math.sin(10 * radians) ** 2) / 2;
    const straightAngle = (lastLatitude - firstLatitude) * radians;
    const normalizedFirst = (sphereRadius / sigma) * firstLatitude * radians;
    const normalizedLast = (sphereRadius / sigma) * (10 - lastLatitude) * radians;
    const expected =
      -Math.log(2 * Math.PI) -
      2 * Math.log(sigma) -
      (normalizedFirst ** 2 + normalizedLast ** 2) / 2 -
      Math.log(beta) -
      (sphereRadius / beta) * (routeAngle - straightAngle);
    expect(Math.abs(result.likelihood[0] - expected)).toBeLessThan(2e-4);
    expect([result.overflow, result.routeOverflow, result.traversedOverflow]).toEqual([0, 0, 0]);
  } finally {
    test.destroy();
  }
});
