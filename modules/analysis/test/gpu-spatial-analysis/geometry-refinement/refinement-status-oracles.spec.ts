// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUPolygonPrecisionReduction,
  GPULinePrecisionReduction,
  GPURatioConcaveHull,
  GPUSubdividePolygons,
  GPUVectorStraightSkeleton
} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import {GPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull-operation';
import {CPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {createTopologyOutput} from './topology-output';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('precision reduction clears poisoned success status for an exact square and exact line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  let serial = 0;
  const topology = createTopologyOutput((format, length) =>
    output(`topology-${serial++}`, format, length)
  );
  graph.add(
    new GPUPolygonPrecisionReduction({
      gridSize: 1,
      lattice: output('lattice', 'float32', 4),
      intersectionCapacity: 8,
      polygons: {
        kind: 'polygons',
        positions: input('square', 'float32x2', Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4)),
        ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
        polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
        featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1))
      },
      output: topology
    })
  );
  const lineStatus = output('line-status', 'uint32', 1);
  graph.add(
    new GPULinePrecisionReduction({
      gridSize: 1,
      mode: 'valid-output',
      positions: input('line', 'float32x2', Float32Array.of(0, 0, 2, 0)),
      lineOffsets: input('lines', 'uint32', Uint32Array.of(0, 2)),
      output: {
        positions: output('line-points', 'float32x2', 2),
        lineOffsets: output('out-lines', 'uint32', 2),
        status: lineStatus
      }
    })
  );
  const compiled = graph.compile();
  // Exact integer coordinates remain unchanged: one polygon, no truncation,
  // line success code 0. Every zero word starts poisoned, not at its expectation.
  try {
    expect(
      await fixture.submitTwice(compiled, () =>
        fixture.readRows([topology.status.count, topology.status.overflow, lineStatus])
      )
    ).toEqual([1, 0, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('ratio hull has no removable triangle on a simplex even with zero removal work', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0, 0, 1));
  const triangleValidity = output('validity', 'uint32', 1),
    converged = output('converged', 'uint32', 1);
  graph.add(
    new GPURatioConcaveHull({
      positions,
      ratio: 0,
      maximumIterations: 0,
      output: {
        triangles: output('triangles', 'uint32x3', 1),
        triangleValidity,
        boundaryEdges: output('edges', 'uint32x2', 3),
        edgeValidity: output('edge-validity', 'uint32', 3),
        status: {converged}
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const frame of [
      {positions: [0, 0, 1, 0, 0, 1], validity: 1},
      {positions: [0, 0, 1, 0, 2, 0], validity: 0},
      {positions: [0, 0, 1, 0, 0, 1], validity: 1}
    ]) {
      fixture.writeInput(positions, Float32Array.from(frame.positions));
      expect(
        await fixture.submitTwice(compiled, () => fixture.readRows([triangleValidity, converged]))
      ).toEqual([frame.validity, 1]);
      for (const Operation of [GPURatioConcaveHullOperation, CPURatioConcaveHullOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {positions: {format: 'float32x2', values: Float32Array.from(frame.positions)}},
          operations: [
            new Operation({
              positions: 'positions',
              pointCount: 3,
              ratio: 0,
              maximumIterations: 0,
              output: {
                triangles: 'triangles',
                triangleValidity: 'validity',
                boundaryEdges: 'edges',
                edgeValidity: 'edge-validity',
                status: {converged: 'converged'}
              }
            })
          ],
          outputs: ['validity', 'converged']
        });
        expect(Array.from(result['validity'])).toEqual([frame.validity]);
        expect(Array.from(result['converged'])).toEqual([1]);
      }
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const budget of [0, 1]) {
  it(`ratio hull validity identifies the unique real removal (budget ${budget})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = Float32Array.of(0, 0, 6, 0, 0, 4, 1, 1);
    // SciPy independently gives three triangles around site 3. Exterior
    // squared lengths 36,16,52 make edge {1,2} the unique longest candidate.
    // Remove {1,2,3}; site 3 is then on the boundary, so erosion is complete.
    // See refinement-status-oracles.oracle.py. Compare identities, not rows.
    const expectHull = (
      triangles: ArrayLike<number>,
      validity: ArrayLike<number>,
      converged: ArrayLike<number>
    ) => {
      expect(triangles.length).toBe(9);
      expect(validity.length).toBe(3);
      const byIdentity = Array.from({length: 3}, (_, row) => ({
        identity: Array.from({length: 3}, (_, corner) => triangles[3 * row + corner])
          .sort((first, second) => first - second)
          .join(','),
        validity: validity[row]
      })).sort((first, second) => first.identity.localeCompare(second.identity));
      expect(byIdentity).toEqual([
        {identity: '0,1,3', validity: 1},
        {identity: '0,2,3', validity: 1},
        {identity: '1,2,3', validity: budget === 0 ? 1 : 0}
      ]);
      expect(Array.from(converged)).toEqual([budget === 0 ? 0 : 1]);
    };
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const triangles = output('triangles', 'uint32x3', 3),
      validity = output('validity', 'uint32', 3),
      converged = output('converged', 'uint32', 1);
    graph.add(
      new GPURatioConcaveHull({
        positions: input('positions', 'float32x2', positions),
        ratio: 0,
        maximumIterations: budget,
        output: {
          triangles,
          triangleValidity: validity,
          boundaryEdges: output('edges', 'uint32x2', 9),
          edgeValidity: output('edge-validity', 'uint32', 9),
          status: {converged}
        }
      })
    );
    const compiled = graph.compile();
    try {
      const result = await fixture.submitTwice(compiled, async () => ({
        triangles: await fixture.read(triangles),
        validity: await fixture.read(validity),
        converged: await fixture.read(converged)
      }));
      expectHull(result.triangles, result.validity, result.converged);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
    for (const Operation of [GPURatioConcaveHullOperation, CPURatioConcaveHullOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {positions: {format: 'float32x2', values: positions}},
        operations: [
          new Operation({
            positions: 'positions',
            pointCount: 4,
            ratio: 0,
            maximumIterations: budget,
            output: {
              triangles: 'triangles',
              triangleValidity: 'validity',
              boundaryEdges: 'edges',
              edgeValidity: 'edge-validity',
              status: {converged: 'converged'}
            }
          })
        ],
        outputs: ['triangles', 'validity', 'converged']
      });
      expectHull(result['triangles'], result['validity'], result['converged']);
    }
  });
}

for (const method of ['convex', 'rectilinear'] as const) {
  it(`subdivision status observes all words for one square (${method})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const status = output('status', 'uint32x4', 1);
    graph.add(
      new GPUSubdividePolygons({
        method,
        maximumVertices: 4,
        positions: input('positions', 'float32x2', Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4)),
        ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
        polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
        featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
        output: {
          positions: output('out-positions', 'float32x2', 4),
          pieces: output('pieces', 'uint32x4', 1),
          status
        }
      })
    );
    const compiled = graph.compile();
    // The convex four-vertex polygon already satisfies the limit: exactly one
    // required/written piece, no capacity loss and no invalid/incomplete work.
    try {
      expect(await fixture.submitTwice(compiled, () => fixture.read(status))).toEqual([1, 1, 0, 0]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('vector skeleton publishes every status word at a zero event budget', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const status = output('status', 'uint32x4', 1);
  graph.add(
    new GPUVectorStraightSkeleton({
      maximumVertices: 4,
      maximumEvents: 0,
      segmentCapacity: 8,
      offsetDistance: 0,
      positions: input('positions', 'float32x2', Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2)),
      ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
      polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
      featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
      output: {
        segments: output('segments', 'float32x4', 8),
        sourceEdges: output('edges', 'uint32x2', 8),
        offsetPositions: output('offset', 'float32x2', 4),
        offsetLinks: output('links', 'uint32x2', 4),
        status
      }
    })
  );
  const compiled = graph.compile();
  // Snapshot at time 0 keeps four input vertices. No event can be processed,
  // so no arc exists and the unresolved rectangle-collapse event forces work bit 4.
  try {
    expect(await fixture.submitTwice(compiled, () => fixture.read(status))).toEqual([0, 0, 4, 4]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
