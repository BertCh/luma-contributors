// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {GPUSphericalPolygonOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createVectorView, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from './output-oracle-fixture';
import {createSphericalPredicateScenes} from './spherical-predicate-scenes.test-utils';

it.each(
  createSphericalPredicateScenes()
)('permanent spherical predicate control: $name', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reversed of [false, true])
    for (const boundary of ['open', 'closed'] as const) {
      const fixture = createOracleFixture(device);
      const contains = fixture.output('uint32', 1);
      const status = fixture.output('uint32', 1);
      fixture.graph.add(
        new GPUSphericalPolygonContains({
          positions: fixture.input(
            'float32x3',
            (reversed ? [...scene.vertices].reverse() : scene.vertices).flat()
          ),
          ringOffsets: fixture.input('uint32', [0, scene.vertices.length]),
          polygonOffsets: fixture.input('uint32', [0, 1]),
          queries: fixture.input('float32x3', scene.query),
          polygonIndices: fixture.input('uint32', [0]),
          tolerance: scene.tolerance,
          boundary,
          output: {contains: contains.view, status: status.view}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        for (let submission = 0; submission < 2; submission++) {
          contains.poison();
          status.poison();
          submitGraph(device, compiled, undefined);
          expect(await status.read(), `${reversed} ${boundary}`).toEqual([scene.status]);
          expect(await contains.read(), `${reversed} ${boundary}`).toEqual([
            scene.status === 2 ? Number(boundary === 'closed') : scene.contains
          ]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});

it.each([
  'closed',
  'open'
] as const)('spherical endpoint caps measure direction rather than radial error (%s)', async boundary => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const query = [1.00002, -3e-7, -4e-7];
  const normalizedQuery = query.map(value => value / Math.hypot(...query));
  const contains = fixture.output('uint32', 2);
  const status = fixture.output('uint32', 2);
  // atan2(hypot(3e-7,4e-7),1.00002) < 1e-6 radians from the +x
  // endpoint. Both accepted radii describe a direction in the boundary cap.
  fixture.graph.add(
    new GPUSphericalPolygonContains({
      positions: fixture.input('float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      ringOffsets: fixture.input('uint32', [0, 3]),
      polygonOffsets: fixture.input('uint32', [0, 1]),
      queries: fixture.input('float32x3', [...query, ...normalizedQuery]),
      polygonIndices: fixture.input('uint32', [0, 0]),
      boundary,
      output: {contains: contains.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      contains.poison();
      status.poison();
      submitGraph(device, compiled, undefined);
      expect(await contains.read()).toEqual(boundary === 'closed' ? [1, 1] : [0, 0]);
      expect(await status.read()).toEqual([2, 2]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it.each([
  {chunked: false, tolerance: 0.001, halfAngle: 0.00075},
  {chunked: true, tolerance: 0.001, halfAngle: 0.00075},
  {chunked: false, tolerance: 1e-6, halfAngle: 0.75e-6},
  {chunked: true, tolerance: 1e-6, halfAngle: 0.75e-6}
])('short spherical minor arc excludes its antipode (chunked=$chunked, tolerance=$tolerance)', async ({
  chunked,
  tolerance,
  halfAngle
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const lengths = chunked ? [1, 1] : [2];
  const queries = [
    [-1, 0, 0],
    [1, 0, 0]
  ];
  const queryViews = lengths.map((length, chunk) =>
    fixture.input('float32x3', (chunked ? [queries[chunk]] : queries).flat())
  );
  const selectors = lengths.map(length => fixture.input('uint32', Array(length).fill(0)));
  const contains = lengths.map(length => fixture.output('uint32', length));
  const statuses = lengths.map(length => fixture.output('uint32', length));
  // The short equatorial edge spans longitudes [-h,h]. The antipode is at
  // least pi/2 from this triangle; +x is exactly its true edge midpoint.
  fixture.graph.add(
    new GPUSphericalPolygonContains({
      positions: fixture.input('float32x3', [
        Math.cos(halfAngle),
        -Math.sin(halfAngle),
        0,
        Math.cos(halfAngle),
        Math.sin(halfAngle),
        0,
        0,
        0,
        1
      ]),
      ringOffsets: fixture.input('uint32', [0, 3]),
      polygonOffsets: fixture.input('uint32', [0, 1]),
      queries: chunked ? createVectorView('queries', 'float32x3', queryViews) : queryViews[0],
      polygonIndices: chunked ? createVectorView('selectors', 'uint32', selectors) : selectors[0],
      tolerance,
      boundary: 'closed',
      output: {
        contains: chunked
          ? createVectorView(
              'contains',
              'uint32',
              contains.map(output => output.view)
            )
          : contains[0].view,
        status: chunked
          ? createVectorView(
              'status',
              'uint32',
              statuses.map(output => output.view)
            )
          : statuses[0].view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [...contains, ...statuses]) output.poison();
      submitGraph(device, compiled, undefined);
      expect((await Promise.all(contains.map(output => output.read()))).flat()).toEqual([0, 1]);
      expect((await Promise.all(statuses.map(output => output.read()))).flat()).toEqual([0, 2]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('spherical polygon containment overwrites every status with the octant classification', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const status = fixture.output('uint32', 5);
  const contains = fixture.output('uint32', 5);
  const direction = 1 / Math.sqrt(3);
  // The positive octant has interior (+++), exterior (---), and boundary +z.
  // Zero is not a unit query; polygon selector 1 is outside the sole polygon.
  fixture.graph.add(
    new GPUSphericalPolygonContains({
      positions: fixture.input('float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      ringOffsets: fixture.input('uint32', [0, 3]),
      polygonOffsets: fixture.input('uint32', [0, 1]),
      queries: fixture.input('float32x3', [
        direction,
        direction,
        direction,
        -direction,
        -direction,
        -direction,
        0,
        0,
        1,
        0,
        0,
        0,
        direction,
        direction,
        direction
      ]),
      polygonIndices: fixture.input('uint32', [0, 0, 0, 0, 1]),
      output: {contains: contains.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      status.poison();
      contains.poison();
      submitGraph(device, compiled, undefined);
      expect(await status.read()).toEqual([0, 0, 2, 1, 1]);
      expect(await contains.read()).toEqual([1, 0, 1, 0, 0]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it.each([
  'complete',
  'empty',
  'vertex capacity',
  'invalid chart input'
] as const)('spherical polygon count and completeness overwrite poison: %s', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const operand = (coordinates: number[][]) => ({
    kind: 'polygons' as const,
    positions: fixture.input(
      'float32x3',
      coordinates.flatMap(([horizontal, vertical]) => {
        const length = Math.hypot(horizontal, vertical, 1);
        return mode === 'invalid chart input'
          ? [0, 0, 0]
          : [horizontal / length, vertical / length, 1 / length];
      })
    ),
    ringOffsets: fixture.input('uint32', [0, 3]),
    polygonOffsets: fixture.input('uint32', [0, 1]),
    featureOffsets: fixture.input('uint32', [0, 1])
  });
  // Strictly nested triangles have a single three-edge intersection and no
  // crossings. Gnomonic projection preserves this containment. Assembly needs
  // four raw vertices including the repeated closure, so two slots cannot hold it.
  const count = fixture.output('uint32', 1);
  const overflow = fixture.output('uint32', 1);
  const candidateOverflow = fixture.output('uint32', 1);
  const boundaryStatus = () => ({
    count: fixture.output('uint32', 1).view,
    requiredCount: fixture.output('uint32', 1).view,
    overflow: fixture.output('uint32', 1).view,
    candidateOverflow: fixture.output('uint32', 1).view
  });
  fixture.graph.add(
    new GPUSphericalPolygonOverlay({
      left: operand([
        [-2, -2],
        [2, -2],
        [0, 2]
      ]),
      right: operand(
        mode === 'empty'
          ? [
              [3, 0],
              [4, 0],
              [3, 1]
            ]
          : [
              [-0.5, -0.5],
              [0.5, -0.5],
              [0, 0.5]
            ]
      ),
      chartCenter: [0, 0, 1],
      operation: 'intersection',
      vertexTolerance: 2 ** -16,
      capacity: {intersections: 32, nodedSegments: 32},
      output: {
        geometry: {
          kind: 'polygons',
          positions: fixture.output('float32x3', mode === 'vertex capacity' ? 2 : 8).view,
          ringOffsets: fixture.output('uint32', 3).view,
          polygonOffsets: fixture.output('uint32', 3).view,
          featureOffsets: fixture.output('uint32', 3).view
        },
        complement: fixture.output('uint32', 1).view,
        status: {
          count: count.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view
        },
        boundary: {
          endpoints: fixture.output('float32x4', 8).view,
          operandIds: fixture.output('uint32', 8).view,
          sourceFeatureIds: fixture.output('uint32', 8).view,
          status: boundaryStatus()
        }
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      count.poison();
      overflow.poison();
      candidateOverflow.poison();
      submitGraph(device, compiled, undefined);
      expect(await count.read()).toEqual([Number(mode === 'complete')]);
      // Invalid unit vectors reject every input edge. That is incomplete input,
      // not a final geometry capacity overflow; candidateOverflow carries it.
      expect(await overflow.read()).toEqual([Number(mode === 'vertex capacity')]);
      expect(await candidateOverflow.read()).toEqual([Number(mode === 'invalid chart input')]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
