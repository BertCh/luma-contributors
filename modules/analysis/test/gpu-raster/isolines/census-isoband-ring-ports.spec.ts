// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/gpu-isoband-rings';
import {getGPUIsobandsParameterValues} from '../../../src/raster-analysis/isolines/isobands-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createCensusRasterHarness, expectCensusValues} from '../census-raster-oracle-utils';

/** Rotate a ring to its lexicographically first vertex, preserving the published winding. */
function rotateRing(coordinates: number[]) {
  const points = Array.from({length: coordinates.length / 2}, (_, index) =>
    coordinates.slice(index * 2, index * 2 + 2)
  );
  const first = points.reduce(
    (best, point, index) =>
      point[0] < points[best][0] || (point[0] === points[best][0] && point[1] < points[best][1])
        ? index
        : best,
    0
  );
  return [...points.slice(first), ...points.slice(0, first)].flat();
}

for (const mode of ['all', 'polygons', 'segments', 'minimal'] as const) {
  it(`census ring ports: one unit square with ${mode} outputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const grid = makeGPURasterMetadata({width: 2, height: 2});
    const includePolygons = mode === 'all' || mode === 'polygons';
    const includeSegments = mode === 'all' || mode === 'segments';
    const diagnostics = mode !== 'minimal';
    const result = await harness.run(
      new GPUIsobandRings({
        grid,
        edgeCapacity: 4,
        values: harness.input('float32', [1, 1, 1, 1]),
        breaks: harness.input('float32', [0.5]),
        parameters: harness.input(
          'float32',
          Array.from(
            getGPUIsobandsParameterValues(
              {breakCount: 1, firstBand: 1, lastBand: 1, extent: [0, 0, 2, 2]},
              grid
            )
          )
        ),
        output: {
          ringOffsets: harness.output('ringOffsets', 'uint32', 3),
          positions: harness.output('positions', 'float32x2', 5),
          ...(includeSegments
            ? {
                segmentRings: harness.output('segmentRings', 'uint32', 4),
                segmentVertices: harness.output('segmentVertices', 'uint32', 4),
                segmentFlags: harness.output('segmentFlags', 'uint32', 4)
              }
            : {}),
          ...(includePolygons
            ? {
                polygons: {
                  kind: 'polygons' as const,
                  positions: harness.output('polygonPositions', 'float32x2', 5),
                  ringOffsets: harness.output('polygonRingOffsets', 'uint32', 3),
                  polygonOffsets: harness.output('polygonOffsets', 'uint32', 3),
                  featureOffsets: harness.output('featureOffsets', 'uint32', 3),
                  sourceIds: harness.output('sourceIds', 'uint32', 2)
                }
              }
            : {}),
          status: {
            count: harness.output('count', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1),
            edgeOverflow: harness.output('edgeOverflow', 'uint32', 1),
            ...(diagnostics
              ? {
                  requiredCount: harness.output('requiredCount', 'uint32', 1),
                  touchingSegmentCount: harness.output('touchingSegmentCount', 'uint32', 1)
                }
              : {})
          }
        }
      })
    );
    // Closed form: the four sample centers bound [.5,1.5]^2, entirely in band 1.
    // A CCW square has four distinct vertices, five closed-ring rows, one polygon and no touches.
    // All coordinates are dyadic, so no floating tolerance is needed.
    const square = [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5];
    expectCensusValues(result['ringOffsets'], [0, 5, 5]);
    expectCensusValues(rotateRing(result['positions'].slice(0, 8)), square);
    expect(result['positions'].slice(8, 10)).toEqual(result['positions'].slice(0, 2));
    expectCensusValues(result['count'], [1]);
    expectCensusValues(result['overflow'], [0]);
    expectCensusValues(result['edgeOverflow'], [0]);
    if (diagnostics) {
      expectCensusValues(result['requiredCount'], [1]);
      expectCensusValues(result['touchingSegmentCount'], [0]);
    }
    if (includeSegments) {
      expectCensusValues(result['segmentRings'], [0, 0, 0, 0]);
      // Cell boundary edges are emitted bottom, right, top, left, directed CCW.
      // Dereference each owner: end-vertex indices have the same sorted set but must fail here.
      for (let edge = 0; edge < 4; edge++) {
        const vertex = result['segmentVertices'][edge];
        expectCensusValues(
          result['positions'].slice(vertex * 2, vertex * 2 + 2),
          square.slice(edge * 2, edge * 2 + 2)
        );
      }
      expectCensusValues(result['segmentFlags'], [0, 0, 0, 0]);
    }
    if (includePolygons) {
      expectCensusValues(rotateRing(result['polygonPositions'].slice(0, 8)), square);
      expectCensusValues(result['polygonRingOffsets'], [0, 4, 4]);
      expectCensusValues(result['polygonOffsets'], [0, 1, 1]);
      expectCensusValues(result['featureOffsets'], [0, 1, 1]);
      expectCensusValues(result['sourceIds'], [1, 0xffffffff]);
    }
  });
}

it('census touching squares publish directed starts, two owners and incoming touch flags', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCensusRasterHarness(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const result = await harness.run(
    new GPUIsobandRings({
      grid,
      edgeCapacity: 8,
      values: harness.input('float32', [1, 1, 1, 1, 1, 1, 1, 1, 1]),
      mask: harness.input('uint32', [1, 1, 0, 1, 1, 1, 0, 1, 1]),
      breaks: harness.input('float32', [0.5]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUIsobandsParameterValues(
            {breakCount: 1, firstBand: 1, lastBand: 1, extent: [0, 0, 3, 3]},
            grid
          )
        )
      ),
      output: {
        ringOffsets: harness.output('ringOffsets', 'uint32', 3),
        positions: harness.output('positions', 'float32x2', 10),
        segmentRings: harness.output('segmentRings', 'uint32', 8),
        segmentVertices: harness.output('segmentVertices', 'uint32', 8),
        segmentFlags: harness.output('segmentFlags', 'uint32', 8),
        status: {
          count: harness.output('count', 'uint32', 1),
          requiredCount: harness.output('requiredCount', 'uint32', 1),
          overflow: harness.output('overflow', 'uint32', 1),
          edgeOverflow: harness.output('edgeOverflow', 'uint32', 1),
          touchingSegmentCount: harness.output('touchingSegmentCount', 'uint32', 1)
        }
      }
    })
  );
  // Only lower-left and upper-right marching cells have four valid samples. Their unit squares
  // meet at (1.5,1.5); each contributes four CCW edges and five closed-ring vertices. Rings are
  // ordered by their first segment. Edges 1 and 7 enter the shared corner, which has two exits.
  const starts = [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 2.5];
  expectCensusValues(result['ringOffsets'], [0, 5, 10]);
  expectCensusValues(result['count'], [2]);
  expectCensusValues(result['requiredCount'], [2]);
  expectCensusValues(result['overflow'], [0]);
  expectCensusValues(result['edgeOverflow'], [0]);
  expectCensusValues(result['touchingSegmentCount'], [2]);
  expectCensusValues(result['segmentRings'], [0, 0, 0, 0, 1, 1, 1, 1]);
  expectCensusValues(result['segmentFlags'], [0, 1, 0, 0, 0, 0, 0, 1]);
  for (let edge = 0; edge < 8; edge++) {
    const vertex = result['segmentVertices'][edge];
    const ring = edge < 4 ? 0 : 1;
    expect(vertex).toBeGreaterThanOrEqual(result['ringOffsets'][ring]);
    expect(vertex).toBeLessThan(result['ringOffsets'][ring + 1] - 1);
    expectCensusValues(
      result['positions'].slice(vertex * 2, vertex * 2 + 2),
      starts.slice(edge * 2, edge * 2 + 2)
    );
  }
});
