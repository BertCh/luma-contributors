// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUCoverageDissolve,
  GPUCoverageValidity,
  getGPUCoverageValidityParameterValues
} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {flattenPolygons, type OraclePolygons} from '../spatial-weights/spatial-weights-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, polygon-topology-03, -05 and -08. Expected values are hand
// derived from the drawings (rectangles and triangles); ring winding is checked with exact
// integer arithmetic over the float32 coordinates.

type TestDevice = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;
type Ring = [number, number][];

type DissolveRun = {
  count: number;
  overflow: number;
  /** `status.uncertainCount`: vertices outside the exact range of the snap keys. */
  uncertainCount: number;
  boundarySegmentCount: number;
  /** Closed rings as written (the last vertex repeats the first). */
  rings: Ring[];
  isHole: number[];
  groups: number[];
};

async function runDissolve(
  device: TestDevice,
  polygons: OraclePolygons,
  labels: number[],
  snapTolerance?: number
): Promise<DissolveRun> {
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const ringCapacity = layout.ringOffsets.length;
  const vertexCapacity = vertexCount + ringCapacity;
  const rig = new WeightsRig(device);
  const out = {
    ringOffsets: rig.output('uint32', ringCapacity + 1),
    positions: rig.output('float32x2', vertexCapacity),
    ringIsHole: rig.output('uint32', ringCapacity),
    ringGroups: rig.output('uint32', ringCapacity),
    count: rig.output('uint32', 1),
    overflow: rig.output('uint32', 1),
    uncertainCount: rig.output('uint32', 1),
    boundarySegmentCount: rig.output('uint32', 1)
  };
  rig.run(
    new GPUCoverageDissolve({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      featureOffsets: rig.input(layout.featureOffsets, 'uint32', layout.featureOffsets.length),
      labels: rig.input(Uint32Array.from(labels), 'uint32', labels.length),
      snapGridSize: snapTolerance,
      output: {
        ringOffsets: out.ringOffsets.view,
        positions: out.positions.view,
        ringIsHole: out.ringIsHole.view,
        ringGroups: out.ringGroups.view,
        status: {
          count: out.count.view,
          overflow: out.overflow.view,
          uncertainCount: out.uncertainCount.view,
          boundarySegmentCount: out.boundarySegmentCount.view
        }
      }
    })
  );
  const [count] = await readUint32(out.count.buffer, 1);
  const offsets = await readUint32(out.ringOffsets.buffer, ringCapacity + 1);
  const coordinates = await readFloat32(out.positions.buffer, 2 * vertexCapacity);
  const rings: Ring[] = [];
  for (let ring = 0; ring < Math.min(count, ringCapacity); ring++) {
    const vertices: Ring = [];
    for (let vertex = offsets[ring]; vertex < offsets[ring + 1]; vertex++) {
      vertices.push([coordinates[2 * vertex], coordinates[2 * vertex + 1]]);
    }
    rings.push(vertices);
  }
  const result: DissolveRun = {
    count,
    overflow: (await readUint32(out.overflow.buffer, 1))[0],
    uncertainCount: (await readUint32(out.uncertainCount.buffer, 1))[0],
    boundarySegmentCount: (await readUint32(out.boundarySegmentCount.buffer, 1))[0],
    rings,
    isHole: await readUint32(out.ringIsHole.buffer, ringCapacity),
    groups: await readUint32(out.ringGroups.buffer, ringCapacity)
  };
  rig.destroy();
  return result;
}

async function runValidity(
  device: TestDevice,
  polygons: OraclePolygons,
  gapWidth: number
): Promise<number[]> {
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const rig = new WeightsRig(device);
  const segmentFlags = rig.output('uint32', vertexCount);
  const parameters = new GPUParameterBuffer(device, {
    id: 'coverage-validity-float32-parameters',
    format: 'float32',
    length: 4,
    values: getGPUCoverageValidityParameterValues({gapWidth})
  });
  rig.graph.add(
    new GPUCoverageValidity({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      featureOffsets: rig.input(layout.featureOffsets, 'uint32', layout.featureOffsets.length),
      parameters: parameters.importToGraph(rig.graph),
      output: {segmentFlags: segmentFlags.view}
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const flags = await readUint32(segmentFlags.buffer, vertexCount);
  compiled.destroy?.();
  parameters.destroy();
  rig.destroy();
  return flags;
}

/** Exact doubled signed area of a closed or open ring of integer-valued coordinates. */
function getDoubledArea(ring: Ring): bigint {
  let doubled = 0n;
  for (let index = 0; index < ring.length; index++) {
    const [x0, y0] = ring[index];
    const [x1, y1] = ring[(index + 1) % ring.length];
    doubled += BigInt(x0) * BigInt(y1) - BigInt(x1) * BigInt(y0);
  }
  return doubled;
}

function getRectangle(x0: number, y0: number, x1: number, y1: number): Ring {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1]
  ];
}

it('GPUCoverageDissolve snap keys follow the extent, not the distance from zero (review 03)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two 100 m squares side by side at a projected northing of 3,000,000 m, 1 mm snap tolerance.
  // Absolute keys clamped both northings to one value, so vertices 100 m apart were one point.
  const northing = 3000000;
  const squares: OraclePolygons = [
    [getRectangle(0, northing, 100, northing + 100)],
    [getRectangle(100, northing, 200, northing + 100)]
  ];
  const merged = await runDissolve(device, squares, [7, 7], 0.001);
  expect(merged.overflow).toBe(0);
  expect(merged.uncertainCount).toBe(0);
  expect(merged.count).toBe(1);
  expect(merged.isHole[0]).toBe(0);
  expect(merged.groups[0]).toBe(7);
  // The 200 x 100 rectangle with the two nodes of the dissolved edge kept: six edges.
  expect(merged.boundarySegmentCount).toBe(6);
  expect(merged.rings[0].length).toBe(7);
  const shifted = merged.rings[0].map(([x, y]) => [x, y - northing] as [number, number]);
  expect(getDoubledArea(shifted)).toBe(40000n);
  const expectedVertices = ['0,0', '100,0', '200,0', '200,100', '100,100', '0,100'];
  expect([...new Set(shifted.map(([x, y]) => `${x},${y}`))].sort()).toEqual(
    expectedVertices.sort()
  );
  // Different labels keep both squares.
  const separate = await runDissolve(device, squares, [1, 2], 0.001);
  expect(separate.overflow).toBe(0);
  expect(separate.count).toBe(2);
  expect(separate.boundarySegmentCount).toBe(8);
  for (const ring of separate.rings) {
    expect(getDoubledArea(ring.map(([x, y]) => [x, y - northing] as [number, number]))).toBe(
      20000n
    );
  }
  device.destroy?.();
});

it('GPUCoverageDissolve counts vertices outside the exact snap range (review 03)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 2^24 snap steps of 1 mm are 16777.216 m. An extent of 16000 m is inside, 20000 m is not.
  const inside: OraclePolygons = [
    [getRectangle(0, 0, 8000, 100)],
    [getRectangle(8000, 0, 16000, 100)]
  ];
  const outside: OraclePolygons = [
    [getRectangle(0, 0, 10000, 100)],
    [getRectangle(10000, 0, 20000, 100)]
  ];
  const insideRun = await runDissolve(device, inside, [0, 0], 0.001);
  expect(insideRun.uncertainCount).toBe(0);
  expect(insideRun.overflow).toBe(0);
  // The two vertices at x = 20000 are 2e7 steps from the first vertex. They are reported in
  // `uncertainCount`; `overflow` stays for dropped rows (fix review fr-geometry-07).
  const outsideRun = await runDissolve(device, outside, [0, 0], 0.001);
  expect(outsideRun.uncertainCount).toBe(2);
  expect(outsideRun.overflow).toBe(0);
  // Without a snap tolerance keys are the float bits: no range.
  const exactRun = await runDissolve(device, outside, [0, 0]);
  expect(exactRun.uncertainCount).toBe(0);
  expect(exactRun.overflow).toBe(0);
  device.destroy?.();
});

// Clockwise slivers whose exact doubled area is -1 and whose float32 shoelace sum is 0. The sum
// is one cross product `a.x * b.y - a.y * b.x`; a compiler may fuse either product into the
// subtraction, which keeps that product exact. The report's sliver gives 0 unless the second
// product is fused; the other one gives 0 unless the first is fused. Unfused, both give 0.
const SLIVERS: {ring: Ring; neighbour: Ring}[] = [
  {
    ring: [
      [0, 0],
      [16777216, 16777215],
      [16777215, 16777214]
    ],
    neighbour: [
      [0, 0],
      [16777216, 16777215],
      [0, 16777215]
    ]
  },
  {
    ring: [
      [0, 0],
      [16773121, 16777216],
      [16769025, 16773119]
    ],
    neighbour: [
      [0, 0],
      [16773121, 16777216],
      [0, 16777216]
    ]
  }
];

it('GPUCoverageDissolve orients a clockwise sliver whose float32 area is zero (review 08)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {ring} of SLIVERS) {
    expect(getDoubledArea(ring)).toBe(-1n);
    const run = await runDissolve(device, [[ring]], [0]);
    expect(run.overflow).toBe(0);
    expect(run.count).toBe(1);
    // Boundary edges are written with the interior on the left: a counter-clockwise shell.
    expect(run.isHole[0]).toBe(0);
    expect(getDoubledArea(run.rings[0].slice(0, -1))).toBe(1n);
  }
  device.destroy?.();
});

it('GPUCoverageValidity accepts a clockwise sliver beside its neighbour (review 08)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The neighbour shares the sliver's long edge and lies on its other side (counter-clockwise).
  // A valid coverage: no segment is flagged. Read as a zero-area ring, the sliver had its
  // interior taken to be on the neighbour's side: `overlap`.
  for (const {ring, neighbour} of SLIVERS) {
    expect(getDoubledArea(neighbour) > 0n).toBe(true);
    const flags = await runValidity(device, [[ring], [neighbour]], 0);
    expect(Array.from(flags)).toEqual([0, 0, 0, 0, 0, 0]);
  }
  device.destroy?.();
});

it('GPUCoverageValidity gap rule flags segments staggered beside each other (review 05)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lower = getRectangle(0, -5, 10, 0);
  // Side by side over the whole of the shorter edge: both facing edges are gaps at width 0.125.
  const beside: OraclePolygons = [[lower], [getRectangle(2, 0.0625, 8, 5)]];
  const besideFlags = await runValidity(device, beside, 0.125);
  expect(Array.from(besideFlags)).toEqual([0, 0, 4, 0, 4, 0, 0, 0]);
  // Staggered: the edges (10,0)-(0,0) (row 2) and (5,0.0625)-(15,0.0625) (row 4) run side by
  // side over x in [5, 10] only. Each projects onto the other over 5 units, and the ends of the
  // two clipped projections, (5,0)-(10,0) and (5,0.0625)-(10,0.0625), are 0.0625 apart: a gap by
  // the GEOS rule. shapely.coverage_invalid_edges (Shapely 2.2.0, GEOS 3.14.1) returns exactly
  // these two edges at gap_width 0.125 and none at 0.03125; see
  // test/oracles/coverage-validity-gaps.py, scene `staggered`. The rule before the closing pass
  // asked for both ends of the shorter edge within the gap width of the longer and flagged nothing.
  const staggered: OraclePolygons = [[lower], [getRectangle(5, 0.0625, 15, 5)]];
  const staggeredFlags = await runValidity(device, staggered, 0.125);
  expect(Array.from(staggeredFlags)).toEqual([0, 0, 4, 0, 4, 0, 0, 0]);
  const narrowFlags = await runValidity(device, staggered, 0.03125);
  expect(Array.from(narrowFlags)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  device.destroy?.();
});
