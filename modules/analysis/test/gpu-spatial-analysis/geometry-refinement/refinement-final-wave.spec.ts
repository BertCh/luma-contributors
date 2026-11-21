// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUMaximumInscribedCircleOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle-operation';
import {CPUMaximumInscribedCircleOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {GPUNormalizePolygonsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons-operation';
import {GPUSegmentSnapOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';
import {GPUVectorStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton-operation';
import {
  GPULinePrecisionReduction,
  GPUPolygonPrecisionReduction
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {
  createRefinementOracleHarness,
  executeGPUOracleOperation
} from './refinement-output-oracle-utils';
import {createTopologyOutput} from './topology-output';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';

const names = {
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  polygonOffsets: 'polygonOffsets',
  featureOffsets: 'featureOffsets'
};
const circleOutput = {
  circles: 'circles',
  witnesses: 'witnesses',
  status: 'status',
  publishedCenterClearance: 'clearance',
  precisionLimited: 'precision',
  witnessOffsets: 'relative'
};

/** executeGPUOracleOperation keys rows by port; the shared assertions read graph value names. */
function getRowsByValue<Rows>(
  operation: {outputs: Readonly<Record<string, {value: string}>>},
  rows: Record<string, Rows>
): Record<string, Rows> {
  return Object.fromEntries(
    Object.entries(operation.outputs).map(([port, output]) => [output.value, rows[port]])
  );
}

/** Undirected XY segments, as sorted keys, so cyclic order and reversal do not matter. */
function getSegmentKeys(segments: readonly (readonly number[])[]): string[] {
  return segments
    .map(([x0, y0, x1, y1]) =>
      x0 < x1 || (x0 === x1 && y0 <= y1) ? `${x0},${y0},${x1},${y1}` : `${x1},${y1},${x0},${y0}`
    )
    .sort();
}

/** The four edges of an axis-aligned rectangle with the given corner coordinates. */
function getRectangleEdgeKeys(xs: readonly number[], ys: readonly number[]): string[] {
  const [left, right] = xs;
  const [bottom, top] = ys;
  return getSegmentKeys([
    [left, bottom, right, bottom],
    [right, bottom, right, top],
    [right, top, left, top],
    [left, top, left, bottom]
  ]);
}

it.each([
  0, 1048576
])('UGE-05 circle publication at origin %i matches independent clearance on both forms', async origin => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs: Record<string, OperationGraphInput> = {
    positions: {
      format: 'float32x2',
      values: Float32Array.of(
        origin,
        origin,
        origin + 0.125,
        origin,
        origin + 0.125,
        origin + 0.125,
        origin,
        origin + 0.125
      )
    },
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  };
  const props = {
    ...names,
    featureCount: 1,
    tolerance: 0.1,
    cellCapacity: 4,
    maximumIterations: 0,
    output: circleOutput
  };
  const assertRows = (rows: Record<string, ArrayLike<number>>, gpu: boolean) => {
    expect(rows['circles'].length).toBe(4);
    expect(rows['witnesses'].length).toBe(2);
    expect(Array.from(rows['circles']).slice(0, 3)).toEqual(
      origin ? [origin, origin, 0] : [0.0625, 0.0625, 0.0625]
    );
    // GPU root Lipschitz gap sqrt(2)/16; CPU attains the half-side envelope cap.
    expect(Number.isFinite(rows['circles'][3])).toBe(true);
    expect(rows['circles'][3]).toBeCloseTo(gpu ? 0.08838834764831845 : 0, 7);
    expect(Array.from(rows['witnesses'])).toEqual(origin ? [origin, origin] : [0, 0.0625]);
    expect(Array.from(rows['relative'])).toEqual(origin ? [0, 0] : [-0.0625, 0]);
    expect(Array.from(rows['clearance'])).toEqual([origin ? 0 : 0.0625]);
    expect(Array.from(rows['precision'])).toEqual([origin ? 1 : 0]);
    expect(Array.from(rows['status'])).toEqual([0]);
  };
  const raw = new GPUMaximumInscribedCircleOperation(props);
  await executeGPUOracleOperation(device, raw, inputs, rows =>
    assertRows(getRowsByValue(raw, rows), true)
  );
  for (const operation of [
    new GPUMaximumInscribedCircleOperation(props),
    new CPUMaximumInscribedCircleOperation(props)
  ]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [operation],
      outputs: Object.values(circleOutput)
    });
    assertRows(rows, operation instanceof GPUMaximumInscribedCircleOperation);
  }
});

it('UGE-05 exterior rounded circle center uses a representable boundary fallback', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const origin = 1048576;
  const inputs: Record<string, OperationGraphInput> = {
    positions: {
      format: 'float32x2',
      values: Float32Array.of(
        origin,
        origin,
        origin + 0.375,
        origin + 0.125,
        origin + 0.5,
        origin + 0.125,
        origin + 0.125,
        origin
      )
    },
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  };
  for (const Operation of [
    GPUMaximumInscribedCircleOperation,
    CPUMaximumInscribedCircleOperation
  ]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [
        new Operation({
          ...names,
          featureCount: 1,
          tolerance: 0.1,
          cellCapacity: 4,
          maximumIterations: 0,
          output: circleOutput
        })
      ],
      outputs: Object.values(circleOutput)
    });
    expect(rows['circles'].length).toBe(4);
    expect(Array.from(rows['circles']).slice(0, 3)).toEqual([origin, origin, 0]);
    expect(Array.from(rows['witnesses'])).toEqual([origin, origin]);
    expect(Array.from(rows['relative'])).toEqual([0, 0]);
    expect(Array.from(rows['clearance'])).toEqual([0]);
    expect(Array.from(rows['precision'])).toEqual([1]);
    expect(Number.isFinite(rows['circles'][3])).toBe(true);
  }
});

it('KI R1 long split rectangle right ridge has exactly source edges 1 and 3', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs: Record<string, OperationGraphInput> = {
    positions: {
      format: 'float32x2',
      values: Float32Array.of(0, 0, 200000, 0, 200002, 0, 200002, 2, 200000, 2, 0, 2)
    },
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 6)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  };
  const output = {
    segments: 'segments',
    sourceEdges: 'sourceEdges',
    offsetPositions: 'offsetPositions',
    offsetLinks: 'offsetLinks',
    status: 'status'
  };
  const skeleton = new GPUVectorStraightSkeletonOperation({
    ...names,
    polygonCount: 1,
    maximumVertices: 6,
    maximumEvents: 8,
    segmentCapacity: 16,
    offsetDistance: 0.5,
    output
  });
  await executeGPUOracleOperation(device, skeleton, inputs, portRows => {
    const rows = getRowsByValue(skeleton, portRows);
    const count = rows['status'][0];
    const row = Array.from({length: count}, (_, index) => index).find(
      index => rows['segments'].slice(index * 4, index * 4 + 4).join(',') === '200000,1,200001,1'
    );
    expect(row).toBeDefined();
    expect(rows['sourceEdges'].slice(row! * 2, row! * 2 + 2).sort()).toEqual([1, 3]);
  });
});

it.each([
  {rings: [0, 4]},
  {rings: [0, 4, 4]},
  {rings: [0, 0, 4]},
  {rings: [0, 0, 0]}
])('KI R4 polygon precision preserves empty ring inactivity ($rings)', async ({rings}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const empty = rings[rings.length - 1] === 0;
  const output = createTopologyOutput(view);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', empty ? 0 : 4, empty ? [] : [10, 10, 14, 10, 14, 12, 10, 12]),
        ringOffsets: view('uint32', rings.length, rings),
        polygonOffsets: view('uint32', 2, [0, rings.length - 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 1,
      lattice: view('float32', 4),
      intersectionCapacity: 16,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([empty ? 0 : 4]);
    expect(await read(output.nodedSegments.status.candidateOverflow!)).toEqual([0]);
    expect(await read(output.status.count)).toEqual([empty ? 0 : 1]);
    expect(await read(output.status.candidateOverflow!)).toEqual([0]);
    expect(await read(output.status.overflow)).toEqual([0]);
    if (!empty) {
      expect((await read(output.polygons.ringOffsets)).slice(0, 2)).toEqual([0, 4]);
      const points = (await read(output.polygons.positions)).slice(0, 8);
      expect(points.length).toBe(8);
      expect(points.every(Number.isFinite)).toBe(true);
      // Paired cyclic edges: duplicated corners, a diagonal or a bow-tie cannot match.
      const edges = Array.from({length: 4}, (_, edge) => [
        points[edge * 2],
        points[edge * 2 + 1],
        points[((edge + 1) % 4) * 2],
        points[((edge + 1) % 4) * 2 + 1]
      ]);
      expect(getSegmentKeys(edges)).toEqual(getRectangleEdgeKeys([10, 14], [10, 12]));
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each([
  {grid: 41, values: [20.5, 0, 61.5, 0], expected: [41, 0, 82, 0]},
  {grid: 41, values: [-61.5, 0, -20.5, 0], expected: [-41, 0, 0, 0]},
  {grid: 7, values: [58720228, 0, 58720228, 7], expected: [58720228, 0, 58720228, 7]},
  {grid: 2, values: [1, 0, 3, 0], expected: [2, 0, 4, 0]}
])('UGE-03 precision certifies original operands on grid $grid ($values)', async ({
  grid,
  values,
  expected
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = {
    positions: view('float32x2', 2),
    lineOffsets: view('uint32', 2),
    status: view('uint32', 1)
  };
  graph.add(
    new GPULinePrecisionReduction({
      positions: view('float32x2', 2, values),
      lineOffsets: view('uint32', 2, [0, 2]),
      gridSize: grid,
      mode: 'valid-output',
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.positions)).toEqual(expected);
    expect(await read(output.lineOffsets)).toEqual([0, 2]);
    expect(await read(output.status)).toEqual([0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each([
  0, 1
])('DGN-09 GPU operation lowering writes canonical zero-row outputs (%i parents)', async count => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs: Record<string, OperationGraphInput> = {
    positions: {format: 'float32x2', values: new Float32Array()},
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0)},
    featureOffsets: {format: 'uint32', values: new Uint32Array(count + 1)},
    lineOffsets: {format: 'uint32', values: new Uint32Array(count + 1)},
    referencePositions: {format: 'float32x2', values: new Float32Array()}
  };
  const circle = await runOperationGraph(device, {
    inputs,
    operations: [
      new GPUMaximumInscribedCircleOperation({
        ...names,
        featureCount: count,
        tolerance: 0.1,
        output: circleOutput
      })
    ],
    outputs: Object.values(circleOutput)
  });
  expect(Array.from(circle['status'])).toEqual(count ? [2] : []);
  for (const [name, length] of [
    ['circles', count * 4],
    ['witnesses', count * 2],
    ['clearance', count],
    ['relative', count * 2]
  ] as const) {
    expect(circle[name].length).toBe(length);
    expect(Array.from(circle[name]).every(Number.isNaN)).toBe(true);
  }
  expect(Array.from(circle['precision'])).toEqual(count ? [0] : []);
  const normalized = await runOperationGraph(device, {
    inputs,
    operations: [
      new GPUNormalizePolygonsOperation({
        ...names,
        rowCount: 0,
        ringCount: 0,
        polygonCount: 0,
        featureCount: count,
        output: {positions: 'normalized', ringOffsets: 'rings', polygonOffsets: 'polygons'}
      })
    ],
    outputs: ['normalized', 'rings', 'polygons']
  });
  expect(Array.from(normalized['normalized'])).toEqual([]);
  expect(Array.from(normalized['rings'])).toEqual([0]);
  expect(Array.from(normalized['polygons'])).toEqual([0]);
  const snapped = await runOperationGraph(device, {
    inputs,
    operations: [
      new GPUSegmentSnapOperation({
        positions: 'positions',
        lineOffsets: 'lineOffsets',
        referencePositions: 'referencePositions',
        rowCount: 0,
        referenceCount: 0,
        lineCount: count,
        maximumVerticesPerLine: 1,
        tolerance: 0,
        output: {
          positions: 'snapped',
          sourceEdges: 'edges',
          counts: 'counts',
          validity: 'validity',
          issueCode: 'issue',
          status: {candidateOverflow: 'loss'}
        }
      })
    ],
    outputs: ['snapped', 'edges', 'counts', 'validity', 'issue', 'loss']
  });
  expect(snapped['snapped'].length).toBe(count * 2);
  expect(snapped['edges'].length).toBe(count);
  expect(Array.from(snapped['counts'])).toEqual(count ? [0, 0, 0, 0] : []);
  expect(Array.from(snapped['validity'])).toEqual(count ? [1] : []);
  expect(Array.from(snapped['issue'])).toEqual(count ? [0] : []);
  expect(Array.from(snapped['loss'])).toEqual(count ? [0] : []);
});

it.each([
  {grid: 41, values: [20.5, 0, 61.5, 0, 61.5, 82, 20.5, 82], xs: [41, 82], ys: [0, 82]},
  {grid: 41, values: [-61.5, 0, -20.5, 0, -20.5, 82, -61.5, 82], xs: [-41, 0], ys: [0, 82]},
  {
    grid: 7,
    // 58720214 = 7 * 8388602 is not a float32 (ulp 4 above 2^25): the stored input and the
    // published grid point are both float32(58720214) = 58720216.
    values: [58720228, 0, 58720228, 14, 58720214, 14, 58720214, 0],
    xs: [Math.fround(58720214), 58720228],
    ys: [0, 14]
  },
  {grid: 2, values: [1, 0, 3, 0, 3, 4, 1, 4], xs: [2, 4], ys: [0, 4]}
])('UGE-03 polygon precision certifies original cells on grid $grid ($values)', async ({
  grid,
  values,
  xs,
  ys
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4, values),
        ringOffsets: view('uint32', 2, [0, 4]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: grid,
      lattice: view('float32', 4),
      intersectionCapacity: 16,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([4]);
    expect(await read(output.nodedSegments.status.candidateOverflow!)).toEqual([0]);
    expect(await read(output.status.count)).toEqual([1]);
    const endpoints = (await read(output.nodedSegments.endpoints)).slice(0, 16);
    expect(endpoints.length).toBe(16);
    expect(endpoints.every(Number.isFinite)).toBe(true);
    const segments = Array.from({length: 4}, (_, segment) =>
      endpoints.slice(segment * 4, segment * 4 + 4)
    );
    expect(getSegmentKeys(segments)).toEqual(getRectangleEdgeKeys(xs, ys));
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('UGE-05 translated diamond retains a relative witness when its absolute witness rounds to the center', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const center = 1048576.25;
  const inputs: Record<string, OperationGraphInput> = {
    positions: {
      format: 'float32x2',
      values: Float32Array.of(
        center,
        center - 0.125,
        center + 0.125,
        center,
        center,
        center + 0.125,
        center - 0.125,
        center
      )
    },
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  };
  const diamond = new GPUMaximumInscribedCircleOperation({
    ...names,
    featureCount: 1,
    tolerance: 0.2,
    cellCapacity: 4,
    maximumIterations: 0,
    output: circleOutput
  });
  await executeGPUOracleOperation(device, diamond, inputs, portRows => {
    const rows = getRowsByValue(diamond, portRows);
    // Distance to x+y=.125 in the local frame is .125/sqrt(2); witness displacement is (-1/16,-1/16).
    expect(rows['circles'].length).toBe(4);
    expect(rows['circles'].every(Number.isFinite)).toBe(true);
    expect(rows['circles'].slice(0, 2)).toEqual([center, center]);
    expect(rows['circles'][2]).toBeCloseTo(0.08838834764831845, 7);
    expect(rows['circles'][3]).toBeCloseTo(0.1767766952966369, 7);
    expect(rows['clearance'][0]).toBeCloseTo(0.08838834764831845, 7);
    expect(rows['relative']).toEqual([-0.0625, -0.0625]);
    expect(rows['witnesses']).toEqual([center, center]);
    expect(rows['precision']).toEqual([1]);
    expect(rows['status']).toEqual([0]);
  });
});

function getSingleRingInputs(coordinates: readonly number[]): Record<string, OperationGraphInput> {
  return {
    positions: {format: 'float32x2', values: Float32Array.from(coordinates)},
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, coordinates.length / 2)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  };
}

it('round 1 B1 GPU normalization ranks declared empty rings without positions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array()},
      ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 0)},
      polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3)},
      featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)}
    },
    operations: [
      new GPUNormalizePolygonsOperation({
        ...names,
        rowCount: 0,
        ringCount: 3,
        polygonCount: 2,
        featureCount: 1,
        output: {positions: 'normalized', ringOffsets: 'rings', polygonOffsets: 'polygons'}
      })
    ],
    outputs: ['normalized', 'rings', 'polygons']
  });
  expect(Array.from(rows['normalized'])).toEqual([]);
  expect(Array.from(rows['rings'])).toEqual([0, 0, 0, 0]);
  // Descending part order: the two-ring part first, then the one-ring part.
  expect(Array.from(rows['polygons'])).toEqual([0, 2, 3]);
});

it.each([
  {
    name: 'B2 split long rectangle',
    coordinates: [-10000000, -1, 3, -1, 10000000, -1, 10000000, 1, -3, 1, -10000000, 1],
    tolerance: 20000000,
    // Exact clearance of (0,0) is 1, witnessed by the foot (0,-1) on the bottom edge.
    expected: {center: [0, 0, 1], witness: [0, -1], relative: [0, -1], clearance: 1, limited: 0}
  },
  {
    name: 'B6 nonrepresentable oblique foot',
    coordinates: [-5, 0, 0, -1, 5, 0, 0, 1],
    tolerance: 10,
    // Foot of the origin on x - 5y + 5 = 0 is (-5/26, 25/26); clearance 5/sqrt(26).
    expected: {
      center: [0, 0, Math.fround(5 / Math.sqrt(26))],
      witness: Array.from(Float32Array.of(-5 / 26, 25 / 26)),
      relative: Array.from(Float32Array.of(-5 / 26, 25 / 26)),
      clearance: Math.fround(5 / Math.sqrt(26)),
      limited: 1
    }
  },
  {
    name: 'B6 representable oblique foot',
    coordinates: [-1, 0, 0, -1, 1, 0, 0, 1],
    tolerance: 10,
    // Foot of the origin on x - y + 1 = 0 is (-1/2, 1/2); clearance 1/sqrt(2).
    expected: {
      center: [0, 0, Math.fround(Math.SQRT1_2)],
      witness: [-0.5, 0.5],
      relative: [-0.5, 0.5],
      clearance: Math.fround(Math.SQRT1_2),
      limited: 0
    }
  }
])('round 1 $name: GPU circle publication', async ({coordinates, tolerance, expected}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = new GPUMaximumInscribedCircleOperation({
    ...names,
    featureCount: 1,
    tolerance,
    cellCapacity: 4,
    maximumIterations: 0,
    output: circleOutput
  });
  await executeGPUOracleOperation(device, operation, getSingleRingInputs(coordinates), portRows => {
    const rows = getRowsByValue(operation, portRows);
    expect(rows['circles'].length).toBe(4);
    expect(rows['circles'].every(Number.isFinite)).toBe(true);
    expect(rows['circles'].slice(0, 2)).toEqual(expected.center.slice(0, 2));
    expect(rows['circles'][2]).toBeCloseTo(expected.center[2], 6);
    expect(rows['witnesses']).toEqual(expected.witness);
    expect(rows['relative']).toEqual(expected.relative);
    expect(rows['clearance'][0]).toBeCloseTo(expected.clearance, 6);
    expect(rows['precision']).toEqual([expected.limited]);
  });
});

it('round 1 B3 translated short split rectangle ridge has source edges 1 and 3', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const skeleton = new GPUVectorStraightSkeletonOperation({
    ...names,
    polygonCount: 1,
    maximumVertices: 6,
    maximumEvents: 8,
    segmentCapacity: 16,
    offsetDistance: 0.5,
    output: {
      segments: 'segments',
      sourceEdges: 'sourceEdges',
      offsetPositions: 'offsetPositions',
      offsetLinks: 'offsetLinks',
      status: 'status'
    }
  });
  // Split at x = 2^23; the ridge midpoint 8388608.5 is not a float32 value.
  const inputs = getSingleRingInputs([
    8388600, 0, 8388608, 0, 8388610, 0, 8388610, 2, 8388608, 2, 8388600, 2
  ]);
  await executeGPUOracleOperation(device, skeleton, inputs, portRows => {
    const rows = getRowsByValue(skeleton, portRows);
    const row = Array.from({length: rows['status'][0]}, (_, index) => index).find(
      index => rows['segments'].slice(index * 4, index * 4 + 4).join(',') === '8388608,1,8388609,1'
    );
    expect(row).toBeDefined();
    expect(rows['sourceEdges'].slice(row! * 2, row! * 2 + 2).sort()).toEqual([1, 3]);
  });
});

it.each([
  // 78293712 = 7 * 11184816 is the right (upper) bound, then the left (lower) bound.
  {coordinates: [78293600, 0, 78293712, 0, 78293712, 14, 78293600, 14], xs: [78293600, 78293712]},
  {coordinates: [78293712, 0, 78293824, 0, 78293824, 14, 78293712, 14], xs: [78293712, 78293824]}
])('round 1 B4 polygon precision keeps exact grid bounds $xs on grid 7', async ({
  coordinates,
  xs
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4, coordinates),
        ringOffsets: view('uint32', 2, [0, 4]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 7,
      lattice: view('float32', 4),
      intersectionCapacity: 16,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([4]);
    expect(await read(output.nodedSegments.status.candidateOverflow!)).toEqual([0]);
    expect(await read(output.status.count)).toEqual([1]);
    const endpoints = (await read(output.nodedSegments.endpoints)).slice(0, 16);
    expect(endpoints.every(Number.isFinite)).toBe(true);
    const segments = Array.from({length: 4}, (_, segment) =>
      endpoints.slice(segment * 4, segment * 4 + 4)
    );
    expect(getSegmentKeys(segments)).toEqual(getRectangleEdgeKeys(xs, [0, 14]));
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('round 1 B5 line precision preserves an exact grid point beyond 2^24 cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = {
    positions: view('float32x2', 2),
    lineOffsets: view('uint32', 2),
    status: view('uint32', 1)
  };
  // 234880912 = 7 * 33554416 and 7 = 7 * 1: both already lie on the grid.
  graph.add(
    new GPULinePrecisionReduction({
      positions: view('float32x2', 2, [234880912, 0, 234880912, 7]),
      lineOffsets: view('uint32', 2, [0, 2]),
      gridSize: 7,
      mode: 'valid-output',
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.positions)).toEqual([234880912, 0, 234880912, 7]);
    expect(await read(output.lineOffsets)).toEqual([0, 2]);
    expect(await read(output.status)).toEqual([0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
