// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {usePrecisionExecutorHooks} from './precision-executor-hooks';
import {
  SQUARE_HOLE_INPUT,
  SQUARE_HOLE_SEGMENTS,
  SQUARE_HOLE_SOURCE_EDGES
} from './vector-square-hole-fixture';
import {CPUNormalizePolygonsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation';
import {expect, it} from 'vitest';
import {GPUNormalizePolygonsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons-operation';
import {GPUSegmentSnapOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';
import {computeVectorStraightSkeletonOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {createTopologyOutput} from './topology-output';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {computeConvexStraightSkeletonOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation';
import {
  CPUMaximumInscribedCircleOperation,
  computeMaximumInscribedCircleOnCPU
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {computeRatioConcaveHullOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import {executeCPUOracleOperation} from './refinement-output-oracle-utils';
import {getUnequalPolygonOracleCase} from './refinement-output-oracles-fixtures';
import {CPUConvexStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation';
import {CPUVectorStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';
import {CPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import {GPUConvexStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton-operation';
import {GPUVectorStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton-operation';
import {GPUMaximumInscribedCircleOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle-operation';
import {GPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull-operation';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import type {GPUSpatialFrame} from '../../../src/analysis-core/contracts/spatial-context';

useCPUExecutorTestHooks();
usePrecisionExecutorHooks();

function executeInterpretedOperation(
  operation: GPUContributorOperation,
  inputs: Record<string, OperationGraphInput>,
  expectedBudget?: number
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const views: Record<string, GraphDataView> = {};
  for (const [port, name] of Object.entries(operation.inputs)) {
    const input = inputs[name];
    views[port] = createTransientView(
      graph,
      port,
      input.format,
      input.values.length / Number(input.format.match(/x(\d)$/)?.[1] ?? 1)
    );
    memory.write(views[port], input.values);
  }
  for (const [port, output] of Object.entries(operation.outputs))
    views[port] = createTransientView(graph, port, output.format, output.length);
  try {
    const contributor = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
    );
    if (expectedBudget !== undefined) {
      expect((contributor as {props?: {maximumIterations?: number}}).props?.maximumIterations).toBe(
        expectedBudget
      );
      if (operation instanceof GPURatioConcaveHullOperation)
        expect((contributor as {props?: {ratio?: number}}).props?.ratio).toBe(0.2);
      return {};
    }
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    return Object.fromEntries(
      Object.entries(operation.outputs).map(([port, output]) => [
        output.value,
        output.format.startsWith('float')
          ? memory.readFloat32(views[port])
          : memory.readUint32(views[port])
      ])
    );
  } finally {
    device.destroy();
  }
}

it.each([
  'convex',
  'vector',
  'circle',
  'hull'
] as const)('review 2 R5 %s operation retains its construction-time work budget', async kind => {
  const names = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets'
  };
  const skeletonProps = {
    ...names,
    polygonCount: 1,
    maximumVertices: 4,
    maximumIterations: 0,
    maximumEvents: 0,
    segmentCapacity: 8,
    offsetDistance: 0.5,
    output: {
      segments: 'segments',
      sourceEdges: 'sourceEdges',
      offsetPositions: 'offsetPositions',
      offsetLinks: 'offsetLinks',
      status: 'status'
    }
  };
  const circleProps = {
    ...names,
    featureCount: 1,
    tolerance: 0.1,
    cellCapacity: 4,
    maximumIterations: 0,
    output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
  };
  const hullProps = {
    positions: 'positions',
    pointCount: 9,
    ratio: 0.2,
    maximumIterations: 0,
    output: {
      triangles: 'triangles',
      triangleValidity: 'triangleValidity',
      boundaryEdges: 'boundaryEdges',
      edgeValidity: 'edgeValidity',
      status: {converged: 'converged'}
    }
  };
  const operations =
    kind === 'convex'
      ? [
          new CPUConvexStraightSkeletonOperation(skeletonProps),
          new GPUConvexStraightSkeletonOperation(skeletonProps)
        ]
      : kind === 'vector'
        ? [
            new CPUVectorStraightSkeletonOperation(skeletonProps),
            new GPUVectorStraightSkeletonOperation(skeletonProps)
          ]
        : kind === 'circle'
          ? [
              new CPUMaximumInscribedCircleOperation(circleProps),
              new GPUMaximumInscribedCircleOperation(circleProps)
            ]
          : [
              new CPURatioConcaveHullOperation(hullProps),
              new GPURatioConcaveHullOperation(hullProps)
            ];
  skeletonProps.maximumIterations = 64;
  skeletonProps.maximumEvents = 64;
  circleProps.maximumIterations = 64;
  hullProps.maximumIterations = 64;
  const inputs: Record<string, OperationGraphInput> = {
    positions: {
      format: 'float32x2',
      values:
        kind === 'hull'
          ? HULL_SITES
          : kind === 'circle'
            ? Float32Array.of(0, 0, 8, 0, 0, 6)
            : RECTANGLE_INPUT.positions
    },
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, kind === 'circle' ? 3 : 4)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  };
  for (const operation of operations) {
    // Delaunay's cooperative kernel cannot run in the interpreter. Inspect the lowered
    // budget here; the headless case checks the eleven-triangle result on the device.
    if (kind === 'hull' && 'createContributor' in operation) {
      executeInterpretedOperation(operation, inputs, 0);
      continue;
    }
    const result =
      'createContributor' in operation
        ? executeInterpretedOperation(operation, inputs)
        : await executeCPUOracleOperation(operation, inputs);
    if (kind === 'convex' || kind === 'vector')
      expect(Array.from(result['status'])).toEqual([0, 0, 4, 4]);
    else if (kind === 'circle') {
      expect(Array.from(result['status'])).toEqual([1]);
      expect(Array.from(result['witnesses'])).toEqual([0, 6]);
    } else {
      expect(Array.from(result['converged'])).toEqual([0]);
      expect(Array.from(result['triangleValidity']).reduce((sum, value) => sum + value, 0)).toBe(
        11
      );
    }
  }
});

it.each([
  1, 2, 3
])('review 2 R2 circle capacity %i retains the root-cell bound', async cellCapacity => {
  const input = {
    positions: Float32Array.of(0, 0, 8, 0, 0, 6),
    ringOffsets: Uint32Array.of(0, 3),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  };
  const options = {tolerance: 0.1, maximumIterations: 4, cellCapacity};
  const pure = computeMaximumInscribedCircleOnCPU(input, options.tolerance, options);
  const operation = new CPUMaximumInscribedCircleOperation({
    ...options,
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    featureCount: 1,
    output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
  });
  const rows = await executeCPUOracleOperation(operation, {
    positions: {format: 'float32x2', values: input.positions},
    ringOffsets: {format: 'uint32', values: input.ringOffsets},
    polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
    featureOffsets: {format: 'uint32', values: input.featureOffsets}
  });
  for (const result of [pure, rows]) {
    expect(Array.from(result['status'])).toEqual([1]);
    expect(Array.from(result['witnesses'])).toEqual([0, 6]);
    expect(Array.from(result['circles']).slice(0, 3)).toEqual([0, 6, 0]);
    // The root center (4,3) is on the hypotenuse, and the half-shorter-side cap gives upper bound 3.
    expect(result['circles'][3]).toBe(3);
  }
});

it('review 2 R1 split collinear edges retain their own terminal ridge intervals', () => {
  const input = {
    positions: Float32Array.of(0, 0, 3, 0, 6, 0, 6, 2, 0, 2),
    ringOffsets: Uint32Array.of(0, 5),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  };
  const options = {maximumVertices: 5, maximumEvents: 8, segmentCapacity: 16, offsetDistance: 0.5};
  const cpuResult = computeVectorStraightSkeletonOnCPU(input, options);
  const interpreted = executeInterpretedOperation(
    new GPUVectorStraightSkeletonOperation({
      ...options,
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      polygonCount: 1,
      output: {
        segments: 'segments',
        sourceEdges: 'sourceEdges',
        offsetPositions: 'offsetPositions',
        offsetLinks: 'offsetLinks',
        status: 'status'
      }
    }),
    {
      positions: {format: 'float32x2', values: input.positions},
      ringOffsets: {format: 'uint32', values: input.ringOffsets},
      polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
      featureOffsets: {format: 'uint32', values: input.featureOffsets}
    }
  );
  // At t=1 the bottom intervals [0,3] and [3,6] meet top edge 3 separately.
  for (const result of [cpuResult, interpreted]) {
    for (const [coordinates, sources] of [
      [
        [1, 1, 3, 1],
        [0, 3]
      ],
      [
        [3, 1, 5, 1],
        [1, 3]
      ]
    ]) {
      const row = Array.from({length: result['status'][0]}, (_, index) => index).find(
        index =>
          Array.from(result['segments'].slice(index * 4, index * 4 + 4)).join(',') ===
          coordinates.join(',')
      );
      expect(row).toBeDefined();
      expect(Array.from(result['sourceEdges'].slice(row! * 2, row! * 2 + 2)).sort()).toEqual(
        sources
      );
    }
  }
});

it('KI R1 long split rectangle uses position-unit interval slack', () => {
  const input = {
    positions: Float32Array.of(0, 0, 200000, 0, 200002, 0, 200002, 2, 200000, 2, 0, 2),
    ringOffsets: Uint32Array.of(0, 6),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  };
  const options = {maximumVertices: 6, maximumEvents: 8, segmentCapacity: 16, offsetDistance: 0.5};
  const cpuResult = computeVectorStraightSkeletonOnCPU(input, options);
  const interpreted = executeInterpretedOperation(
    new GPUVectorStraightSkeletonOperation({
      ...options,
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      polygonCount: 1,
      output: {
        segments: 'segments',
        sourceEdges: 'sourceEdges',
        offsetPositions: 'offsetPositions',
        offsetLinks: 'offsetLinks',
        status: 'status'
      }
    }),
    {
      positions: {format: 'float32x2', values: input.positions},
      ringOffsets: {format: 'uint32', values: input.ringOffsets},
      polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
      featureOffsets: {format: 'uint32', values: input.featureOffsets}
    }
  );
  // At t=1 the bottom intervals [0,3] and [3,6] meet top edge 3 separately.
  for (const result of [cpuResult, interpreted]) {
    for (const [coordinates, sources] of [
      [
        [1, 1, 200000, 1],
        [0, 4]
      ],
      [
        [200000, 1, 200001, 1],
        [1, 3]
      ]
    ]) {
      const row = Array.from({length: result['status'][0]}, (_, index) => index).find(
        index =>
          Array.from(result['segments'].slice(index * 4, index * 4 + 4)).join(',') ===
          coordinates.join(',')
      );
      expect(row).toBeDefined();
      expect(Array.from(result['sourceEdges'].slice(row! * 2, row! * 2 + 2)).sort()).toEqual(
        sources
      );
    }
  }
});

function createPrecisionCase(
  coordinates: number[],
  gridSize: number,
  rings = [0, coordinates.length / 2]
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(
    format: Format,
    length: number,
    values?: number[]
  ): GraphDataView<Format> => {
    const result = createTransientView(graph, `review-${serial++}`, format, length);
    const ArrayType = format.startsWith('float') ? Float32Array : Uint32Array;
    memory.write(
      result,
      values ? ArrayType.from(values) : new ArrayType((length * result.rowByteLength) / 4).fill(913)
    );
    return result;
  };
  const output = createTopologyOutput(view);
  const props = {
    id: 'review-precision',
    polygons: {
      kind: 'polygons' as const,
      positions: view('float32x2', coordinates.length / 2, coordinates),
      ringOffsets: view('uint32', rings.length, rings),
      polygonOffsets: view('uint32', 2, [0, rings.length - 1]),
      featureOffsets: view('uint32', 2, [0, 1])
    },
    gridSize,
    intersectionCapacity: 16,
    lattice: view('float32', 4),
    output
  };
  const contributor = new GPUPolygonPrecisionReduction(props);
  const execute = () => {
    const nodes = contributor.getCommandNodes(graph);
    const placement = nodes.findIndex(
      node => node.id === 'review-precision-polygonize-faces-propagate-duplicate-owners'
    );
    expect(placement).toBeGreaterThan(0);
    executeCommandNodesOnCPU(
      [
        ...nodes.slice(0, placement + 1),
        ...nodes.filter(
          node =>
            node.id === 'review-precision-source-edge-rows' ||
            node.id === 'review-precision-original-edge-rows' ||
            node.id === 'review-precision-restore-grid-1'
        )
      ],
      memory
    );
  };
  return {device, graph, memory, view, output, props, contributor, execute};
}

it('review 2 R3 precision refuses a retired final requiredCount even beside requiredRingCount', () => {
  const fixture = createPrecisionCase([0, 0, 4, 0, 4, 2, 0, 2], 1);
  try {
    const status = {...fixture.output.status, requiredCount: fixture.view('uint32', 1)};
    expect(
      () =>
        new GPUPolygonPrecisionReduction({...fixture.props, output: {...fixture.output, status}})
    ).toThrow('output.status.requiredCount');
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  false,
  true
])('review 2 R6 precision captures frame getters once (first origin present: %s)', firstPresent => {
  const fixture = createPrecisionCase([0, 0, 4, 0, 4, 2, 0, 2], 1);
  const origin: [number, number] = [3, 4];
  const areaOfUse: [number, number, number, number] = [0, 0, 10, 10];
  const unit = {name: 'meters', metersPerUnit: 1};
  const reads = {origin: 0, areaOfUse: 0, unit: 0};
  const frame = {
    id: 'review-frame',
    kind: 'crs',
    get origin() {
      reads.origin++;
      return firstPresent || reads.origin > 1 ? origin : undefined;
    },
    get areaOfUse() {
      reads.areaOfUse++;
      return areaOfUse;
    },
    get unit() {
      reads.unit++;
      return unit;
    }
  } as GPUSpatialFrame;
  try {
    const contributor = new GPUPolygonPrecisionReduction({
      ...fixture.props,
      spatialContext: {coordinateSpace: 'planar', metric: 'native', frame}
    });
    expect(reads).toEqual({origin: 1, areaOfUse: 1, unit: 1});
    origin[0] = 77;
    areaOfUse[0] = 88;
    unit.metersPerUnit = 99;
    expect(contributor.props.spatialContext?.frame?.origin).toEqual(
      firstPresent ? [3, 4] : undefined
    );
    expect(contributor.props.spatialContext?.frame?.areaOfUse).toEqual([0, 0, 10, 10]);
    expect(contributor.props.spatialContext?.frame?.unit).toEqual({
      name: 'meters',
      metersPerUnit: 1
    });
  } finally {
    fixture.device.destroy();
  }
});

it('review 2 R4 precision sizes the lattice from unperturbed decimal-grid bounds', () => {
  const fixture = createPrecisionCase(
    [-158750.609375, 0.1, 260679.796875, 0.1, 260679.796875, 0.2, -158750.609375, 0.2],
    0.1
  );
  try {
    fixture.execute();
    // Width / float32(0.1) is exactly 2^22; it requires no coarser lattice.
    expect(fixture.memory.readFloat32(fixture.props.lattice)[2]).toBe(1);
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([4]);
  } finally {
    fixture.device.destroy();
  }
});

it('R2 decimal precision preserves the side of the original half-grid boundary', () => {
  const fixture = createPrecisionCase([0.35, 0, 0.41, 0, 0.41, 0.61, 0.35, 0.61], 0.1);
  try {
    fixture.execute();
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([4]);
    const endpoints = fixture.memory
      .readFloat32(fixture.output.nodedSegments.endpoints)
      .slice(0, 16);
    const expected = [0.3, 0, 0.4, 0, 0.4, 0, 0.4, 0.6, 0.4, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0];
    expected.forEach((value, index) =>
      expect(Math.abs(endpoints[index] - value)).toBeLessThan(2e-7)
    );
  } finally {
    fixture.device.destroy();
  }
});

it('R1 terminal rectangle ridge has exactly the bottom and top source supports', () => {
  const result = computeVectorStraightSkeletonOnCPU(
    {
      positions: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2),
      ringOffsets: Uint32Array.of(0, 4),
      polygonOffsets: Uint32Array.of(0, 1),
      featureOffsets: Uint32Array.of(0, 1)
    },
    {maximumVertices: 4, maximumEvents: 8, segmentCapacity: 8, offsetDistance: 0.5}
  );
  // At t=1, y=0+t and y=2-t meet along (1,1)--(3,1): original supports 0 and 2.
  const ridge = Array.from({length: result.status[0]}, (_, row) => row).find(
    row => Array.from(result.segments.slice(row * 4, row * 4 + 4)).join(',') === '1,1,3,1'
  );
  expect(ridge).toBeDefined();
  expect(Array.from(result.sourceEdges.slice(ridge! * 2, ridge! * 2 + 2)).sort()).toEqual([0, 2]);
});

it('R3 clockwise corner arc keeps the original outgoing-edge rows', () => {
  const result = computeVectorStraightSkeletonOnCPU(
    {
      positions: Float32Array.of(0, 0, 0, 2, 4, 2, 4, 0),
      ringOffsets: Uint32Array.of(0, 4),
      polygonOffsets: Uint32Array.of(0, 1),
      featureOffsets: Uint32Array.of(0, 1)
    },
    {maximumVertices: 4, maximumEvents: 8, segmentCapacity: 8, offsetDistance: 0.5}
  );
  const corner = Array.from({length: result.status[0]}, (_, row) => row).find(
    row => Array.from(result.segments.slice(row * 4, row * 4 + 4)).join(',') === '4,0,3,1'
  );
  expect(corner).toBeDefined();
  // The caller's vertical right edge is row 2; its bottom edge is row 3.
  expect(Array.from(result.sourceEdges.slice(corner! * 2, corner! * 2 + 2)).sort()).toEqual([2, 3]);
});

const RECTANGLE_INPUT = {
  positions: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2),
  ringOffsets: Uint32Array.of(0, 4),
  polygonOffsets: Uint32Array.of(0, 1),
  featureOffsets: Uint32Array.of(0, 1)
};
it('R4 convex zero budget captures the offset before refusing arc work', () => {
  const options = {
    maximumVertices: 4,
    maximumIterations: 0,
    segmentCapacity: 8,
    offsetDistance: 0.5
  };
  const result = computeConvexStraightSkeletonOnCPU(RECTANGLE_INPUT, options);
  expect(Array.from(result.status)).toEqual([0, 0, 4, 4]);
  expect(Array.from(result.offsetPositions)).toEqual([0.5, 0.5, 3.5, 0.5, 3.5, 1.5, 0.5, 1.5]);
});
it('R4 vector zero budget captures the offset before refusing arc work', () => {
  const result = computeVectorStraightSkeletonOnCPU(RECTANGLE_INPUT, {
    maximumVertices: 4,
    maximumEvents: 0,
    segmentCapacity: 8,
    offsetDistance: 0.5
  });
  expect(Array.from(result.status)).toEqual([0, 0, 4, 4]);
  expect(Array.from(result.offsetLinks)).toEqual([3, 1, 0, 2, 1, 3, 2, 0]);
});
it('R4 circle zero budget retains the last-vertex seed and reports the unresolved root bound', async () => {
  const operation = new CPUMaximumInscribedCircleOperation({
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    featureCount: 1,
    tolerance: 0.1,
    maximumIterations: 0,
    cellCapacity: 4,
    output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
  });
  const result = await executeCPUOracleOperation(operation, {
    positions: {format: 'float32x2', values: Float32Array.of(0, 0, 8, 0, 0, 6)},
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 3)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  });
  // Bounds center (4,3) lies on the hypotenuse: radius 0 cannot improve last vertex (0,6).
  expect(Array.from(result['status'])).toEqual([1]);
  expect(Array.from(result['witnesses'])).toEqual([0, 6]);
  expect(Array.from(result['circles']).slice(0, 3)).toEqual([0, 6, 0]);
  expect(result['circles'][3]).toBe(3);
});
const HULL_SITES = Float32Array.of(
  11.75,
  5.25,
  4.5,
  8.5,
  14,
  10,
  8,
  5.75,
  12.5,
  1.75,
  12.5,
  3.25,
  6.75,
  13.25,
  9.25,
  2.25,
  10.75,
  5.75
);
it('R4 hull zero removals retains all eleven Delaunay triangles and reports incomplete erosion', () => {
  const options = {ratio: 0.2, maximumIterations: 0};
  const result = computeRatioConcaveHullOnCPU(HULL_SITES, options);
  // Euler: 9 sites, 5 convex-hull vertices => 2*9-2-5 = 11 triangles before erosion.
  expect(Array.from(result.triangleValidity).reduce((sum, value) => sum + value, 0)).toBe(11);
  expect(Array.from(result.converged)).toEqual([0]);
});

it('R6 precision keeps the originally validated destination when the caller replaces a nested view', () => {
  const fixture = createPrecisionCase([0, 0, 4, 0, 4, 2, 0, 2], 1);
  const original = fixture.output.nodedSegments.sourceEdgeIds;
  const replacement = fixture.view('uint32', 32);
  fixture.output.nodedSegments.sourceEdgeIds = replacement;
  try {
    fixture.execute();
    expect(Array.from(fixture.memory.readUint32(original)).slice(0, 4)).toEqual([0, 1, 2, 3]);
    expect(Array.from(fixture.memory.readUint32(replacement))).toEqual(Array(32).fill(913));
  } finally {
    fixture.device.destroy();
  }
});

it('R6 precision snapshots scalar props and input records while retaining graph views', () => {
  const fixture = createPrecisionCase([0, 0, 4, 0, 4, 2, 0, 2], 1);
  fixture.props.gridSize = 4;
  fixture.props.polygons.ringOffsets = fixture.view('uint32', 2, [0, 0]);
  try {
    fixture.execute();
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([4]);
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.sourceEdgeIds)).slice(0, 4)
    ).toEqual([0, 1, 2, 3]);
  } finally {
    fixture.device.destroy();
  }
});

it('R7 polygon oracle distinguishes copying input ring offsets', () => {
  const fixture = getUnequalPolygonOracleCase();
  expect(fixture.expected.ringOffsetsOutput).not.toEqual(
    Array.from(fixture.inputs.ringOffsets.values)
  );
});

it('collapsed-chain directed diagnostics account for recursive dangle peeling', () => {
  const fixture = createPrecisionCase([0, 0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0, 0, 0], 1);
  try {
    fixture.execute();
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([6]);
    expect(Array.from(fixture.memory.readUint32(fixture.output.directedEdges.flags))).toEqual([
      ...Array(12).fill(2),
      ...Array(52).fill(0)
    ]);
    expect(Array.from(fixture.memory.readUint32(fixture.output.directedEdges.ringIds))).toEqual(
      Array(64).fill(0xffffffff)
    );
  } finally {
    fixture.device.destroy();
  }
});

it('a bridge anchored by two closed squares survives recursive dangle peeling', () => {
  const fixture = createPrecisionCase(
    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 3, 0, 4, 0, 4, 1, 3, 1, 3, 0, 1, 0],
    1
  );
  try {
    fixture.execute();
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([10]);
    // The two source occurrences of the bridge are rows 4 and 9. Their four directed
    // sides belong to the exterior walk (ring 1), and neither is peeled as a dangle.
    const rings = fixture.memory.readUint32(fixture.output.directedEdges.ringIds);
    const flags = fixture.memory.readUint32(fixture.output.directedEdges.flags);
    expect([rings[8], rings[9], rings[18], rings[19]]).toEqual([1, 1, 1, 1]);
    expect(Array.from(flags.slice(0, 20), flag => flag & 2)).toEqual(Array(20).fill(0));
  } finally {
    fixture.device.destroy();
  }
});

it.each([
  'nodedSegments',
  'directedEdges'
] as const)('KI R3 refuses final ring demand on %s before any empty lowering', stage => {
  for (const coordinates of [[], [10, 10, 14, 10, 14, 12, 10, 12]]) {
    const fixture = createPrecisionCase(coordinates, 1);
    try {
      for (const inherited of [false, true]) {
        const status = {...fixture.output[stage].status};
        const demand = fixture.view('uint32', 1);
        if (inherited) Object.setPrototypeOf(status, {requiredRingCount: demand});
        else Object.defineProperty(status, 'requiredRingCount', {value: demand});
        expect(
          () =>
            new GPUPolygonPrecisionReduction({
              ...fixture.props,
              output: {...fixture.output, [stage]: {...fixture.output[stage], status}}
            })
        ).toThrow(`output.${stage}.status.requiredRingCount`);
      }
      expect(() => new GPUPolygonPrecisionReduction(fixture.props)).not.toThrow();
    } finally {
      fixture.device.destroy();
    }
  }
});

it.each([
  {rings: [0, 4]},
  {rings: [0, 4, 4]},
  {rings: [0, 0, 4]}
])('KI R4 inactive empty rings preserve lattice diagnostics ($rings)', ({rings}) => {
  const fixture = createPrecisionCase([10, 10, 14, 10, 14, 12, 10, 12], 1, rings);
  try {
    fixture.execute();
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.candidateOverflow!))
    ).toEqual([0]);
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([4]);
  } finally {
    fixture.device.destroy();
  }
});

it.each([0, 1])('DGN-09 GPU factories lower zero rows with %i declared empty parents', count => {
  const inputs: Record<string, OperationGraphInput> = {
    positions: {format: 'float32x2', values: new Float32Array()},
    ringOffsets: {format: 'uint32', values: Uint32Array.of(0)},
    polygonOffsets: {format: 'uint32', values: Uint32Array.of(0)},
    featureOffsets: {format: 'uint32', values: new Uint32Array(count + 1)},
    lineOffsets: {format: 'uint32', values: new Uint32Array(count + 1)},
    referencePositions: {format: 'float32x2', values: new Float32Array()}
  };
  const circle = executeInterpretedOperation(
    new GPUMaximumInscribedCircleOperation({
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      featureCount: count,
      tolerance: 0.1,
      output: {
        circles: 'circles',
        witnesses: 'witnesses',
        status: 'status',
        publishedCenterClearance: 'clearance',
        precisionLimited: 'precision',
        witnessOffsets: 'relative'
      }
    }),
    inputs
  );
  expect(Array.from(circle['status'])).toEqual(count ? [2] : []);
  for (const [name, length] of [
    ['circles', 4 * count],
    ['witnesses', 2 * count],
    ['clearance', count],
    ['relative', 2 * count]
  ] as const) {
    expect(circle[name].length).toBe(length);
    expect(Array.from(circle[name]).every(Number.isNaN)).toBe(true);
  }
  expect(Array.from(circle['precision'])).toEqual(count ? [0] : []);
  const normalized = executeInterpretedOperation(
    new GPUNormalizePolygonsOperation({
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      rowCount: 0,
      ringCount: 0,
      polygonCount: 0,
      featureCount: count,
      output: {
        positions: 'positionsOutput',
        ringOffsets: 'ringOffsetsOutput',
        polygonOffsets: 'polygonOffsetsOutput'
      }
    }),
    inputs
  );
  expect(Array.from(normalized['positionsOutput'])).toEqual([]);
  expect(Array.from(normalized['ringOffsetsOutput'])).toEqual([0]);
  expect(Array.from(normalized['polygonOffsetsOutput'])).toEqual([0]);
  const snapped = executeInterpretedOperation(
    new GPUSegmentSnapOperation({
      positions: 'positions',
      lineOffsets: 'lineOffsets',
      referencePositions: 'referencePositions',
      rowCount: 0,
      referenceCount: 0,
      lineCount: count,
      tolerance: 0,
      maximumVerticesPerLine: 1,
      output: {
        positions: 'snapPositions',
        sourceEdges: 'sourceEdges',
        counts: 'counts',
        validity: 'validity',
        issueCode: 'issueCode',
        status: {candidateOverflow: 'candidateOverflow'}
      }
    }),
    inputs
  );
  expect(snapped['snapPositions'].length).toBe(count * 2);
  expect(snapped['sourceEdges'].length).toBe(count);
  expect(Array.from(snapped['counts'])).toEqual(count ? [0, 0, 0, 0] : []);
  expect(Array.from(snapped['validity'])).toEqual(count ? [1] : []);
  expect(Array.from(snapped['issueCode'])).toEqual(count ? [0] : []);
  expect(Array.from(snapped['candidateOverflow'])).toEqual(count ? [0] : []);
});

it('UGE-10 validates terminal coverage even for single-offset compact levels', () => {
  const options = {segmentCapacity: 8, offsetDistance: 0};
  const empty = {
    positions: new Float32Array(),
    ringOffsets: Uint32Array.of(0),
    polygonOffsets: Uint32Array.of(0),
    featureOffsets: Uint32Array.of(0)
  };
  expect(computeVectorStraightSkeletonOnCPU(empty, options).status.length).toBe(0);
  for (const input of [
    {...empty, positions: Float32Array.of(0, 0)},
    {...empty, ringOffsets: Uint32Array.of(0, 0)},
    {...empty, polygonOffsets: Uint32Array.of(0, 0)},
    {...empty, positions: Float32Array.of(0, 0), ringOffsets: Uint32Array.of(1)}
  ])
    expect(() => computeVectorStraightSkeletonOnCPU(input, options)).toThrow();
});

it.each([
  'cpu',
  'gpu'
] as const)('KI R5 %s hull snapshots only its supported fields', async route => {
  let reads = 0;
  const props = {
    positions: 'positions',
    pointCount: 9,
    ratio: 0.2,
    get maximumIterations() {
      reads++;
      return reads === 1 ? 0 : 64;
    },
    output: {
      triangles: 'triangles',
      triangleValidity: 'triangleValidity',
      boundaryEdges: 'boundaryEdges',
      edgeValidity: 'edgeValidity',
      status: {converged: 'converged'}
    }
  };
  Object.setPrototypeOf(props, {
    get maximumEvents() {
      throw new Error('unrelated maximumEvents');
    }
  });
  Object.setPrototypeOf(props.output, {
    get circles() {
      throw new Error('unrelated circles');
    }
  });
  const operation =
    route === 'cpu'
      ? new CPURatioConcaveHullOperation(props)
      : new GPURatioConcaveHullOperation(props);
  expect(reads).toBe(1);
  props.ratio = 1;
  props.output.triangles = 'changed';
  props.output.status.converged = 'changed-convergence';
  const inputs = {positions: {format: 'float32x2' as const, values: HULL_SITES}};
  if (operation instanceof CPURatioConcaveHullOperation) {
    const result = await executeCPUOracleOperation(operation, inputs);
    expect(result['triangles']).toBeDefined();
    expect(Array.from(result['converged'])).toEqual([0]);
    expect(result['changed']).toBeUndefined();
    expect(result['changed-convergence']).toBeUndefined();
  } else {
    executeInterpretedOperation(operation, inputs, 0);
    expect(operation.outputs['triangles'].value).toBe('triangles');
    expect(operation.outputs['converged'].value).toBe('converged');
  }
  expect(reads).toBe(1);
});

it('UGE-05 full GPU circle search retains local clearance before float32 publication', () => {
  for (const origin of [0, 1048576]) {
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
    const result = executeInterpretedOperation(
      new GPUMaximumInscribedCircleOperation({
        positions: 'positions',
        ringOffsets: 'ringOffsets',
        polygonOffsets: 'polygonOffsets',
        featureOffsets: 'featureOffsets',
        featureCount: 1,
        tolerance: 0.1,
        cellCapacity: 4,
        maximumIterations: 0,
        output: {
          circles: 'circles',
          witnesses: 'witnesses',
          status: 'status',
          publishedCenterClearance: 'clearance',
          precisionLimited: 'precision',
          witnessOffsets: 'relative'
        }
      }),
      inputs
    );
    expect(Array.from(result['circles']).slice(0, 3)).toEqual(
      origin ? [origin, origin, 0] : [0.0625, 0.0625, 0.0625]
    );
    // Root half-size .0625, with Lipschitz gap sqrt(2)*.0625 independent of publication.
    expect(result['circles'][3]).toBeCloseTo(0.08838834764831845, 7);
    expect(Array.from(result['precision'])).toEqual([origin ? 1 : 0]);
    expect(Array.from(result['clearance'])).toEqual([origin ? 0 : 0.0625]);
    expect(Array.from(result['relative'])).toEqual(origin ? [0, 0] : [-0.0625, 0]);
    expect(Array.from(result['witnesses'])).toEqual(origin ? [origin, origin] : [0, 0.0625]);
    expect(Array.from(result['status'])).toEqual([0]);
  }
});

it('KI R4 all-empty polygon precision has zero noding count and clear loss', () => {
  const fixture = createPrecisionCase([], 1, [0, 0, 0]);
  try {
    fixture.execute();
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.count))
    ).toEqual([0]);
    expect(
      Array.from(fixture.memory.readUint32(fixture.output.nodedSegments.status.candidateOverflow!))
    ).toEqual([0]);
  } finally {
    fixture.device.destroy();
  }
});

it('KI R1 retains the square-hole provenance control on CPU and interpreted WGSL', () => {
  const options = {
    maximumVertices: 8,
    maximumEvents: 16,
    segmentCapacity: 32,
    offsetDistance: 0.25
  };
  const cpu = computeVectorStraightSkeletonOnCPU(SQUARE_HOLE_INPUT, options);
  const gpu = executeInterpretedOperation(
    new GPUVectorStraightSkeletonOperation({
      ...options,
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      polygonCount: 1,
      output: {
        segments: 'segments',
        sourceEdges: 'sourceEdges',
        offsetPositions: 'offsetPositions',
        offsetLinks: 'offsetLinks',
        status: 'status'
      }
    }),
    {
      positions: {format: 'float32x2', values: SQUARE_HOLE_INPUT.positions},
      ringOffsets: {format: 'uint32', values: SQUARE_HOLE_INPUT.ringOffsets},
      polygonOffsets: {format: 'uint32', values: SQUARE_HOLE_INPUT.polygonOffsets},
      featureOffsets: {format: 'uint32', values: SQUARE_HOLE_INPUT.featureOffsets}
    }
  );
  for (const result of [cpu, gpu]) {
    expect(Array.from(result['status'])).toEqual([12, 12, 8, 0]);
    expect(Array.from(result['segments'].slice(0, 48))).toEqual(SQUARE_HOLE_SEGMENTS);
    expect(Array.from(result['sourceEdges'].slice(0, 24))).toEqual(SQUARE_HOLE_SOURCE_EDGES);
  }
});

it('UGE-05 interpreted publication diagnoses a coincident absolute diamond witness', () => {
  const center = 1048576.25;
  const result = executeInterpretedOperation(
    new GPUMaximumInscribedCircleOperation({
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      featureCount: 1,
      tolerance: 0.2,
      cellCapacity: 4,
      maximumIterations: 0,
      output: {
        circles: 'circles',
        witnesses: 'witnesses',
        status: 'status',
        publishedCenterClearance: 'clearance',
        precisionLimited: 'precision',
        witnessOffsets: 'relative'
      }
    }),
    {
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
    }
  );
  expect(result['circles'].length).toBe(4);
  expect(Array.from(result['circles']).every(Number.isFinite)).toBe(true);
  expect(Array.from(result['circles']).slice(0, 2)).toEqual([center, center]);
  expect(result['circles'][2]).toBeCloseTo(0.08838834764831845, 7);
  expect(result['circles'][3]).toBeCloseTo(0.1767766952966369, 7);
  expect(Array.from(result['relative'])).toEqual([-0.0625, -0.0625]);
  expect(Array.from(result['witnesses'])).toEqual([center, center]);
  expect(Array.from(result['precision'])).toEqual([1]);
});

it('round 1 B1 normalization preserves and ranks declared empty rings on both forms', async () => {
  const input = {
    positions: new Float32Array(),
    ringOffsets: Uint32Array.of(0, 0, 0, 0),
    polygonOffsets: Uint32Array.of(0, 1, 3),
    featureOffsets: Uint32Array.of(0, 2)
  };
  const props = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    rowCount: 0,
    ringCount: 3,
    polygonCount: 2,
    featureCount: 1,
    output: {positions: 'normalized', ringOffsets: 'rings', polygonOffsets: 'polygons'}
  };
  const inputs: Record<string, OperationGraphInput> = {
    positions: {format: 'float32x2', values: input.positions},
    ringOffsets: {format: 'uint32', values: input.ringOffsets},
    polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
    featureOffsets: {format: 'uint32', values: input.featureOffsets}
  };
  for (const rows of [
    executeInterpretedOperation(new GPUNormalizePolygonsOperation(props), inputs),
    await executeCPUOracleOperation(new CPUNormalizePolygonsOperation(props), inputs)
  ]) {
    expect(Array.from(rows['normalized'])).toEqual([]);
    expect(Array.from(rows['rings'])).toEqual([0, 0, 0, 0]);
    // Descending canonical part order: two empty rings first, then one empty ring.
    expect(Array.from(rows['polygons'])).toEqual([0, 2, 3]);
  }
});

function executeRoundOneCircle(coordinates: number[], tolerance = 2) {
  return executeInterpretedOperation(
    new GPUMaximumInscribedCircleOperation({
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      featureCount: 1,
      tolerance,
      cellCapacity: 4,
      maximumIterations: 0,
      output: {
        circles: 'circles',
        witnesses: 'witnesses',
        status: 'status',
        publishedCenterClearance: 'clearance',
        precisionLimited: 'precision',
        witnessOffsets: 'relative'
      }
    }),
    {
      positions: {format: 'float32x2', values: Float32Array.from(coordinates)},
      ringOffsets: {format: 'uint32', values: Uint32Array.of(0, coordinates.length / 2)},
      polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
      featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
    }
  );
}

it('round 1 B2 split long rectangle clearance and witness use a stable projection', () => {
  const result = executeRoundOneCircle(
    [-10000000, -1, 3, -1, 10000000, -1, 10000000, 1, -3, 1, -10000000, 1],
    20000000
  );
  expect(Array.from(result['circles']).slice(0, 3)).toEqual([0, 0, 1]);
  expect(Array.from(result['witnesses'])).toEqual([0, -1]);
  expect(Array.from(result['relative'])).toEqual([0, -1]);
  expect(Array.from(result['clearance'])).toEqual([1]);
  expect(Array.from(result['precision'])).toEqual([0]);
});

it('round 1 B3 translated ridge tests its midpoint in endpoint-relative coordinates', () => {
  const positions = Float32Array.of(
    8388600,
    0,
    8388608,
    0,
    8388610,
    0,
    8388610,
    2,
    8388608,
    2,
    8388600,
    2
  );
  const result = executeInterpretedOperation(
    new GPUVectorStraightSkeletonOperation({
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      polygonCount: 1,
      maximumVertices: 6,
      maximumEvents: 8,
      segmentCapacity: 16,
      offsetDistance: 0.5,
      output: {
        segments: 'segments',
        sourceEdges: 'sources',
        offsetPositions: 'offsets',
        offsetLinks: 'links',
        status: 'status'
      }
    }),
    {
      positions: {format: 'float32x2', values: positions},
      ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 6)},
      polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
      featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
    }
  );
  const row = Array.from({length: result['status'][0]}, (_, index) => index).find(
    index =>
      Array.from(result['segments'].slice(index * 4, index * 4 + 4)).join(',') ===
      '8388608,1,8388609,1'
  );
  expect(row).toBeDefined();
  expect(Array.from(result['sources'].slice(row! * 2, row! * 2 + 2)).sort()).toEqual([1, 3]);
});

it.each([
  {coordinates: [-5, 0, 0, -1, 5, 0, 0, 1], limited: 1},
  {coordinates: [-1, 0, 0, -1, 1, 0, 0, 1], limited: 0}
])('round 1 B6 witness representability is certified ($coordinates)', ({coordinates, limited}) => {
  const result = executeRoundOneCircle(coordinates, 10);
  expect(result['circles'].length).toBe(4);
  expect(Array.from(result['circles']).every(Number.isFinite)).toBe(true);
  expect(Array.from(result['precision'])).toEqual([limited]);
  // Foot of the origin on x - 5y + 5 = 0 is (-5/26, 25/26), not a float32 pair; on x - y + 1 = 0
  // it is (-1/2, 1/2), exactly representable. The CPU twin reports the same flags.
  if (limited) {
    expect(Array.from(result['witnesses'])).toEqual(Array.from(Float32Array.of(-5 / 26, 25 / 26)));
  } else {
    expect(Array.from(result['witnesses'])).toEqual([-0.5, 0.5]);
  }
  const cpu = computeMaximumInscribedCircleOnCPU(
    {
      positions: Float32Array.from(coordinates),
      ringOffsets: Uint32Array.of(0, 4),
      polygonOffsets: Uint32Array.of(0, 1),
      featureOffsets: Uint32Array.of(0, 1)
    },
    10,
    {maximumIterations: 0, cellCapacity: 4}
  );
  expect(Array.from(cpu.circles.slice(0, 2))).toEqual([0, 0]);
  expect(Array.from(cpu.precisionLimited)).toEqual([limited]);
});

it.each([
  'cpu',
  'gpu'
] as const)('round 1 B8 %s vector factory never reads undeclared spatialContext', route => {
  const props = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    polygonCount: 1,
    maximumVertices: 4,
    maximumEvents: 0,
    segmentCapacity: 8,
    offsetDistance: 0,
    output: {
      segments: 'segments',
      sourceEdges: 'sources',
      offsetPositions: 'offsets',
      offsetLinks: 'links',
      status: 'status'
    }
  };
  Object.setPrototypeOf(props, {
    get spatialContext() {
      throw new Error('undeclared spatialContext');
    }
  });
  expect(() =>
    route === 'cpu'
      ? new CPUVectorStraightSkeletonOperation(props)
      : new GPUVectorStraightSkeletonOperation(props)
  ).not.toThrow();
});

it('round 1 S1 GPU snap validates capacity at construction, even when zero lines allocate zero outputs', () => {
  const props = {
    positions: 'positions',
    lineOffsets: 'lines',
    referencePositions: 'references',
    rowCount: 0,
    referenceCount: 0,
    lineCount: 1,
    maximumVerticesPerLine: 0,
    tolerance: 0,
    output: {positions: 'out', sourceEdges: 'sources', counts: 'counts'}
  };
  expect(() => new GPUSegmentSnapOperation(props)).toThrow('maximumVerticesPerLine');
  expect(() => new GPUSegmentSnapOperation({...props, lineCount: 0})).toThrow(
    'maximumVerticesPerLine'
  );
  expect(
    () => new GPUSegmentSnapOperation({...props, lineCount: 0, maximumVerticesPerLine: 1})
  ).not.toThrow();
});

it.each([
  'nodedSegments',
  'directedEdges'
] as const)('round 1 S2 %s captures getter-only proxy status before rejecting ring demand', stage => {
  const fixture = createPrecisionCase([0, 0, 4, 0, 4, 2, 0, 2], 1);
  const status = new Proxy(fixture.output[stage].status, {
    get(target, property, receiver) {
      return property === 'requiredRingCount'
        ? fixture.view('uint32', 1)
        : Reflect.get(target, property, receiver);
    },
    has(target, property) {
      return property === 'requiredRingCount' ? false : Reflect.has(target, property);
    }
  });
  try {
    expect(
      () =>
        new GPUPolygonPrecisionReduction({
          ...fixture.props,
          output: {...fixture.output, [stage]: {...fixture.output[stage], status}}
        })
    ).toThrow(`output.${stage}.status.requiredRingCount`);
  } finally {
    fixture.device.destroy();
  }
});
