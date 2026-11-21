// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getVectorOracleCase,
  expectOracleRows,
  getLoopOracleCase
} from './refinement-output-oracles-fixtures';
import {
  executeGPUOracleOperation,
  createRefinementOracleHarness
} from './refinement-output-oracle-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {createTopologyOutput} from './topology-output';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUConvexStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton-operation';
import {CPUConvexStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation';
import {GPUVectorStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton-operation';
import {CPUVectorStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';
import {GPUMaximumInscribedCircleOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle-operation';
import {CPUMaximumInscribedCircleOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {GPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull-operation';
import {CPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import type {GraphDataView, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';

it('review 2 R4 precision preserves the requested decimal step at the lattice extent limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  const lattice = view('float32', 4);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view(
          'float32x2',
          4,
          [-158750.609375, 0.1, 260679.796875, 0.1, 260679.796875, 0.2, -158750.609375, 0.2]
        ),
        ringOffsets: view('uint32', 2, [0, 4]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 0.1,
      intersectionCapacity: 16,
      lattice,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    // Shapely on stored float32 coordinates and step preserves four sides and one face.
    expect((await read(lattice))[2]).toBeCloseTo(0.1, 7);
    expect(await read(output.nodedSegments.status.count)).toEqual([4]);
    expect(await read(output.status.count)).toEqual([1]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each([
  1, 2, 3
])('review 2 R2 circle capacity %i publishes the unresolved root on both forms', async cellCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    featureCount: 1,
    tolerance: 0.1,
    maximumIterations: 4,
    cellCapacity,
    output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 8, 0, 0, 6)},
    ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 3)},
    polygonOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)}
  };
  const assertRows = (rows: Record<string, ArrayLike<number>>, error: number) => {
    expect(Array.from(rows['status'])).toEqual([1]);
    expect(Array.from(rows['witnesses'])).toEqual([0, 6]);
    expect(Array.from(rows['circles']).slice(0, 3)).toEqual([0, 6, 0]);
    expect(rows['circles'][3]).toBeCloseTo(error, 5);
  };
  for (const operation of [
    new CPUMaximumInscribedCircleOperation(props),
    new GPUMaximumInscribedCircleOperation(props)
  ]) {
    const rows = await runOperationGraph(device, {
      operations: [operation],
      inputs,
      outputs: Object.values(props.output)
    });
    assertRows(rows, operation instanceof CPUMaximumInscribedCircleOperation ? 3 : 4 * Math.SQRT2);
  }
});

it('review 2 R1 split collinear supports belong to separate terminal ridge intervals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = new GPUVectorStraightSkeletonOperation({
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    polygonCount: 1,
    maximumVertices: 5,
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
  await executeGPUOracleOperation(
    device,
    operation,
    {
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, 3, 0, 6, 0, 6, 2, 0, 2)},
      ringOffsets: {format: 'uint32', values: Uint32Array.of(0, 5)},
      polygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
      featureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
    },
    rows => {
      expect(rows['status'][3]).toBe(0);
      // Each bottom interval meets top edge 3 at y=1; a shared infinite line is insufficient.
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
        const row = Array.from({length: rows['status'][0]}, (_, index) => index).find(index =>
          rows['segments']
            .slice(index * 4, index * 4 + 4)
            .every((value, axis) => Math.abs(value - coordinates[axis]) < 3e-5)
        );
        expect(row).toBeDefined();
        expect(rows['sourceEdges'].slice(row! * 2, row! * 2 + 2).sort()).toEqual(sources);
      }
    }
  );
});

it('R1 rectangle ridge uses the original bottom and top supports', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = getVectorOracleCase();
  const result = await executeGPUOracleOperation(device, fixture.gpu(), fixture.inputs, rows =>
    expectOracleRows(rows, fixture.expected)
  );
  const ridge = Array.from({length: result['status'][0]}, (_, row) => row).find(
    row => result['segments'].slice(row * 4, row * 4 + 4).join(',') === '1,1,3,1'
  );
  expect(ridge).toBeDefined();
  expect(result['sourceEdges'].slice(ridge! * 2, ridge! * 2 + 2).sort()).toEqual([0, 2]);
});

it('R5 loop resubmission replaces a rational crossing with the independently scaled crossing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const fixture = getLoopOracleCase();
  const operation = fixture.gpu();
  const views: Record<string, GraphDataView> = {};
  for (const [name, input] of Object.entries(fixture.inputs))
    views[name] = harness.view(
      input.format,
      input.values.length / (input.format === 'float32x2' ? 2 : 1),
      Array.from(input.values)
    );
  for (const [name, output] of Object.entries(operation.outputs))
    views[name] = harness.view(output.format, output.length);
  harness.graph.add(
    operation.createContributor(
      <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
    )
  );
  const compiled = harness.graph.compile();
  try {
    for (const scale of [1, 3]) {
      harness.write(
        views['inputPositions'],
        Array.from(fixture.inputs.inputPositions.values, value => scale * value)
      );
      for (const name of Object.keys(operation.outputs)) harness.poison(views[name]);
      submitGraph(device, compiled);
      const result: Record<string, number[]> = {};
      for (const name of Object.keys(operation.outputs))
        result[name] = await harness.read(views[name]);
      // Multiplying both intersecting lines by three moves (1/3,1/3) to (1,1), preserving rows.
      expectOracleRows(result, {
        ...fixture.expected,
        positions: scale === 1 ? [0, 0, 1 / 3, 1 / 3, 1, -1] : [0, 0, 1, 1, 3, -3]
      });
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('R6 precision writes the original destination after the caller replaces its nested output member', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  const original = output.nodedSegments.sourceEdgeIds;
  const replacement = view('uint32', 32);
  const contributor = new GPUPolygonPrecisionReduction({
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 4, [0, 0, 4, 0, 4, 2, 0, 2]),
      ringOffsets: view('uint32', 2, [0, 4]),
      polygonOffsets: view('uint32', 2, [0, 1]),
      featureOffsets: view('uint32', 2, [0, 1])
    },
    gridSize: 1,
    intersectionCapacity: 16,
    lattice: view('float32', 4),
    output
  });
  output.nodedSegments.sourceEdgeIds = replacement;
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect((await read(original)).slice(0, 4)).toEqual([0, 1, 2, 3]);
    expect(await read(replacement)).toEqual(Array(32).fill(913));
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('R2 decimal precision keeps four sides and the polygon preserved by Shapely', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRefinementOracleHarness(device);
  const {graph, view, read} = harness;
  const output = createTopologyOutput(view);
  graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4, [0.35, 0, 0.41, 0, 0.41, 0.61, 0.35, 0.61]),
        ringOffsets: view('uint32', 2, [0, 4]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      gridSize: 0.1,
      intersectionCapacity: 16,
      lattice: view('float32', 4),
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await read(output.nodedSegments.status.count)).toEqual([4]);
    expect(await read(output.status.count)).toEqual([1]);
    const endpoints = (await read(output.nodedSegments.endpoints)).slice(0, 16);
    const expected = [0.3, 0, 0.4, 0, 0.4, 0, 0.4, 0.6, 0.4, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0];
    // Two f32 output roundings at this scale are below 2e-7. NaN must not pass.
    expected.forEach((value, index) => {
      expect(Number.isFinite(endpoints[index])).toBe(true);
      expect(Math.abs(endpoints[index] - value)).toBeLessThan(2e-7);
    });
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('R3 clockwise corner arc reports caller edges two and three', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = getVectorOracleCase();
  fixture.inputs.positions.values = Float32Array.of(0, 0, 0, 2, 4, 2, 4, 0);
  const result = await executeGPUOracleOperation(device, fixture.gpu(), fixture.inputs, rows => {
    const corner = Array.from({length: rows['status'][0]}, (_, row) => row).find(
      row => rows['segments'].slice(row * 4, row * 4 + 4).join(',') === '4,0,3,1'
    );
    expect(corner).toBeDefined();
    expect(rows['sourceEdges'].slice(corner! * 2, corner! * 2 + 2).sort()).toEqual([2, 3]);
  });
  const corner = Array.from({length: result['status'][0]}, (_, row) => row).find(
    row => result['segments'].slice(row * 4, row * 4 + 4).join(',') === '4,0,3,1'
  );
  expect(corner).toBeDefined();
  expect(result['sourceEdges'].slice(corner! * 2, corner! * 2 + 2).sort()).toEqual([2, 3]);
});

for (const kind of ['convex', 'vector', 'circle', 'hull'] as const)
  it(`R4 ${kind} CPU and GPU operations honor explicit zero work`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const names = {
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets'
    };
    const skeletonOutput = {
      segments: 'segments',
      sourceEdges: 'sourceEdges',
      offsetPositions: 'offsetPositions',
      status: 'status'
    };
    const rectangle = {
      positions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2)},
      ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 4)},
      polygonOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
      featureOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)}
    };
    if (kind === 'convex' || kind === 'vector') {
      const props = {
        ...names,
        polygonCount: 1,
        maximumVertices: 4,
        maximumIterations: 0,
        maximumEvents: 0,
        segmentCapacity: 8,
        offsetDistance: 0.5,
        output: {...skeletonOutput, offsetLinks: 'offsetLinks'}
      };
      const operations =
        kind === 'convex'
          ? [
              new GPUConvexStraightSkeletonOperation(props),
              new CPUConvexStraightSkeletonOperation(props)
            ]
          : [
              new GPUVectorStraightSkeletonOperation(props),
              new CPUVectorStraightSkeletonOperation(props)
            ];
      props.maximumIterations = 64;
      props.maximumEvents = 64;
      for (const operation of operations) {
        const outputs =
          kind === 'convex' ? Object.values(skeletonOutput) : Object.values(props.output);
        const result = await runOperationGraph(device, {
          inputs: rectangle,
          operations: [operation],
          outputs
        });
        expect(Array.from(result['status'])).toEqual([0, 0, 4, 4]);
        expect(Array.from(result['offsetPositions'])).toEqual([
          0.5, 0.5, 3.5, 0.5, 3.5, 1.5, 0.5, 1.5
        ]);
        if (kind === 'vector')
          expect(Array.from(result['offsetLinks'])).toEqual([3, 1, 0, 2, 1, 3, 2, 0]);
      }
    } else if (kind === 'circle') {
      const props = {
        ...names,
        featureCount: 1,
        tolerance: 0.1,
        maximumIterations: 0,
        cellCapacity: 4,
        output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
      };
      const operations = [
        new GPUMaximumInscribedCircleOperation(props),
        new CPUMaximumInscribedCircleOperation(props)
      ];
      props.maximumIterations = 64;
      for (const operation of operations) {
        const inputs = {
          ...rectangle,
          positions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 8, 0, 0, 6)},
          ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 3)}
        };
        const result = await runOperationGraph(device, {
          inputs,
          operations: [operation],
          outputs: Object.values(props.output)
        });
        expect(Array.from(result['status'])).toEqual([1]);
        expect(Array.from(result['witnesses'])).toEqual([0, 6]);
        expect(Array.from(result['circles']).slice(0, 3)).toEqual([0, 6, 0]);
        // GPU retains the root Lipschitz gap; CPU also applies the envelope cap.
        expect(result['circles'][3]).toBeCloseTo(
          operation instanceof CPUMaximumInscribedCircleOperation ? 3 : 4 * Math.SQRT2,
          5
        );
      }
    } else {
      const props = {
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
      const operations = [
        new GPURatioConcaveHullOperation(props),
        new CPURatioConcaveHullOperation(props)
      ];
      props.maximumIterations = 64;
      for (const operation of operations) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {
              format: 'float32x2',
              values: Float32Array.of(
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
              )
            }
          },
          operations: [operation],
          outputs: ['triangleValidity', 'converged']
        });
        expect(Array.from(result['triangleValidity']).reduce((sum, value) => sum + value, 0)).toBe(
          11
        );
        expect(Array.from(result['converged'])).toEqual([0]);
      }
    }
  });
