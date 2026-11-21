// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {readFileSync} from 'node:fs';
import typescript from 'typescript';
import {expect, expectTypeOf, it, vi} from 'vitest';
import * as spatial from '@luma-contributors/analysis/gpu-spatial-analysis';
import * as ontology from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type * as network from '@luma-contributors/analysis/gpu-network';
import type * as terrain from '@luma-contributors/analysis/gpu-terrain';
import type {
  CPUBrownianBridgeUtilizationResult,
  BrownianBridgeTailPolicy,
  CPUHealpixPointToCellResult,
  CPUMultipleTestingAdjustmentResult,
  CPUNearestFeatureJoinResult,
  CPUPointInPolygonCountsResult,
  CPUSpatialLagImpactsResult,
  CPUSpatialPredicateJoinResult,
  CPUSpatialWeightsValidationResult,
  GPUBoundedFacilityAllocationOutput,
  GPUBufferSelectionMatches,
  GPURankMarkovProps,
  GPUSpatialJoinGeometryReadback,
  GPUSpatialJoinPreparedMemoryPlan,
  GPUSpatialPredicateJoinRouteShape,
  GPUStopValueCubeRecipeProps,
  GPUStopValueCubeRecipeResult,
  HealpixDGGSJSONFromReadbackProps,
  PrepareGeometryInputOptions,
  PreparedGeometryInput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisOutputDeclaration,
  GPUAnalysisOutputDescription,
  GPUAnalysisRoute,
  GPUAnalysisTaskResult,
  GPUAnalysisArrayForm,
  GPUAnalysisFormContract
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {
  CPUNetworkTimeRoutingResult,
  CPUNetworkTravelTimePercentilesResult,
  CPURadiationFlowModelResult
} from '@luma-contributors/analysis/gpu-network';
import type {CPUTerrainCatchmentSummaryOperationProps} from '@luma-contributors/analysis/gpu-terrain';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {GPU_SPATIAL_PREDICATE_JOIN_PLANNER} from '../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-route-plan';
import {GPU_RANK_MARKOV_PLANNER} from '../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';

function getNamedFamilyExports(family: string): string[] {
  const source = readFileSync(
    new URL(`../../src/gpu-spatial-analysis/${family}/index.ts`, import.meta.url),
    'utf8'
  );
  const syntax = typescript.createSourceFile('index.ts', source, typescript.ScriptTarget.Latest);
  return syntax.statements.flatMap(statement =>
    typescript.isExportDeclaration(statement) &&
    statement.exportClause &&
    typescript.isNamedExports(statement.exportClause)
      ? statement.exportClause.elements.map(element => element.name.text)
      : []
  );
}

function expectCompactScalarLength(
  view: GPUBufferSelectionMatches['status']['requiredCount']
): void {
  expect(view?.length).toBe(1);
}

it('B1 rejects a missing optional compact scalar with an assertion failure', () => {
  expect(() => expectCompactScalarLength(undefined)).toThrow(/expected undefined to be 1/);
});

it('CPL-07 exports bounded allocation diagnostics with a separate completeness certificate', () => {
  expect(getNamedFamilyExports('location-allocation')).toContain(
    'GPUBoundedFacilityAllocationOutput'
  );
  expectTypeOf<GPUBoundedFacilityAllocationOutput>().toEqualTypeOf<
    spatial.GPUBoundedFacilityAllocationProps['output']
  >();
  // The three diagnostic words cannot stand in for the independent positive certificate.
  const diagnosticStorage = Uint32Array.of(99, 0, 1, 0, 99);
  const diagnostics = new Uint8Array(diagnosticStorage.buffer, 4, 12);
  const certificate = Uint32Array.of(0);
  expect(spatial.decodeGPUBoundedFacilityAllocationStatus(diagnostics, certificate)).toEqual({
    unassignedDemandCount: 0,
    validInput: 1,
    globalOptimum: 0,
    complete: 0
  });
  // A completed capacitated allocation has complete=1 without claiming a global optimum.
  expect(spatial.decodeGPUBoundedFacilityAllocationStatus(diagnostics, Uint32Array.of(1))).toEqual({
    unassignedDemandCount: 0,
    validInput: 1,
    globalOptimum: 0,
    complete: 1
  });
  expect(() =>
    spatial.decodeGPUBoundedFacilityAllocationStatus(diagnostics, new Uint32Array())
  ).toThrow();
  expect(() =>
    spatial.decodeGPUBoundedFacilityAllocationStatus(diagnostics.subarray(1), certificate)
  ).toThrow();
});

it('GEO-112 exports the compact matches type and preserves public row/count/loss views', () => {
  expect(getNamedFamilyExports('spatial-join')).toContain('GPUBufferSelectionMatches');
  expectTypeOf<GPUBufferSelectionMatches>().toEqualTypeOf<
    ReturnType<spatial.GPUBufferSelection['getMatches']>
  >();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const supplied = {
      ids: createTransientView(graph, 'selected-rows', 'uint32', 2),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required-count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1)
      }
    };
    const selection = new spatial.GPUBufferSelection({
      // No sourceIds: compact ids identify rows of these two points.
      points: {kind: 'points', positions: createTransientView(graph, 'points', 'float32x2', 2)},
      features: {
        kind: 'points',
        positions: createTransientView(graph, 'features', 'float32x2', 1)
      },
      radius: createTransientView(graph, 'radius', 'float32', 1),
      candidateCapacity: 2,
      spatialSort: false,
      querySort: false,
      output: {matches: supplied}
    });
    const matches: GPUBufferSelectionMatches = selection.getMatches(graph);
    expect(selection.getMatches(graph)).toBe(matches);
    expect(matches.ids).toBe(supplied.ids);
    expect(matches.ids.length).toBe(2);
    for (const name of ['count', 'requiredCount', 'overflow', 'candidateOverflow'] as const) {
      expect(matches.status[name]).toBe(supplied.status[name]);
      expectCompactScalarLength(matches.status[name]);
    }
  } finally {
    device.destroy();
  }
});

it('ING-12 publishes the exact accepted spatial-relate slot boundary before allocation', () => {
  // Two words per slot: floor((2^32 - 1) / 2) = 2,147,483,647 slots.
  expect(spatial.MAXIMUM_SPATIAL_RELATE_PATTERN_SLOT_COUNT).toBe(2147483647);
  const originalConstructor = Uint32Array;
  const allocationLengths: number[] = [];
  const allocationStopped = new Error('boundary accepted; stop before allocating 16 GiB');
  let acceptedBoundaryError: unknown;
  let rejectedBoundaryError: unknown;
  vi.stubGlobal(
    'Uint32Array',
    new Proxy(originalConstructor, {
      construct(_target, argumentsList) {
        allocationLengths.push(argumentsList[0] as number);
        throw allocationStopped;
      }
    })
  );
  try {
    try {
      spatial.packGPUSpatialRelatePattern('T********', 2147483647);
    } catch (error) {
      acceptedBoundaryError = error;
    }
    try {
      spatial.packGPUSpatialRelatePattern('T********', 2147483648);
    } catch (error) {
      rejectedBoundaryError = error;
    }
  } finally {
    vi.unstubAllGlobals();
  }
  expect(acceptedBoundaryError).toBe(allocationStopped);
  expect(allocationLengths).toEqual([4294967294]);
  expect(String(rejectedBoundaryError)).toMatch(/slotCount.*2147483647/);
  expect(rejectedBoundaryError).toMatchObject({received: 2147483648});
});

it('SCL-16 preserves public predicate planner identity and the exact auto crossover', () => {
  expect(spatial.GPU_SPATIAL_PREDICATE_JOIN_PLANNER).toBe(GPU_SPATIAL_PREDICATE_JOIN_PLANNER);
  const shape: GPUSpatialPredicateJoinRouteShape = {
    predicate: 'intersects',
    leftKind: 'polygons',
    rightKind: 'polygons',
    leftCount: 1,
    rightCount: 1,
    leftVertexCount: 31,
    rightVertexCount: 32
  };
  // Average vertex products 31*32=992 and 32*32=1024 straddle the documented crossover.
  expect(spatial.getGPUSpatialPredicateJoinExecutionPlan(shape).route).toBe('fast');
  expect(
    spatial.GPU_SPATIAL_PREDICATE_JOIN_PLANNER.getExecutionPlan(
      {...shape, leftVertexCount: 32},
      undefined
    ).route
  ).toBe('relate');
  expect(spatial.GPU_SPATIAL_PREDICATE_JOIN_ROUTES[0].minimum).toBe(1024);
});

it('W5-10 publishes an S2 per-invocation edge tile rather than a total input ceiling', () => {
  expect(spatial.MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION).toBe(4096);
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['serial', 'level-synchronous'] as const) {
      const graph = new GPUCommandGraph(device);
      const covering = new spatial.GPUS2PolygonCovering({
        positions: createTransientView(graph, 'positions', 'float32x3', 4097),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
        maximumLevel: 0,
        maximumVisitedCells: 6,
        capacity: 6,
        refinement,
        output: {
          cells: createTransientView(graph, 'cells', 'uint32x2', 6),
          counts: createTransientView(graph, 'counts', 'uint32x2', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      });
      const nodes = covering.getCommandNodes(graph);
      expect(
        nodes.find(node => node.id.endsWith('-prepare-vertices'))?.workload?.maximumInvocationCount
      ).toBe(4097);
      const longTileNode = nodes.find(node => node.id.includes('-long-tiles-'));
      expect(longTileNode).toBeDefined();
      const longTileKernel = getGPUCommandNodeKernelSources(nodes).find(
        kernel => kernel.id === longTileNode?.id
      );
      expect(longTileKernel?.source).toContain('const COVER_INPUT_TILE:u32=4096u;');
      expect(longTileKernel?.source).toContain('vertex-begin<COVER_INPUT_TILE');
      expect(longTileKernel?.source).toContain('let start=tiles[descriptor+2u];');
      expect(longTileKernel?.source).toContain(
        'coverRingContribution(cell,begin,end,start,min(end,start+4096u))'
      );
    }
  } finally {
    device.destroy();
  }
});

it('W5-11 exposes the Rank Markov planner shared with the class on both ranking routes', () => {
  expect(spatial.GPU_RANK_MARKOV_PLANNER).toBe(GPU_RANK_MARKOV_PLANNER);
  expect(spatial.GPU_RANK_MARKOV_PLANNER.getExecutionPlan).toBe(
    spatial.getGPURankMarkovExecutionPlan
  );
  expect(spatial.GPU_RANK_MARKOV_ROUTES).toBe(spatial.GPURankMarkov.routes);
  const descriptor = spatial.GPU_RANK_MARKOV_ROUTES[0];
  const publicExports = spatial as unknown as Record<string, unknown>;
  expect(publicExports[descriptor.planner!.exportName]).toBe(spatial.GPURankMarkov);
  const device = createNullWebGPUDevice();
  try {
    // Auto chooses the pairwise reference through 64 rows, then sorting at 65 rows.
    for (const [rows, expectedRoute] of [
      [64, 'pairwise'],
      [65, 'sorted']
    ] as const) {
      const graph = new GPUCommandGraph(device);
      const props: GPURankMarkovProps = {
        values: createTransientView(graph, 'values', 'float32', rows * 2),
        periodCount: 2,
        output: {
          ranks: createTransientView(graph, 'ranks', 'uint32', rows * 2),
          counts: createTransientView(graph, 'counts', 'uint32', rows * rows)
        }
      };
      const plan = spatial.getGPURankMarkovExecutionPlan(props, device.limits);
      expect(plan.route).toBe(expectedRoute);
      expect(plan.feasible).toBe(true);
      expect(spatial.GPURankMarkov.getExecutionPlan(props, device.limits)).toEqual(plan);
      expect(spatial.GPU_RANK_MARKOV_PLANNER.getExecutionPlan(props, device.limits)).toEqual(plan);
      expect(new spatial.GPURankMarkov(props).rankingMethod).toBe(expectedRoute);
      expect(spatial.getGPURankMarkovExecutionPlan({...props, rankingMethod: 'sorted'}).route).toBe(
        'sorted'
      );
    }
  } finally {
    device.destroy();
  }
});

it('FST-09 / UGE-15 exposes sparse CPU assignments without dense membership', () => {
  const result: CPUPointInPolygonCountsResult = spatial.computePointInPolygonCountsOnCPU(
    {
      points: {kind: 'points', positions: Float32Array.of(1, 1, 3, 1)},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0, 4, 0, 4, 2, 0, 2, 0, 0),
        featureOffsets: Uint32Array.of(0, 1, 2),
        polygonOffsets: Uint32Array.of(0, 1, 2),
        ringOffsets: Uint32Array.of(0, 5, 10)
      }
    },
    {candidateCapacity: 4, matchCapacity: 2}
  );
  // (1,1) is in both rectangles and assigned to row 0; (3,1) is only in row 1.
  expect(result.count).toEqual(Uint32Array.of(2));
  expect(result.polygonAssignmentCounts).toEqual(Uint32Array.of(1, 1));
  expect(result.polygonMembershipCounts).toEqual(Uint32Array.of(1, 2));
  expect(result.polygonAssignmentCounts).not.toEqual(result.polygonMembershipCounts);
  expect(result.pointFeatureRows).toEqual(Uint32Array.of(0, 1));
  expect(result.overflow).toEqual(Uint32Array.of(0));
  expect(result).not.toHaveProperty('featurePolygonMembership');
});

it('S1 sparse counts never allocate and discard the Cartesian-product membership matrix', () => {
  const pointCount = 128,
    polygonCount = 64;
  const square = [0, 0, 2, 0, 2, 2, 0, 2, 0, 0];
  const input: Parameters<typeof spatial.computePointInPolygonCountsOnCPU>[0] = {
    points: {kind: 'points', positions: new Float32Array(pointCount * 2).fill(10)},
    polygons: {
      kind: 'polygons',
      positions: Float32Array.from(
        {length: polygonCount * 10},
        (_, index) => square[index % square.length]
      ),
      featureOffsets: Uint32Array.from({length: polygonCount + 1}, (_, index) => index),
      polygonOffsets: Uint32Array.from({length: polygonCount + 1}, (_, index) => index),
      ringOffsets: Uint32Array.from({length: polygonCount + 1}, (_, index) => index * 5)
    }
  };
  const originalConstructor = Uint32Array;
  const allocationLengths: number[] = [];
  const guardedConstructor = new Proxy(originalConstructor, {
    construct(target, argumentsList) {
      const length = argumentsList[0];
      if (typeof length === 'number') {
        allocationLengths.push(length);
        if (length >= pointCount * polygonCount) throw new Error('S1 dense membership allocation');
      }
      return Reflect.construct(target, argumentsList);
    }
  });
  vi.stubGlobal('Uint32Array', guardedConstructor);
  try {
    const result = spatial.computePointInPolygonCountsOnCPU(input, {
      candidateCapacity: 1,
      matchCapacity: 0
    });
    expect(result.count[0]).toBe(0); // Every point is outside every [0,2] square.
    expect(result).not.toHaveProperty('featurePolygonMembership');
  } finally {
    vi.unstubAllGlobals();
  }
  expect(allocationLengths.length).toBeGreaterThan(0);
  expect(allocationLengths.every(length => length < 8192)).toBe(true);
});

it('B1 resolves the memory-plan declaration import under packed NodeNext ESM rules', () => {
  const source = readFileSync(
    new URL('../../src/gpu-spatial-analysis/spatial-join/index.ts', import.meta.url),
    'utf8'
  );
  const specifier = /typeof import\('([^']+)'\)/.exec(source)?.[1];
  expect(specifier).toBeDefined();
  const directory = '/packed/analysis/dist/gpu-spatial-analysis/spatial-join';
  const target = `${directory}/spatial-join-prepared-planning.d.ts`;
  // A virtual package is sufficient to check this declaration edge; no compiler/build runs.
  const files = new Map([
    ['/packed/analysis/package.json', '{"type":"module"}'],
    [target, 'export declare function getGPUSpatialJoinPreparedMemoryPlan(): unknown;']
  ]);
  const resolution = typescript.resolveModuleName(
    specifier!,
    `${directory}/index.d.ts`,
    {
      module: typescript.ModuleKind.NodeNext,
      moduleResolution: typescript.ModuleResolutionKind.NodeNext
    },
    {
      fileExists: path => files.has(path),
      readFile: path => files.get(path)
    },
    undefined,
    undefined,
    typescript.ModuleKind.ESNext
  );
  expect(resolution.resolvedModule?.resolvedFileName).toBe(target);
});

it('B2 resolves the predicate route qualified planner identity and calls the public planner', () => {
  const route = spatial.GPU_SPATIAL_PREDICATE_JOIN_ROUTES[0];
  expect(route.planner).toEqual({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'GPU_SPATIAL_PREDICATE_JOIN_PLANNER'
  });
  const exports = spatial as unknown as Record<string, unknown>;
  const planner = exports[route.planner!.exportName] as GPUAnalysisExecutionPlanner<
    GPUSpatialPredicateJoinRouteShape,
    unknown
  >;
  const shape: GPUSpatialPredicateJoinRouteShape = {
    predicate: 'intersects',
    leftKind: 'polygons',
    rightKind: 'polygons',
    leftCount: 1,
    rightCount: 1,
    leftVertexCount: 32,
    rightVertexCount: 32
  };
  expect(planner.getExecutionPlan(shape, undefined).route).toBe('relate');
  expect(spatial.getGPUSpatialPredicateJoinExecutionPlan(shape).route).toBe('relate');
  // Two words per pattern, with the last complete slot below the uint32 word bound.
  expect(spatial.MAXIMUM_SPATIAL_RELATE_PATTERN_SLOT_COUNT).toBe(2147483647);
});

it('EGR-03 exposes HEALPix readback export with the completeness guard intact', () => {
  const props: HealpixDGGSJSONFromReadbackProps = {
    order: 1,
    ancestor: 0n,
    properties: {count: {values: [4]}},
    output: {
      // The second cell is capacity tail and must never become an observation.
      cells: Uint32Array.of(0, 0, 1, 0),
      status: {count: 1, requiredCount: 1, overflow: 0}
    }
  };
  const result = spatial.createHealpixDGGSJSONFromReadback(props);
  // Order one has four children; only the first has an observed value.
  expect(result.values['count'][0].data).toEqual([4, null, null, null]);
  for (const [status, message] of [
    [{count: 1, requiredCount: 2, overflow: 0}, /requiredCount 2/],
    [{count: 1, requiredCount: 1, overflow: 1}, /overflow 1/]
  ] as const) {
    expect(() =>
      spatial.createHealpixDGGSJSONFromReadback({
        ...props,
        output: {...props.output, status}
      })
    ).toThrow(message);
  }
});

it('ING-16 exposes checked native Polygon ingress with metric XY and source identity', () => {
  const coordinateType = new arrow.FixedSizeList(2, new arrow.Field('xy', new arrow.Float64()));
  const ringType = new arrow.List(new arrow.Field('vertices', coordinateType));
  const polygonType = new arrow.List(new arrow.Field('rings', ringType));
  const column = arrow.vectorFromArray(
    [
      [
        [
          [10, 20],
          [30, 20],
          [10, 35],
          [10, 20]
        ]
      ],
      null
    ],
    polygonType
  );
  const options: PrepareGeometryInputOptions = {
    encoding: 'geoarrow.polygon',
    sourceRowOffset: 7,
    sourceIds: Uint32Array.of(91, 92)
  };
  const result: PreparedGeometryInput = spatial.prepareGeometryInput(column, options);
  // The adapter retains actual coordinates, not equality-token positions.
  expect(result.geometry.positions).toEqual(Float32Array.of(10, 20, 30, 20, 10, 35, 10, 20));
  expect(result.geometry.featureOffsets).toEqual(Uint32Array.of(0, 1, 1));
  expect(result.validity).toEqual(Uint32Array.of(1, 0));
  expect(result.sourceRows).toEqual(Uint32Array.of(7, 8));
  expect(result.sourceIds).toEqual(Uint32Array.of(91, 92));
});

it('SCL-14 exports the whole-domain memory plan before allocation', () => {
  const plan: GPUSpatialJoinPreparedMemoryPlan = spatial.getGPUSpatialJoinPreparedMemoryPlan(
    {kind: 'points', featureCount: 8388609, vertexCount: 8388609},
    {maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456}
  );
  // Next power of two is 16,777,216; each of 2*leaves-1 BVH nodes needs 8 bytes.
  expect(plan.leafCapacity).toBe(16777216);
  expect(plan.largestBindingBytes).toBe(268435448);
  expect(plan.feasible).toBe(false);
});

it('CMP-12 exports the named recipe result and discriminated loss stages', () => {
  expect(spatial.addStopValueCubeRecipe).toBeTypeOf('function');
  expectTypeOf<GPUStopValueCubeRecipeProps['time']>().toEqualTypeOf<'start' | 'finish'>();
  expectTypeOf<GPUStopValueCubeRecipeResult>().toEqualTypeOf<
    ReturnType<typeof spatial.addStopValueCubeRecipe>
  >();
  expectTypeOf<GPUStopValueCubeRecipeResult['status']['stages'][number]['stage']>().toEqualTypeOf<
    'stops' | `slice-${number}` | 'cube'
  >();
  expectTypeOf<GPUStopValueCubeRecipeResult['status']['stages'][number]['status']>().toHaveProperty(
    'overflow'
  );
  expectTypeOf<GPUStopValueCubeRecipeResult['status']['stages'][number]['status']>().toHaveProperty(
    'candidateOverflow'
  );
  expectTypeOf<GPUStopValueCubeRecipeProps['stops']['status']>().toHaveProperty('overflow');
  expectTypeOf<GPUStopValueCubeRecipeProps['stops']['status']>().toHaveProperty(
    'candidateOverflow'
  );
});

it('TYP-16 provides one cross-entry fixture for all ten CPU results and terrain props', () => {
  expectTypeOf<CPUHealpixPointToCellResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeHealpixPointToCellOnCPU>
  >();
  expectTypeOf<CPUMultipleTestingAdjustmentResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeMultipleTestingAdjustmentOnCPU>
  >();
  expectTypeOf<CPUSpatialPredicateJoinResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeSpatialPredicateJoinOnCPU>
  >();
  expectTypeOf<CPUNearestFeatureJoinResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeNearestFeatureJoinOnCPU>
  >();
  expectTypeOf<CPUSpatialLagImpactsResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeSpatialLagImpactsOnCPU>
  >();
  expectTypeOf<CPUSpatialWeightsValidationResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeSpatialWeightsValidationOnCPU>
  >();
  expectTypeOf<CPUBrownianBridgeUtilizationResult>().toEqualTypeOf<
    ReturnType<typeof spatial.computeBrownianBridgeUtilizationOnCPU>
  >();
  expectTypeOf<CPUNetworkTimeRoutingResult>().toEqualTypeOf<
    ReturnType<typeof network.computeNetworkTimeRoutingOnCPU>
  >();
  expectTypeOf<CPURadiationFlowModelResult>().toEqualTypeOf<
    ReturnType<typeof network.computeRadiationFlowModelOnCPU>
  >();
  expectTypeOf<CPUNetworkTravelTimePercentilesResult>().toEqualTypeOf<
    ReturnType<typeof network.computeNetworkTravelTimePercentilesOnCPU>
  >();
  expectTypeOf<CPUTerrainCatchmentSummaryOperationProps>().toEqualTypeOf<
    ConstructorParameters<typeof terrain.CPUTerrainCatchmentSummaryOperation>[0]
  >();
});

it('EGR-01 existing public spherical count readers preserve byte offsets and whole words', () => {
  // Each unaligned subview contains a valid producer count layout.
  const bytes = new Uint8Array(18);
  const view = new DataView(bytes.buffer);
  [300, 2, 400, 1].forEach((value, index) => view.setUint32(1 + 4 * index, value, true));
  const convexBytes = new Uint8Array(10);
  const convexView = new DataView(convexBytes.buffer);
  convexView.setUint32(1, 4, true);
  convexView.setUint32(5, 4, true);
  expect(spatial.decodeGPUSphericalConvexIntersectionCounts(convexBytes.subarray(1, 9))).toEqual({
    writtenVertexCount: 4,
    requiredVertexCount: 4
  });
  expect(spatial.decodeGPUSphericalArcOverlayCounts(bytes.subarray(1, 17))).toEqual({
    writtenVertexCount: 300,
    writtenRingCount: 2,
    requiredSegmentCount: 400,
    referenceInside: true
  });
  expect(() =>
    spatial.decodeGPUSphericalConvexIntersectionCounts(convexBytes.subarray(1, 8))
  ).toThrow();
  expect(() => spatial.decodeGPUSphericalArcOverlayCounts(bytes.subarray(1, 13))).toThrow();
});

it('handoffs expose the delivered geometry reader, CSR decoder and task search', () => {
  const input: GPUSpatialJoinGeometryReadback = {
    kind: 'points',
    positions: Float32Array.of(10, 20, 30, 40)
  };
  expect(spatial.decodeGPUSpatialJoinGeometry(input, 1).positions).toEqual(Float32Array.of(10, 20));
  expect(
    spatial.decodeGPUVisibilityGraphAdjacencyCSRStatus(Uint32Array.of(2, 2, 0, 0, 0))
  ).toMatchObject({count: 2, requirement: {kind: 'exact', value: 2}});
  const results: readonly GPUAnalysisTaskResult[] = ontology.searchGPUAnalysisTasks('NDVI');
  expect(results.some(result => result.identity.exportName === 'GPURasterNDVI')).toBe(true);
  expectTypeOf<BrownianBridgeTailPolicy>().toEqualTypeOf<'full' | 'truncate-at-seven-sigma'>();
  expectTypeOf<GPUAnalysisArrayForm['contract']>().toEqualTypeOf<
    GPUAnalysisFormContract | undefined
  >();
});

it('SCL-16 exports a route descriptor and executable planner contract from the public entry', () => {
  const route: GPUAnalysisRoute = spatial.GPU_SPATIAL_JOIN_PREPARED_ROUTES[0];
  type Planner = GPUAnalysisExecutionPlanner<
    Parameters<typeof spatial.getGPUSpatialJoinPreparedMemoryPlan>[0],
    Parameters<typeof spatial.getGPUSpatialJoinPreparedMemoryPlan>[1]
  >;
  const planner: Planner = {getExecutionPlan: spatial.getGPUSpatialJoinPreparedMemoryPlan};
  expect(route.scope).toBe('binding');
  expect(planner.getExecutionPlan).toBeTypeOf('function');
});

it('UCO-17 exposes output descriptions with qualified identities unchanged', () => {
  const identity = {
    package: '@luma-contributors/analysis/gpu-spatial-analysis' as const,
    exportName: 'GPULineChunk'
  };
  const result: GPUAnalysisOutputDescription = ontology.describeOutputs(identity, 'chunk', {
    lineCapacity: 3,
    vertexCapacity: 8
  });
  const offsets: GPUAnalysisOutputDeclaration | undefined = result.outputs.find(
    output => output.path === 'output.lineOffsets'
  );
  expect(offsets?.capacity).toBe(4); // Three line rows plus the terminal offset.
  expect(offsets?.status).toEqual(['output.status.count', 'output.status.overflow']);
  expect(identity).toEqual({
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    exportName: 'GPULineChunk'
  });
  expect(
    ontology.describeOutputs(
      {...identity, package: '@luma-contributors/analysis/gpu-network'},
      'chunk',
      {lineCapacity: 3, vertexCapacity: 8}
    )
  ).toEqual({declaration: 'absent', outputs: []});
});
