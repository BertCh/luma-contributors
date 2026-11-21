// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../../analysis-core/contracts/analysis-execution-planning';
import type {GPUSpatialCoordinateSpace} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUNeighborSearchStructure} from './gpu-neighbor-search';
import {
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS
} from './neighbor-search-limits';
import {NEIGHBOR_SEARCH_BVH_NODE_WORDS} from './neighbor-search-kernels';
import {getNextPowerOfTwo} from '../spatial-join/spatial-join-passes';

/** Numeric admission inputs; no buffers or GPU allocations are needed. */
export type GPUNeighborSearchShape = {
  mode: 'radius' | 'knn';
  searchStructure?: GPUNeighborSearchStructure;
  coordinateSpace?: GPUSpatialCoordinateSpace;
  hasTimes?: boolean;
  targetRowCount: number;
  queryRowCount?: number;
  /** Explicit cross-join route; defaults to whether query and target row counts differ. */
  hasQueryPositions?: boolean;
  /** A self-join query mask requires a separate grid order. */
  hasQueryMask?: boolean;
  includeInvalidCount?: boolean;
  /** Include the internally allocated distance column when no output distances are supplied. */
  distanceCapacity?: number;
  gridSize: readonly [number, number];
  timeBucketCount?: number;
  neighborCount?: number;
};

export type GPUNeighborSearchDeviceLimits = Pick<
  Device['limits'],
  'maxBufferSize' | 'maxStorageBufferBindingSize'
>;

const SOURCE = 'modules/analysis/src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts';
const PLANNER = {
  package: '@luma-contributors/analysis/gpu-spatial-analysis',
  exportName: 'GPUNeighborSearch'
} as const;
const ACCURACY =
  'Grid and BVH preserve exact neighbor selection; capacity truncation may retain different neighbors in the cut row.';

/** Catalog-compatible descriptors for the route and the numeric checks performed before allocation. */
export const GPU_NEIGHBOR_SEARCH_ROUTES: readonly GPUAnalysisRoute[] = Object.freeze([
  {
    name: 'auto-grid',
    kind: 'auto-crossover',
    quantity: 'searchStructure',
    unit: 'elements',
    conditionExpression: 'searchStructure omitted or auto => grid; explicit bvh => bvh',
    appliesWhen: 'all shapes; bvh requires planar coordinates and no times',
    scope: 'domain',
    alternative: 'explicit planar bvh without times',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'gridSize cells',
    kind: 'allocation',
    quantity: 'gridSize[0] * gridSize[1] * timeBucketCount',
    unit: 'bytes',
    byteExpression:
      '(gridSize[0] * gridSize[1] * timeBucketCount + 1) * 4 <= maxStorageBufferBindingSize',
    appliesWhen: 'all routes, including explicit bvh (preserves existing grid admission)',
    scope: 'binding',
    alternative: 'reduce gridSize or timeBucketCount',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'k-nearest lists',
    kind: 'allocation',
    quantity: 'queryRowCount * neighborCount',
    unit: 'bytes',
    byteExpression: 'queryRowCount * neighborCount * 4 <= maxStorageBufferBindingSize',
    appliesWhen: 'mode knn; one binding each for IDs and distances',
    scope: 'binding',
    alternative: 'lower neighborCount or split query rows',
    accuracyImplications:
      'Lowering neighborCount changes the requested neighborhood; splitting queries preserves it.',
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'knn',
    kind: 'hard-limit',
    quantity: 'neighborCount',
    unit: 'rows',
    maximum: GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
    appliesWhen: 'mode knn; required, ignored in radius mode',
    scope: 'output',
    alternative: 'radius',
    accuracyImplications: 'Radius selects by distance rather than a fixed neighbor count.',
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'time buckets',
    kind: 'hard-limit',
    quantity: 'timeBucketCount',
    unit: 'elements',
    maximum: GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS,
    appliesWhen: 'times supplied; defaults to 1',
    scope: 'domain',
    alternative: 'reduce timeBucketCount',
    accuracyImplications: 'Temporal bucket sizing changes traversal, not the selected time window.',
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'grid records',
    kind: 'allocation',
    quantity: 'target and separately indexed query records',
    unit: 'bytes',
    byteExpression:
      'max(targetRowCount, separateQueryIndex ? queryRowCount : 0) * (hasTimes ? 4 : 3) * 4 <= min(maxBufferSize,maxStorageBufferBindingSize)',
    appliesWhen: 'selected grid route; separate query index for a cross join or query mask',
    scope: 'binding',
    alternative: 'split query rows; split target domains only when the neighborhood permits it',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  },
  {
    name: 'bvh records',
    kind: 'allocation',
    quantity: 'power-of-two padded BVH nodes',
    unit: 'bytes',
    byteExpression:
      '(2 * max(2, nextPowerOfTwo(targetRowCount)) - 1) * 5 * 4 <= min(maxBufferSize,maxStorageBufferBindingSize)',
    appliesWhen:
      'selected bvh route; node bounds/children and sorted feature records also checked by the planner',
    scope: 'binding',
    alternative: 'grid',
    accuracyImplications: ACCURACY,
    planner: PLANNER,
    source: SOURCE
  }
]);

/** Resolves the actual selected route; auto deliberately uses grid at every size. @internal */
export function resolveGPUNeighborSearchStructure(
  id: string,
  searchStructure: GPUNeighborSearchStructure = 'auto',
  coordinateSpace: GPUSpatialCoordinateSpace = 'planar',
  hasTimes = false
): 'grid' | 'bvh' {
  validateRequirement(
    id,
    ['auto', 'grid', 'bvh'].includes(searchStructure),
    `searchStructure must be auto, grid or bvh; received ${String(searchStructure)}`
  );
  validateRequirement(
    id,
    coordinateSpace === 'planar' || coordinateSpace === 'longitude-latitude',
    `coordinateSpace received ${coordinateSpace}; expected planar or longitude-latitude`
  );
  validateRequirement(
    id,
    searchStructure !== 'bvh' || coordinateSpace === 'planar',
    `searchStructure 'bvh' supports planar positions only; received ${coordinateSpace}; use grid`
  );
  validateRequirement(
    id,
    searchStructure !== 'bvh' || !hasTimes,
    `times need searchStructure 'grid' or 'auto'; received ${searchStructure}`
  );
  return searchStructure === 'bvh' ? 'bvh' : 'grid';
}

export type GPUNeighborSearchExecutionPlan = ReturnType<
  GPUAnalysisExecutionPlanner<
    GPUNeighborSearchShape,
    GPUNeighborSearchDeviceLimits
  >['getExecutionPlan']
> & {
  route: 'grid' | 'bvh';
  gridCellCount: number;
  gridDimension: 2 | 3;
  recordWords: 3 | 4;
  nearestListLength: number;
  bvhLeafCapacity: number;
  bvhNodeCount: number;
  bindingRequirements: readonly {name: string; byteLength: number}[];
};

/**
 * Plans route and numeric grid/list admission with the same rules as declaration. `feasible`
 * covers these declared constraints; actual views still require graph ownership, binding-range,
 * resource-usage and output-alias checks. Auto never allocates an unused BVH.
 */
export function getGPUNeighborSearchExecutionPlan(
  shape: GPUNeighborSearchShape,
  limits: GPUNeighborSearchDeviceLimits
): GPUNeighborSearchExecutionPlan {
  const id = 'GPUNeighborSearch.getExecutionPlan';
  validateRequirement(
    {id, contributor: 'GPUNeighborSearch', messagePrefix: `${id}:`},
    !('k' in shape),
    'k was renamed; use neighborCount'
  );
  validateRequirement(
    id,
    shape.mode === 'knn' || shape.mode === 'radius',
    `mode received ${String(shape.mode)}; expected knn or radius`
  );
  const route = resolveGPUNeighborSearchStructure(
    id,
    shape.searchStructure,
    shape.coordinateSpace,
    shape.hasTimes
  );
  validateCapacity(id, 'targetRowCount', shape.targetRowCount, 1, 2 ** 31 - 1);
  const queryRowCount = shape.queryRowCount ?? shape.targetRowCount;
  validateCapacity(id, 'queryRowCount', queryRowCount, 1, 2 ** 31 - 1);
  for (const axis of [0, 1] as const)
    validateCapacity(id, `gridSize[${axis}]`, shape.gridSize[axis], 1, Number.MAX_SAFE_INTEGER);
  const timeBucketCount = shape.hasTimes ? (shape.timeBucketCount ?? 1) : 1;
  validateRequirement(
    id,
    Boolean(shape.hasTimes) || shape.timeBucketCount === undefined,
    `timeBucketCount received ${String(shape.timeBucketCount)}; timeBucketCount needs times`
  );
  validateCapacity(
    id,
    'timeBucketCount',
    timeBucketCount,
    1,
    GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS
  );
  const neighborCount = shape.mode === 'knn' ? (shape.neighborCount ?? Number.NaN) : 1;
  if (shape.mode === 'knn')
    validateCapacity(id, 'neighborCount', neighborCount, 1, GPU_NEIGHBOR_SEARCH_MAXIMUM_K);
  const maximum = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const gridCellCount = shape.gridSize[0] * shape.gridSize[1] * timeBucketCount;
  const nearestListLength = shape.mode === 'knn' ? queryRowCount * neighborCount : 0;
  const gridDimension = timeBucketCount > 1 ? 3 : 2;
  const recordWords = shape.hasTimes ? 4 : 3;
  const crossJoin = shape.hasQueryPositions ?? queryRowCount !== shape.targetRowCount;
  validateRequirement(
    id,
    crossJoin || queryRowCount === shape.targetRowCount,
    `queryRowCount must equal targetRowCount without query positions; received ${queryRowCount} and ${shape.targetRowCount}`
  );
  const sharesTargetIndex = !crossJoin && !shape.hasQueryMask;
  const bvhLeafCapacity =
    route === 'bvh' ? Math.max(2, getNextPowerOfTwo(shape.targetRowCount)) : 0;
  const bvhNodeCount = route === 'bvh' ? 2 * bvhLeafCapacity - 1 : 0;
  const bindingRequirements = [
    {name: 'gridPositions', byteLength: shape.targetRowCount * gridDimension * 4},
    {name: 'query counts and starts (each)', byteLength: queryRowCount * 4},
    ...(route === 'grid'
      ? [
          {name: 'sortedTargets', byteLength: shape.targetRowCount * recordWords * 4},
          ...(!sharesTargetIndex
            ? [
                {name: 'sortedQueries', byteLength: queryRowCount * recordWords * 4},
                {name: 'queryGridPositions', byteLength: queryRowCount * gridDimension * 4}
              ]
            : [])
        ]
      : [
          {name: 'BVH nodeData', byteLength: bvhNodeCount * NEIGHBOR_SEARCH_BVH_NODE_WORDS * 4},
          {name: 'BVH minima, maxima and children (each)', byteLength: bvhNodeCount * 8},
          {name: 'BVH leafIds', byteLength: bvhLeafCapacity * 4},
          {name: 'BVH feature and sorted bounds (each)', byteLength: shape.targetRowCount * 8}
        ]),
    ...(shape.includeInvalidCount
      ? [
          {
            name: 'invalid flags and starts (each)',
            byteLength: (shape.targetRowCount + (crossJoin ? queryRowCount : 0)) * 4
          }
        ]
      : []),
    ...(shape.distanceCapacity !== undefined
      ? [{name: 'distance scratch', byteLength: shape.distanceCapacity * 4}]
      : [])
  ];
  const constraints = [
    {
      ...GPU_NEIGHBOR_SEARCH_ROUTES[1],
      maximum,
      received: (gridCellCount + 1) * 4,
      satisfied: (gridCellCount + 1) * 4 <= maximum
    },
    ...(shape.mode === 'knn'
      ? [
          {
            ...GPU_NEIGHBOR_SEARCH_ROUTES[2],
            maximum,
            received: nearestListLength * 4,
            satisfied: nearestListLength * 4 <= maximum
          }
        ]
      : []),
    ...bindingRequirements.map(({name, byteLength}) => ({
      name,
      kind: 'allocation' as const,
      quantity: name,
      unit: 'bytes' as const,
      appliesWhen: `selected ${route} route`,
      scope: 'binding' as const,
      accuracyImplications: ACCURACY,
      planner: PLANNER,
      source: SOURCE,
      maximum,
      received: byteLength,
      satisfied: Number.isSafeInteger(byteLength) && byteLength >= 0 && byteLength <= maximum,
      byteExpression: `${name} <= min(maxStorageBufferBindingSize,maxBufferSize)`
    }))
  ];
  return {
    route,
    feasible: constraints.every(constraint => constraint.satisfied),
    constraints,
    accuracyImplications: [ACCURACY],
    gridCellCount,
    gridDimension,
    recordWords,
    nearestListLength,
    bvhLeafCapacity,
    bvhNodeCount,
    bindingRequirements
  };
}
