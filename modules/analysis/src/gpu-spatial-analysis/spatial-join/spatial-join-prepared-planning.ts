// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../ontology/spatial-analysis-ontology';
import {
  validateCapacity,
  validatePowerOfTwo,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getPolygonEdgeLayoutLength
} from './spatial-join-passes';
import {POLYGON_EDGE_SLAB_PARAMS_STRIDE} from './polygon-edge-slabs';

/** Shape of the whole-domain prepared index; partitioning never reduces these allocations. */
export type GPUSpatialJoinPreparedShape = {
  kind: 'points' | 'lines' | 'polygons';
  featureCount: number;
  vertexCount: number;
  polygonCount?: number;
  ringCount?: number;
  /** Multipart point/line featureOffsets are allocated separately from parts. */
  multipart?: boolean;
  /** Allocated line parts for multipart lines; ordinary lines default to featureCount. */
  partCount?: number;
  leafCapacity?: number;
  spatialSort?: boolean;
  edgeSlabs?: boolean;
  /** Caller storage removes internal tree allocations, but not their binding checks. */
  externalStorage?: boolean;
};

export type GPUSpatialJoinPreparedDeviceLimits = Pick<
  Device['limits'],
  'maxStorageBufferBindingSize' | 'maxBufferSize'
>;

/** @internal Constructor and planner use exactly the same power-of-two leaf selection. */
export function resolveSpatialJoinPreparedShape(
  shape: GPUSpatialJoinPreparedShape,
  id = 'GPUSpatialJoinPrepared'
) {
  validateRequirement(
    id,
    ['points', 'lines', 'polygons'].includes(shape.kind),
    `kind must be points, lines or polygons; received ${shape.kind}`
  );
  validateCapacity(id, 'featureCount', shape.featureCount);
  validateCapacity(id, 'vertexCount', shape.vertexCount, 0);
  const leafCapacity = shape.leafCapacity ?? getNextPowerOfTwo(Math.max(shape.featureCount, 1));
  validatePowerOfTwo(id, 'leafCapacity', leafCapacity, 1, 0x80000000);
  return {
    leafCapacity,
    spatialSort: shape.spatialSort ?? getDefaultSpatialSort(shape.featureCount)
  };
}

const SOURCE =
  'modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts';
const ACCURACY =
  'Whole-domain BVH over the retained geometry; partitioning does not tile the index. A leaf capacity below featureCount loses candidates.';

/** Executable allocation plan. Caller offsets/aliases and graph ownership are checked at declaration. */
export function getGPUSpatialJoinPreparedMemoryPlan(
  shape: GPUSpatialJoinPreparedShape,
  limits: GPUSpatialJoinPreparedDeviceLimits
) {
  const {leafCapacity, spatialSort} = resolveSpatialJoinPreparedShape(shape);
  const {featureCount, vertexCount} = shape;
  const sort = spatialSort && featureCount > 1;
  const nodeCount = 2 * leafCapacity - 1;
  const scratchByteLengths: Record<string, number> = {
    'generation byte table': 1024,
    'generation state': 16,
    'requested generation bytes': 32,
    'BVH node bounds/children': shape.externalStorage ? 0 : nodeCount * 8,
    'BVH leaf IDs': shape.externalStorage ? 0 : leafCapacity * 4,
    'BVH status': 4,
    'feature bounds and sorted bounds': featureCount * 8,
    'feature radix histogram': sort ? 16 * Math.ceil(featureCount / 256) * 4 : 0,
    'feature sort tile/scene bounds': sort ? Math.max(1, Math.ceil(featureCount / 256)) * 16 : 0,
    'feature ring ranges':
      shape.kind === 'polygons' && !shape.externalStorage ? featureCount * 8 : 0
  };
  const geometryByteLengths: Record<string, number> = {
    'geometry positions snapshot': vertexCount * 8
  };
  if (shape.kind === 'polygons' || shape.multipart)
    geometryByteLengths['geometry featureOffsets snapshot'] = (featureCount + 1) * 4;
  if (shape.kind === 'lines') {
    validateRequirement(
      'GPUSpatialJoinPrepared',
      !shape.multipart || shape.partCount !== undefined,
      'partCount is required for multipart lines; use the allocated lineOffsets.length - 1'
    );
    const partCount = shape.partCount ?? featureCount;
    validateCapacity('GPUSpatialJoinPrepared', 'partCount', partCount, 0);
    geometryByteLengths['geometry lineOffsets snapshot'] = (partCount + 1) * 4;
  }
  if (shape.kind === 'polygons') {
    validateCapacity('GPUSpatialJoinPrepared', 'polygonCount', shape.polygonCount ?? NaN, 0);
    validateCapacity('GPUSpatialJoinPrepared', 'ringCount', shape.ringCount ?? NaN, 0);
    geometryByteLengths['geometry polygonOffsets snapshot'] = (shape.polygonCount! + 1) * 4;
    geometryByteLengths['geometry ringOffsets snapshot'] = (shape.ringCount! + 1) * 4;
  }
  scratchByteLengths['snapshot gate'] = Math.max(
    4,
    12 * Object.values(geometryByteLengths).filter(bytes => bytes > 0).length
  );
  const slabEntryCapacity = shape.edgeSlabs
    ? Math.min(
        Math.max(0, Math.floor(limits.maxStorageBufferBindingSize / 8) - vertexCount),
        Math.max(4 * vertexCount, 1024)
      )
    : 0;
  validateRequirement(
    'GPUSpatialJoinPrepared',
    !shape.edgeSlabs || shape.kind === 'polygons',
    'edgeSlabs requires polygon geometry'
  );
  if (shape.edgeSlabs) {
    validateCapacity('GPUSpatialJoinPrepared', 'polygonCount', shape.polygonCount ?? NaN, 0);
    validateCapacity('GPUSpatialJoinPrepared', 'ringCount', shape.ringCount ?? NaN, 0);
    Object.assign(scratchByteLengths, {
      'edge slab parameters':
        Math.max(shape.polygonCount!, 1) * POLYGON_EDGE_SLAB_PARAMS_STRIDE * 4,
      'edge slab owner scans': (shape.polygonCount! + 1) * 4,
      'edge slab ring owners': Math.max(shape.ringCount!, 1) * 4,
      'edge slab vertex owners': vertexCount * 4,
      'edge slab counters/offsets': (Math.floor(vertexCount / 4) + 2) * 4,
      'edge slab entries': (vertexCount + slabEntryCapacity) * 8,
      'edge slab layout': getPolygonEdgeLayoutLength(vertexCount, featureCount) * 4
    });
  }
  // The position snapshot and tree are whole-domain even when input partition metadata is supplied.
  const bindingByteLengths = {
    ...scratchByteLengths,
    ...geometryByteLengths,
    'BVH node bounds/children': nodeCount * 8,
    'BVH leaf IDs': leafCapacity * 4
  };
  const maximum = Math.min(
    limits.maxStorageBufferBindingSize,
    limits.maxBufferSize,
    0xffffffff * 4
  );
  const constraints = Object.entries(bindingByteLengths).map(([name, received]) => ({
    name,
    kind: 'allocation' as const,
    quantity: name,
    unit: 'bytes' as const,
    appliesWhen: 'whole-domain prepared index, including partitioned input',
    maximum,
    byteExpression: name === 'BVH node bounds/children' ? '8 * (2 * leafCapacity - 1)' : undefined,
    scope: 'binding' as const,
    accuracyImplications: ACCURACY,
    source: SOURCE,
    received,
    satisfied: received <= maximum
  }));
  return {
    route: spatialSort ? 'sorted-whole-domain-bvh' : 'whole-domain-bvh',
    feasible: constraints.every(constraint => constraint.satisfied),
    constraints,
    accuracyImplications: [ACCURACY],
    leafCapacity,
    nodeCount,
    spatialSort,
    slabEntryCapacity,
    scratchByteLengths,
    bindingByteLengths,
    largestBindingBytes: Math.max(...Object.values(bindingByteLengths))
  };
}

/** Catalog planning contract backed by the same sizing used during declaration. */
export const GPU_SPATIAL_JOIN_PREPARED_PLANNER: GPUAnalysisExecutionPlanner<
  GPUSpatialJoinPreparedShape,
  GPUSpatialJoinPreparedDeviceLimits
> = {getExecutionPlan: getGPUSpatialJoinPreparedMemoryPlan};

/** Descriptive catalog route; use the planner to check a particular shape and device. */
export const GPU_SPATIAL_JOIN_PREPARED_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'whole-domain-bvh',
    kind: 'allocation',
    quantity: 'padded BVH node bounds/children',
    unit: 'bytes',
    appliesWhen: 'all prepared routes, including partitioned input',
    byteExpression: '8 * (2 * (leafCapacity ?? nextPowerOfTwo(featureCount)) - 1)',
    scope: 'binding',
    alternative: 'use independent prepared handles over separate domains',
    accuracyImplications: ACCURACY,
    source: SOURCE,
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPU_SPATIAL_JOIN_PREPARED_PLANNER'
    }
  }
];
