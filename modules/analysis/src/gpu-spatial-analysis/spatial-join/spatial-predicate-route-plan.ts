// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../../analysis-core/contracts/analysis-execution-planning';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {SpatialPredicateName} from './spatial-predicate-wgsl';

/** Average-vertex product at which auto topology selects the cooperative relate route. */
export const SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT = 1024;

/** Numeric metadata used by construction to select an exact predicate implementation. */
export type GPUSpatialPredicateJoinRouteShape = {
  predicate: SpatialPredicateName;
  engine?: 'auto' | 'fast' | 'relate';
  leftKind: 'points' | 'lines' | 'polygons';
  rightKind: 'points' | 'lines' | 'polygons';
  leftCount: number;
  rightCount: number;
  leftVertexCount: number;
  rightVertexCount: number;
  multipart?: boolean;
  outputRelate?: boolean;
  /** Selects great-circle distance for `dwithin`. Omit for topological predicates. */
  greatCircle?: boolean;
};

const SOURCE =
  'modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts';
const ACCURACY =
  'Topological routes use exact orientation signs on supplied float32 coordinates; dwithin remains float32 metric arithmetic. Cooperative dwithin retains uncertainty in candidate flag bit 1 and reduces it into uncertainCount.';

const PREDICATES = [
  'intersects',
  'contains',
  'within',
  'dwithin',
  'covers',
  'coveredBy',
  'touches',
  'crosses',
  'overlaps',
  'equals',
  'containsProperly',
  'relate'
] as const;

export const GPU_SPATIAL_PREDICATE_JOIN_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'relate-auto-crossover',
    kind: 'auto-crossover',
    quantity: 'average left vertices * average right vertices',
    unit: 'elements',
    appliesWhen: 'engine auto; intersects/contains/within; non-point singlepart operands',
    minimum: SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT,
    scope: 'domain',
    alternative: 'engine fast',
    accuracyImplications: ACCURACY,
    source: SOURCE,
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPU_SPATIAL_PREDICATE_JOIN_PLANNER'
    }
  }
];

/** Resolves the same routes as the GPU constructor; contains no GPU allocations. */
export function getGPUSpatialPredicateJoinExecutionPlan(
  shape: GPUSpatialPredicateJoinRouteShape,
  id = 'GPUSpatialPredicateJoin'
) {
  validateRequirement(
    id,
    (PREDICATES as readonly string[]).includes(shape.predicate),
    `predicate must be ${PREDICATES.join(', ')}; received ${shape.predicate}`
  );
  validateRequirement(
    id,
    !shape.greatCircle || shape.predicate === 'dwithin',
    `greatCircle requires predicate 'dwithin' when true. Received '${shape.predicate}'. Omit greatCircle for topological predicates.`
  );
  for (const name of ['leftCount', 'rightCount', 'leftVertexCount', 'rightVertexCount'] as const)
    validateCapacity(id, name, shape[name], name.endsWith('VertexCount') ? 0 : 1);
  for (const name of ['leftKind', 'rightKind'] as const)
    validateRequirement(
      id,
      ['points', 'lines', 'polygons'].includes(shape[name]),
      `${name} must be points, lines or polygons; received ${shape[name]}`
    );
  const engine = shape.engine ?? 'auto';
  validateSpatialPredicateRelateOutput(id, shape.predicate, Boolean(shape.outputRelate));
  validateRequirement(
    id,
    ['auto', 'fast', 'relate'].includes(engine),
    `engine must be auto, fast or relate; received ${engine}`
  );
  const legacy =
    ['intersects', 'contains', 'within', 'dwithin'].includes(shape.predicate) ||
    Boolean(shape.multipart && shape.predicate === 'coveredBy');
  validateRequirement(
    id,
    engine !== 'fast' || legacy,
    'engine fast requires intersects, contains, within, dwithin or multipart coveredBy'
  );
  validateRequirement(
    id,
    !shape.greatCircle || engine !== 'relate',
    'engine relate is unavailable for great-circle dwithin; use auto or fast'
  );
  const vertexProduct =
    (shape.leftVertexCount / Math.max(shape.leftCount, 1)) *
    (shape.rightVertexCount / Math.max(shape.rightCount, 1));
  const autoRelate =
    !shape.multipart &&
    shape.leftKind !== 'points' &&
    shape.rightKind !== 'points' &&
    vertexProduct >= SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT;
  const prefersWorkgroups = engine === 'relate' || (engine === 'auto' && autoRelate);
  const usesRelateEngine =
    !legacy || Boolean(shape.outputRelate) || (shape.predicate !== 'dwithin' && prefersWorkgroups);
  const usesWorkgroupDistance =
    shape.predicate === 'dwithin' && prefersWorkgroups && !shape.greatCircle;
  return {
    route: usesWorkgroupDistance ? 'workgroup-distance' : usesRelateEngine ? 'relate' : 'fast',
    feasible: true,
    constraints: GPU_SPATIAL_PREDICATE_JOIN_ROUTES.map(route => ({
      ...route,
      received: vertexProduct,
      satisfied: vertexProduct >= SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT
    })),
    accuracyImplications: [ACCURACY],
    usesRelateEngine,
    usesWorkgroupDistance
  };
}

/** @internal Shared precondition for the planner and contributor constructor. */
export function validateSpatialPredicateRelateOutput(
  id: string,
  predicate: SpatialPredicateName,
  outputRelate: boolean
): void {
  validateRequirement(
    id,
    !outputRelate || predicate !== 'dwithin',
    'output.relate needs a topological predicate because dwithin classifies metric proximity; use a separate relate join for DE-9IM output'
  );
}

/** Catalog planner for route selection; capacity/device allocation planning is separate. */
export const GPU_SPATIAL_PREDICATE_JOIN_PLANNER: GPUAnalysisExecutionPlanner<
  GPUSpatialPredicateJoinRouteShape,
  unknown
> = {
  getExecutionPlan: (shape: GPUSpatialPredicateJoinRouteShape) =>
    getGPUSpatialPredicateJoinExecutionPlan(shape)
};
