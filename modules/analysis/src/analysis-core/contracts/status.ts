// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUStatusScalar
} from '../kernels/gpu-contributor-types';
import {validateRequirement} from '../kernels/gpu-contributor-utils';

// The scalar, cardinality and overflow ports sit in utils so the compact-output contract there
// is typed with them; this module is their public home.
export type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUStatusScalar
} from '../kernels/gpu-contributor-types';

/** Completeness of an intermediate candidate stage. */
export type GPUCandidateStatusPort = {
  /** One when refinement may be incomplete because an intermediate capacity was too small. */
  candidateOverflow?: GPUStatusScalar;
};

/** Numerical classification fallback and remaining unresolved work of a result or named stage. */
export type GPUUncertaintyStatusPort = {
  /** Exact number of classifications resolved through the documented uncertainty fallback. */
  uncertainCount?: GPUStatusScalar;
  /** Number of classifications or work items still unresolved after the stage's available work. */
  unresolvedCount?: GPUStatusScalar;
};

/** Completeness certificate for the producer's documented result, or one named recipe stage. */
export type GPUCompletenessStatusPort = {
  /**
   * One when the entire promised result is present, accounting for all supplied loss evidence;
   * zero otherwise. In `GPURecipeStageStatusPort.status`, certifies only the stage named by
   * `stage`, not the aggregate recipe. The producer documents the exact result being certified.
   * Omission provides no certificate and must not be interpreted as completeness.
   */
  complete?: GPUStatusScalar;
};

/** Lon/lat polygon rings a degree-plane consumer could not interpret (results may be wrong there). */
export type GPULonLatPolygonStatusPort = {
  /** Rings with an edge whose raw longitude step exceeds 180 (antimeridian crossing, not split). */
  antimeridianUncertainCount?: GPUStatusScalar;
  /** Rings whose wrapped longitude steps sum to +-360 (enclose a pole, not capped). */
  poleUncertainCount?: GPUStatusScalar;
};

/** Input validity diagnostics. */
export type GPUInvalidStatusPort = {
  /**
   * Number of input rows rejected by validation or, for `GPURasterFillNoData`, the number of
   * cells of the result that are still invalid after the fill.
   */
  invalidCount?: GPUStatusScalar;
};

/**
 * Status for bounded iterative work: the one convergence-status shape. Solvers
 * (`GPUSolverStatusPort`), searches and iterative contributors use
 * these field names for the executed iteration count and the stopping flags.
 */
export type GPUIterationStatusPort = {
  /** Number of iterations, rounds or passes actually executed. */
  iterationCount?: GPUStatusScalar;
  /** One when the operation met its stopping condition. */
  converged?: GPUStatusScalar;
  /** One when the iteration budget was exhausted before convergence. */
  iterationLimitReached?: GPUStatusScalar;
};

/** Status for algorithms that intentionally return an approximation. */
export type GPUApproximationStatusPort = {
  /** One when the result uses an intentional approximation rather than exact semantics. */
  approximation?: GPUStatusScalar;
};

/**
 * Composable status vocabulary for a capacity-bounded analytical result. Concrete writable
 * outputs must select the fields they actually write and reject unsupported requests before
 * snapshotting. This vocabulary alone does not promise writers for every optional field.
 */
export type GPUBoundedResultStatusPort = GPUCardinalityPort &
  GPUCompletenessStatusPort &
  GPUOverflowStatusPort &
  GPUCandidateStatusPort &
  GPUUncertaintyStatusPort &
  GPUInvalidStatusPort &
  GPUIterationStatusPort &
  GPUApproximationStatusPort;

/**
 * Vertex sub-count of generated geometry (contributor contract rule 8). `count` and
 * `requiredCount` of generated geometry count features; the vertex totals have these names.
 */
export type GPUVertexCountStatusPort = {
  /** Number of vertices written, clamped to the vertex capacity. */
  vertexCount?: GPUStatusScalar;
  /** Unclamped number of vertices the result needs. */
  requiredVertexCount?: GPUStatusScalar;
};

/** Ring sub-count of generated polygon geometry (contributor contract rule 8). */
export type GPURingCountStatusPort = {
  /** Number of rings written, clamped to the ring capacity. */
  ringCount?: GPUStatusScalar;
  /** Unclamped number of rings the result needs. */
  requiredRingCount?: GPUStatusScalar;
};

/**
 * Status of generated geometry (contributor contract rule 8): `count` and `requiredCount` count
 * output features, `overflow` is one when any final capacity (features, rings or vertices) dropped
 * rows, and the sub-levels have their own named counts.
 */
export type GPUGeneratedGeometryStatusPort = GPUBoundedResultStatusPort &
  GPUVertexCountStatusPort &
  GPURingCountStatusPort;

/** Status attributed to one named recipe stage; `status.complete` certifies only that stage. */
export type GPURecipeStageStatusPort = {
  stage: string;
  status: Partial<GPUBoundedResultStatusPort>;
  /**
   * Optional one-row result code of a stage whose contributor reports a code rather than row
   * counts (for example a solver's `status.code`): 0 on success, otherwise the contributor's
   * documented failure code. Test it against 0; it is not a count and must not be summed.
   */
  code?: GPUStatusScalar;
};

/** Aggregate status that preserves the stage responsible for each condition. */
export type GPURecipeStatusPort = {
  stages: readonly GPURecipeStageStatusPort[];
};

function validateStatusScalar(id: string, name: string, scalar: GPUStatusScalar): void {
  validateRequirement(
    id,
    scalar.format === 'uint32' && scalar.length === 1,
    `${name} must contain one uint32 row`,
    {
      code: 'REQUIREMENT',
      path: name,
      received: {format: scalar.format, length: scalar.length},
      expected: {format: 'uint32', length: 1}
    }
  );
}

/** Validates the format and scalar shape of every present status field. */
export function validateGPUStatusPort(
  id: string,
  status: Partial<GPUBoundedResultStatusPort> & GPULonLatPolygonStatusPort
): void {
  for (const [name, scalar] of Object.entries(status)) {
    if (!scalar) {
      continue;
    }
    validateStatusScalar(id, name, scalar);
  }
}

/** Validates every stage of a recipe status aggregate. */
export function validateGPURecipeStatusPort(id: string, status: GPURecipeStatusPort): void {
  const stages = new Set<string>();
  for (const stage of status.stages) {
    if (!stage.stage || stages.has(stage.stage)) {
      // Stage names identify the operation responsible for an incomplete result.
      throw new Error(`${id} recipe status stages must have unique non-empty names`);
    }
    stages.add(stage.stage);
    validateGPUStatusPort(`${id} stage ${stage.stage}`, stage.status);
    if (stage.code) {
      validateStatusScalar(`${id} stage ${stage.stage}`, 'code', stage.code);
    }
  }
}
