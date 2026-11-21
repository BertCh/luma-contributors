// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';

/** Numeric device limits used by density allocation and dispatch preflight. */
export type GPUDensityDeviceLimits = Readonly<{
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}>;

/** A numeric route plan; constraints describe the supplied shape, without allocating GPU resources. */
export type GPUDensityExecutionPlan = Readonly<{
  route: string;
  feasible: boolean;
  constraints: readonly (GPUAnalysisRoute & Readonly<{satisfied: boolean; received: number}>)[];
  accuracyImplications: readonly string[];
}>;

/** @internal Evaluate source-backed numeric bounds in the catalog's route vocabulary. */
export function makeDensityExecutionPlan(
  route: string,
  limits: readonly GPUAnalysisRoute[],
  received: readonly number[],
  accuracyImplications: readonly string[]
): GPUDensityExecutionPlan {
  const constraints = limits.map((limit, index) => ({
    ...limit,
    received: received[index],
    satisfied:
      Number.isFinite(received[index]) &&
      (limit.minimum === undefined || received[index] >= limit.minimum) &&
      (limit.maximum === undefined || received[index] <= limit.maximum)
  }));
  return {
    route,
    feasible: constraints.every(constraint => constraint.satisfied),
    constraints,
    accuracyImplications
  };
}

/** Logical chunks must describe exactly the live row count claimed by the shape. */
export function hasDensityShapeRows(pointCount: number, chunks: readonly number[]): boolean {
  return (
    chunks.every(length => Number.isSafeInteger(length) && length >= 0) &&
    chunks.reduce((total, length) => total + length, 0) === pointCount
  );
}
