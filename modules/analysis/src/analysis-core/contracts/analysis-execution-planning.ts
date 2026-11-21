// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** The nine public entry points whose contributors the operator catalog covers. */
export type GPUAnalysisPackage =
  | '@luma-contributors/analysis/computation-record'
  | '@luma-contributors/analysis/gpu-spatial-analysis'
  | '@luma-contributors/analysis/gpu-network'
  | '@luma-contributors/analysis/gpu-terrain'
  | '@luma-contributors/analysis/raster-analysis'
  | '@luma-contributors/analysis/dataframe-analysis'
  | '@luma-contributors/analysis/pointcloud-analysis'
  | '@luma-contributors/analysis/geospatial-analysis'
  | '@luma-contributors/analysis/analysis-core';

/** Public identity: the full entry-point subpath and its exported name. */
export type GPUAnalysisOperatorIdentity = Readonly<{
  package: GPUAnalysisPackage;
  exportName: string;
}>;

/** A route constraint, distinct from a crossover and from a whole-domain maximum. */
export type GPUAnalysisRoute = Readonly<{
  name: string;
  kind: 'hard-limit' | 'auto-crossover' | 'allocation';
  quantity: string;
  unit: 'rows' | 'bytes' | 'commands' | 'elements';
  appliesWhen: string;
  conditionExpression?: string;
  maximum?: number;
  minimum?: number;
  byteExpression?: string;
  scope: 'chunk' | 'domain' | 'binding' | 'output' | 'dispatch';
  alternative?: string;
  accuracyImplications: string;
  /** Planner identity, when verified against construction; absence means descriptive only. */
  planner?: GPUAnalysisOperatorIdentity;
  source: string;
}>;

/** Family planners implement this contract; descriptive metadata alone never certifies fit. */
export type GPUAnalysisExecutionPlanner<Shape, DeviceLimits> = Readonly<{
  getExecutionPlan: (
    shape: Shape,
    deviceLimits: DeviceLimits
  ) => Readonly<{
    route: string;
    feasible: boolean;
    constraints: readonly (GPUAnalysisRoute & Readonly<{satisfied: boolean; received: number}>)[];
    accuracyImplications: readonly string[];
  }>;
}>;
