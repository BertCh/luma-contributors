// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getViewBindingRange, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';
import {
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Omit device limits for topology-only planning; pass actual limits to certify binding sizes. */
export type GPUStatisticsPlanDeviceLimits = {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
};
export type GPUStatisticsRouteCheck = GPUAnalysisRoute & {received: number; satisfied: boolean};

/** Evaluates numeric descriptors; auto crossovers select a route and do not reject a request. */
export function evaluateStatisticsRoute(
  descriptor: GPUAnalysisRoute,
  received: number
): GPUStatisticsRouteCheck {
  return {
    ...descriptor,
    received,
    satisfied:
      Number.isFinite(received) &&
      Number.isInteger(received) &&
      received >= (descriptor.minimum ?? 0) &&
      received <= (descriptor.maximum ?? Infinity)
  };
}

/** Appends the same global command budget and binding/allocation limits used by constructors. */
export function getStatisticsExecutionPlan<Route extends string>(
  route: Route,
  constraints: GPUStatisticsRouteCheck[],
  estimatedCommandNodeCount: number,
  bindings: Readonly<Record<string, number>>,
  deviceLimits: GPUStatisticsPlanDeviceLimits | undefined,
  descriptor: GPUAnalysisRoute
) {
  const checks = [
    ...constraints,
    evaluateStatisticsRoute(
      {
        ...descriptor,
        name: 'command-schedule',
        kind: 'hard-limit',
        quantity: 'estimatedCommandNodeCount',
        unit: 'commands',
        scope: 'domain',
        maximum: GPU_MAXIMUM_COMMAND_NODE_COUNT,
        minimum: 0,
        appliesWhen: `selected route ${route}`
      },
      estimatedCommandNodeCount
    )
  ];
  if (deviceLimits)
    for (const [name, bytes] of Object.entries(bindings)) {
      checks.push(
        evaluateStatisticsRoute(
          {
            ...descriptor,
            name,
            kind: 'allocation',
            quantity: name,
            unit: 'bytes',
            scope: 'binding',
            minimum: 0,
            maximum: Math.min(deviceLimits.maxStorageBufferBindingSize, deviceLimits.maxBufferSize),
            byteExpression: name,
            appliesWhen: `selected route ${route}`
          },
          bytes
        )
      );
    }
  return {
    route,
    feasible: checks.every(check => check.kind === 'auto-crossover' || check.satisfied),
    constraints: checks.map(check =>
      check.kind === 'auto-crossover' ? check : {...check, alternative: undefined}
    ),
    estimatedCommandNodeCount,
    bindings,
    deviceChecked: Boolean(deviceLimits),
    accuracyImplications: [descriptor.accuracyImplications]
  };
}

/** Checks numeric feasibility before allocating any scratch or emitting commands. */
export function validateStatisticsExecutionPlan(
  id: string,
  plan: ReturnType<typeof getStatisticsExecutionPlan>
): void {
  const failed = plan.constraints.find(
    check => check.kind !== 'auto-crossover' && !check.satisfied
  );
  validateRequirement(
    id,
    !failed,
    failed
      ? failed.unit === 'bytes'
        ? `${failed.quantity} needs ${failed.received} bytes in one binding; limit is ${failed.maximum} bytes (maxStorageBufferBindingSize/maxBufferSize)`
        : `${failed.quantity} received ${failed.received}; requires [${failed.minimum ?? 0}, ${failed.maximum ?? Infinity}] for ${plan.route}`
      : 'execution plan must be feasible'
  );
}

/** Storage bindings include the view's alignment prefix, matching graph binding preflight. */
export function getStatisticsViewBindingByteLength(view: GraphDataView): number {
  return getViewBindingRange(view).size;
}

/** Numeric prerequisite attached to the same descriptor contract as route limits. */
export function getStatisticsRequirementCheck(
  descriptor: GPUAnalysisRoute,
  name: string,
  satisfied: boolean
): GPUStatisticsRouteCheck {
  return evaluateStatisticsRoute(
    {
      ...descriptor,
      name,
      quantity: name,
      kind: 'hard-limit',
      unit: 'elements',
      scope: 'output',
      maximum: 0,
      minimum: 0,
      alternative: undefined,
      appliesWhen: 'selected inputs and outputs'
    },
    Number(!satisfied)
  );
}
