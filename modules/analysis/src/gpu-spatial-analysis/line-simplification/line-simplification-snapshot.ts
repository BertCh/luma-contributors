// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPULineSimplificationProps} from './gpu-line-simplification';
import type {LineSimplificationOperationProps} from './gpu-line-simplification-operation';

export const LINE_SIMPLIFICATION_STATUS_FIELDS = [
  'count',
  'overflow',
  'candidateOverflow',
  'requiredCount',
  'converged',
  'iterationCount'
] as const;

function snapshotOutput<Output extends object & {ids?: unknown; status?: object}>(
  id: string,
  output: Output | undefined,
  computeImportance: boolean
): Output | undefined {
  if (!output) return undefined;
  const captured = snapshotConstructionFields(output, [
    'ids',
    'outputMask',
    'lineCounts',
    'lineStarts',
    'status'
  ] as (keyof Output)[]);
  const status = captured.status;
  if (status) {
    const checkName = (name: string | symbol) =>
      validateRequirement(
        id,
        typeof name === 'string' &&
          (LINE_SIMPLIFICATION_STATUS_FIELDS as readonly string[]).includes(name),
        `output.status.${String(name)} is unsupported; use the named selection or convergence diagnostics`
      );
    for (
      let descriptor = status;
      descriptor && descriptor !== Object.prototype;
      descriptor = Object.getPrototypeOf(descriptor)
    ) {
      for (const name of Reflect.ownKeys(descriptor)) {
        if (descriptor !== status) {
          const property = Object.getOwnPropertyDescriptor(descriptor, name);
          if (name === 'constructor' || (property && typeof property.value === 'function'))
            continue;
        }
        checkName(name);
      }
    }
  }
  // Explicit names include inherited/non-enumerable getters. Capture values once,
  // before any validation or deferred operation factory can reread the caller.
  const capturedStatus =
    status &&
    snapshotConstructionFields(
      status as Record<string, unknown>,
      LINE_SIMPLIFICATION_STATUS_FIELDS
    );
  const hasSelection = captured.ids !== undefined;
  const statusValues = capturedStatus as Record<string, unknown> | undefined;
  for (const field of ['count', 'overflow', 'requiredCount', 'candidateOverflow']) {
    validateRequirement(
      id,
      hasSelection || statusValues?.[field] === undefined,
      `output.status.${field} requires output.ids for selection; omit selection diagnostics in importance-only mode`
    );
  }
  for (const field of ['outputMask', 'lineCounts', 'lineStarts']) {
    validateRequirement(
      id,
      hasSelection || (captured as Record<string, unknown>)[field] === undefined,
      `output.${field} requires output.ids for selection`
    );
  }
  validateRequirement(
    id,
    computeImportance ||
      (statusValues?.['converged'] === undefined && statusValues?.['iterationCount'] === undefined),
    'output.status.converged and iterationCount require computeImportance'
  );
  return Object.freeze({...captured, status: capturedStatus}) as Output;
}

export function snapshotLineSimplificationProps(
  props: GPULineSimplificationProps
): GPULineSimplificationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'positions',
    'lineOffsets',
    'spatialContext',
    'timestamps',
    'method',
    'metric',
    'neighborhoodRadius',
    'importance',
    'computeImportance',
    'parameters',
    'maximumIterations',
    'finishSpanLimit',
    'output'
  ]);
  return Object.freeze({
    ...captured,
    spatialContext:
      captured.spatialContext && snapshotConstructionSpatialContext(captured.spatialContext),
    output: snapshotOutput(
      captured.id ?? 'line-simplification',
      captured.output,
      captured.computeImportance !== false
    )
  });
}

export function snapshotLineSimplificationOperationProps(
  props: LineSimplificationOperationProps
): LineSimplificationOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'positions',
    'lineOffsets',
    'spatialContext',
    'timestamps',
    'method',
    'metric',
    'neighborhoodRadius',
    'importance',
    'computeImportance',
    'parameters',
    'maximumIterations',
    'finishSpanLimit',
    'rowCount',
    'lineCount',
    'capacity',
    'output'
  ]);
  const output = snapshotOutput(
    captured.id ?? 'line-simplification',
    captured.output,
    captured.computeImportance !== false
  );
  validateRequirement(
    captured.id ?? 'line-simplification',
    captured.computeImportance !== false || output?.ids !== undefined,
    'computeImportance=false requires output.ids for selection'
  );
  validateRequirement(
    captured.id ?? 'line-simplification',
    output?.ids === undefined ||
      (captured.parameters !== undefined &&
        output.status?.count !== undefined &&
        output.status?.overflow !== undefined),
    'output.ids requires parameters and status.count/overflow'
  );
  return Object.freeze({
    ...captured,
    spatialContext:
      captured.spatialContext && snapshotConstructionSpatialContext(captured.spatialContext),
    output
  });
}
