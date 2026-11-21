// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import type {GPUPolygonizeOutput} from './topology-types';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Capture supported and retired declared fields, including inherited getters, exactly once. */
export function snapshotTopologyStatus<Status extends object>(status: Status): Status {
  return snapshotConstructionFields(status, [
    'count',
    'requiredCount',
    'requiredRingCount',
    'overflow',
    'candidateOverflow',
    'uncertainCount',
    'invalidCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation'
  ] as (keyof Status)[]);
}

/** Different stage/meaning status outputs may not alias; uncertainty copies may share storage. */
export function validateTopologyStatusAliases(id: string, output: GPUPolygonizeOutput): void {
  const fields = [
    ...Object.entries(output.status).map(([name, view]) => ({
      path: `output.status.${name}`,
      name,
      view
    })),
    ...Object.entries(output.nodedSegments.status).map(([name, view]) => ({
      path: `output.nodedSegments.status.${name}`,
      name,
      view
    })),
    ...Object.entries(output.directedEdges.status).map(([name, view]) => ({
      path: `output.directedEdges.status.${name}`,
      name,
      view
    }))
  ];
  for (let first = 0; first < fields.length; first++)
    for (let second = first + 1; second < fields.length; second++) {
      const left = fields[first];
      const right = fields[second];
      if (left.name === 'uncertainCount' && right.name === 'uncertainCount') continue;
      validateOutlineOutputPasses(id, {[left.path]: left.view, [right.path]: right.view}, []);
    }
}

/** Reject generic status fields that topology kernels do not publish, before snapshotting. */
export function validateSupportedTopologyStatus(id: string, status: object, path: string): void {
  for (const name of [
    'invalidCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation'
  ]) {
    validateRequirement(
      id,
      !(name in status),
      `${path}.${name} is unsupported by fixed-pass topology; use count, overflow, candidateOverflow and uncertainCount for completeness`
    );
  }
}

/** Checks final feature and ring cardinality scalars without conflating their storage. */
export function validatePolygonizeStatus(id: string, status: GPUPolygonizeOutput['status']): void {
  validateRequirement(
    id,
    !('requiredCount' in status),
    'output.status.requiredCount is not published for final polygon features; use output.status.requiredRingCount for ring capacity'
  );
  validateRequirement(
    id,
    Boolean(status.requiredRingCount && status.candidateOverflow),
    'output.status requires requiredRingCount and candidateOverflow'
  );
  validateTopologyStatusScalars(id, status, 'output.status');
  validateOutlineOutputPasses(`${id} output.status`, status, []);
}

/** Validate every supplied scalar before an internal destination can replace it. */
export function validateTopologyStatusScalars(id: string, status: object, path: string): void {
  for (const [name, view] of Object.entries(status)) {
    if (!view) continue;
    validateSinglePackedView(view, ['uint32'], `${id} ${path}.${name}`);
    validateCapacity(id, `${path}.${name}.length`, view.length, 1);
  }
}
