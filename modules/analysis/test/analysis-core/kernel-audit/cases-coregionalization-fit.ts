// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCoregionalizationFit,
  type GPUCoregionalizationFitProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Maximum reference-tier shape; all named status ports wired. */
export function createCoregionalizationAuditProps({
  view
}: KernelAuditContext): GPUCoregionalizationFitProps {
  return {
    variableCount: 4,
    structureCount: 4,
    maximumIterations: 1024,
    distances: view('float32', 256),
    semivariances: view('float32', 2560),
    pairCounts: view('uint32', 256),
    parameters: view('float32', 17),
    output: {
      matrices: view('float32', 64),
      loadings: view('float32', 64),
      diagnostics: view('float32x4', 1),
      status: {
        validity: view('uint32', 1),
        iterationCount: view('uint32', 1),
        converged: view('uint32', 1),
        iterationLimitReached: view('uint32', 1),
        invalidCount: view('uint32', 1)
      }
    }
  };
}
export const COREGIONALIZATION_FIT_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUCoregionalizationFit',
    name: 'four variables, four structures, 256 bins and 1024 sweeps',
    build(context) {
      context.graph.add(new GPUCoregionalizationFit(createCoregionalizationAuditProps(context)));
    }
  }
];

// Above-limit companions are rejected before any kernel is declared.
for (const [property, maximum] of [
  ['variableCount', 4],
  ['structureCount', 4],
  ['maximumIterations', 1024],
  ['distances.length', 256]
] as const) {
  COREGIONALIZATION_FIT_AUDIT_CASES.push({
    exportName: 'GPUCoregionalizationFit',
    name: `${property}-above-maximum`,
    build(context) {
      const props = createCoregionalizationAuditProps(context);
      context.graph.add(
        new GPUCoregionalizationFit({
          ...props,
          ...(property === 'distances.length'
            ? {distances: context.view('float32', maximum + 1)}
            : {[property]: maximum + 1})
        })
      );
    }
  });
}
