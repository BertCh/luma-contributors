// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_SPATIAL_ANALYSIS_CAPABILITIES} from '../spatial-analysis-catalog';
import {GPU_SPATIAL_ANALYSIS_OPERATORS} from '../spatial-analysis-operators';
import type {
  GPUAnalysisOperator,
  GPUAnalysisOperatorIdentity,
  GPUAnalysisCapability,
  GPUAnalysisOperatorQuery,
  GPUSpatialAnalysisCapability
} from '../spatial-analysis-ontology';
import {FITTING_KERNEL_CAPABILITY, FITTING_KERNEL_OPERATORS} from '../fitting-kernel-catalog';
import {ENTRY_POINT_CAPABILITIES} from './entry-point-capabilities';
import {ENTRY_POINT_OPERATOR_DETAILS} from './entry-point-operator-details';
import {OPERATOR_ANNOTATIONS} from './operator-annotations';
import {SPATIOTEMPORAL_REVIEW_OPERATORS} from '../spatiotemporal-review-catalog';

import {GPU_ANALYSIS_OPERATOR_KIND_RECORDS} from '../analysis-experience/analysis-operator-kind-records';

import {REUSED_ANALYSIS_OPERATORS} from '../reused-analysis-catalog';
import {freezeCatalog} from '../freeze-catalog';

const DOCUMENTATION_FALLBACK = 'docs/api/analysis-catalog.md';

function makeEntryPointOperator(
  capability: GPUSpatialAnalysisCapability,
  exportName: string
): GPUAnalysisOperator {
  const detail = ENTRY_POINT_OPERATOR_DETAILS[exportName];
  if (!detail) throw new Error(`${exportName} has no entry in ENTRY_POINT_OPERATOR_DETAILS`);
  const kinds = GPU_ANALYSIS_OPERATOR_KIND_RECORDS[exportName];
  if (!kinds) throw new Error(`${exportName} has no per-operator kind record`);
  const annotation = OPERATOR_ANNOTATIONS[exportName] ?? {};
  return {
    exportName,
    primaryCapability: capability.id,
    tags: [],
    package: detail.package,
    // Family kinds are a union of contracts, never the ports of an individual operator.
    inputs: kinds.inputs,
    outputs: kinds.outputs,
    dimensionality: detail.dimensionality ?? 'unspecified',
    coordinateModels: capability.coordinateModels,
    topology: 'unspecified',
    cardinality: capability.outputBound,
    semantics: 'unspecified',
    deterministic: 'unspecified',
    status: detail.status ?? 'unknown',
    readback: detail.readback ?? 'unknown',
    summary: detail.summary,
    role: annotation.role ?? 'operator',
    tier: annotation.tier ?? 'unknown',
    ...(annotation.limits ? {limits: annotation.limits} : {}),
    ...(annotation.limitation ? {limitation: annotation.limitation} : {}),
    evidence: {
      status: 'partial',
      tests: detail.tests,
      ...(detail.additionalTests ? {additionalTests: detail.additionalTests} : {}),
      documentation: detail.documentation || DOCUMENTATION_FALLBACK,
      contract: 'unaudited'
    }
  };
}

/** Every capability family of every entry point: the original families followed by the new ones. */
const CAPABILITY_FAMILIES: readonly GPUSpatialAnalysisCapability[] = [
  ...GPU_SPATIAL_ANALYSIS_CAPABILITIES,
  ...ENTRY_POINT_CAPABILITIES,
  FITTING_KERNEL_CAPABILITY
];

const ALL_OPERATORS: readonly GPUAnalysisOperator[] = freezeCatalog(
  (
    [
      ...GPU_SPATIAL_ANALYSIS_OPERATORS,
      ...REUSED_ANALYSIS_OPERATORS,
      ...ENTRY_POINT_CAPABILITIES.flatMap(capability =>
        capability.operators.map(exportName => makeEntryPointOperator(capability, exportName))
      ),
      // Operators of `gpu-spatial-analysis` added after the original 261; their families are original.
      ...SPATIOTEMPORAL_REVIEW_OPERATORS,
      ...FITTING_KERNEL_OPERATORS,
      {
        ...GPU_SPATIAL_ANALYSIS_OPERATORS.find(
          operator => operator.exportName === 'addVisibilityGraphRecipe'
        )!,
        package: '@luma-contributors/analysis/gpu-network',
        primaryCapability: 'network-structure',
        summary:
          'Visibility adjacency with budgeted exact or deterministic sampled CSR network metrics.',
        inputs: ['points', 'lines', 'polygons', 'parameters'],
        outputs: ['network', 'attributes', 'summary'],
        status: [
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow',
          'invalidCount',
          'approximation',
          'converged',
          'iterationLimitReached',
          'zeroWeightCycle'
        ],
        limitation:
          'CSR metrics use budgeted exact all-source calculations or deterministic sampled estimates. Auto mode samples when the exact command budget cannot fit; explicit exact mode rejects an insufficient budget. Sampled maximum depth is a lower bound.',
        evidence: {
          status: 'partial',
          tests:
            'modules/analysis/test/gpu-spatial-analysis/urban-ecology/visibility-graph-metrics.spec.ts',
          documentation: 'docs/api/gpu-network/visibility-graph.md',
          contract: 'GPUVisibilityGraphCSRProps | GPUVisibilityGraphProps'
        }
      }
    ] satisfies GPUAnalysisOperator[]
  ).map(operator => {
    // Every public assembly path passes through annotation application, including fitting helpers.
    const annotation =
      OPERATOR_ANNOTATIONS[`${operator.package}:${operator.exportName}`] ??
      OPERATOR_ANNOTATIONS[operator.exportName];
    if (!annotation) return operator;
    return {...operator, ...annotation};
  })
);

const ALL_CAPABILITIES: readonly GPUAnalysisCapability[] = freezeCatalog(
  CAPABILITY_FAMILIES.map(capability => ({
    ...capability,
    operators: ALL_OPERATORS.filter(operator => operator.primaryCapability === capability.id).map(
      operator => operator.exportName
    ),
    operatorIdentities: ALL_OPERATORS.filter(
      operator => operator.primaryCapability === capability.id
    ).map(({package: packageName, exportName}) => ({package: packageName, exportName}))
  }))
);

const OPERATORS_BY_NAME = new Map<string, GPUAnalysisOperator[]>();
const OPERATOR_BY_IDENTITY = new Map<string, GPUAnalysisOperator>();
for (const operator of ALL_OPERATORS) {
  const key = `${operator.package}:${operator.exportName}`;
  if (OPERATOR_BY_IDENTITY.has(key)) throw new Error(`Duplicate catalog identity ${key}`);
  OPERATOR_BY_IDENTITY.set(key, operator);
  const named = OPERATORS_BY_NAME.get(operator.exportName) ?? [];
  named.push(operator);
  OPERATORS_BY_NAME.set(operator.exportName, named);
}

/**
 * Capability families of all nine entry points. `GPU_SPATIAL_ANALYSIS_CAPABILITIES` keeps
 * returning the original `gpu-spatial-analysis` families only.
 */
export function getGPUAnalysisCapabilityCatalog(): readonly GPUAnalysisCapability[] {
  return ALL_CAPABILITIES;
}

/**
 * Every contributor class and recipe of all nine entry points, with `tier` and `limits`.
 * `GPU_SPATIAL_ANALYSIS_OPERATORS` keeps returning the original `gpu-spatial-analysis` operators only.
 */
export function getGPUAnalysisOperatorCatalog(): readonly GPUAnalysisOperator[] {
  return ALL_OPERATORS;
}

/** Returns one frozen descriptor. Bare names must be unique; qualify ambiguous names by package. */
export function getGPUAnalysisOperator(
  identity: string | GPUAnalysisOperatorIdentity
): GPUAnalysisOperator | undefined {
  if (typeof identity !== 'string')
    return OPERATOR_BY_IDENTITY.get(`${identity.package}:${identity.exportName}`);
  const named = OPERATORS_BY_NAME.get(identity);
  if (named && named.length > 1) {
    throw new Error(
      `getGPUAnalysisOperator: ambiguous exportName "${identity}"; specify package (${named.map(operator => operator.package).join(', ')})`
    );
  }
  return named?.[0];
}

/**
 * Finds operators of any entry point by package, tier, task, data kind or free text. Helpers
 * (`role: 'helper'`) are left out unless `includeHelpers` is true.
 */
export function queryGPUAnalysisOperators(
  query: GPUAnalysisOperatorQuery = {}
): readonly GPUAnalysisOperator[] {
  const search = query.search?.trim().toLowerCase();
  return ALL_OPERATORS.filter(operator => {
    if (!query.includeHelpers && operator.role === 'helper') return false;
    if (query.package && operator.package !== query.package) return false;
    if (query.tier && operator.tier !== query.tier) return false;
    if (query.readback && operator.readback !== query.readback) return false;
    if (query.dimensionality && operator.dimensionality !== query.dimensionality) return false;
    if (query.primaryCapability && operator.primaryCapability !== query.primaryCapability)
      return false;
    if (query.tag && !operator.tags.includes(query.tag)) return false;
    if (query.input && !operator.inputs.includes(query.input)) return false;
    if (query.output && !operator.outputs.includes(query.output)) return false;
    if (query.coordinateModel && !operator.coordinateModels.includes(query.coordinateModel))
      return false;
    if (query.status) {
      if (query.status === 'unknown') {
        if (operator.status !== 'unknown') return false;
      } else if (operator.status === 'unknown' || !operator.status.includes(query.status))
        return false;
    }
    if (query.semantics && operator.semantics !== query.semantics) return false;
    if (query.evidence && operator.evidence.status !== query.evidence) return false;
    if (!search) return true;
    return [
      operator.exportName,
      operator.primaryCapability,
      operator.summary ?? '',
      ...operator.tags,
      ...operator.inputs,
      ...operator.outputs,
      operator.evidence.contract
    ]
      .join(' ')
      .toLowerCase()
      .includes(search);
  });
}
