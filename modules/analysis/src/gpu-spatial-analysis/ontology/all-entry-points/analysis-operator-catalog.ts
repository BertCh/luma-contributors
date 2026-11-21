// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_SPATIAL_ANALYSIS_CAPABILITIES} from '../spatial-analysis-catalog';
import {GPU_SPATIAL_ANALYSIS_OPERATOR_DESCRIPTORS} from '../spatial-analysis-operators';
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
import {resolveOperatorDescriptor} from './resolve-operator-descriptor';
import type {OperatorDescriptor} from './resolve-operator-descriptor';
import {SPATIOTEMPORAL_REVIEW_OPERATORS} from '../spatiotemporal-review-catalog';

import {GPU_ANALYSIS_OPERATOR_KIND_RECORDS} from '../analysis-experience/analysis-operator-kind-records';

import {REUSED_ANALYSIS_OPERATORS} from '../reused-analysis-catalog';
import {freezeCatalog} from '../freeze-catalog';

import {BUILDING_TESSELLATION_OPERATORS} from './building-tessellation-catalog';
import {SPATIAL_OPERATION_FORMS} from './spatial-operation-forms';
import {OPERATOR_OUTPUT_MODES} from './operator-output-declarations';
import {OPERATOR_ROUTES} from './operator-routes';
import {OPERATOR_ROUTE_DETAILS} from './operator-route-details';
import {
  OPERATOR_FORM_RELATIONSHIPS,
  PREPARED_TILED_FORM_RELATIONSHIPS
} from './operator-form-relationships';
import {RECIPE_OPERATION_FORMS} from './recipe-operation-forms';
import {OPERATION_FORM_CONTRACTS} from './operation-form-contracts';
import {getGPUAnalysisOperatorQuestions} from '../analysis-experience/analysis-operator-questions';
import type {GPUAnalysisQuestion} from '../analysis-experience/analysis-experience-types';
import {GPU_ANALYSIS_QUESTIONS} from '../analysis-experience/analysis-experience-types';
export {describeOutputs} from './operator-output-declarations';

const DOCUMENTATION_FALLBACK = 'docs/api/analysis-catalog.md';
// Registry membership is independent of evaluating a source-owned planner's metadata.
const ROUTED_OPERATOR_IDENTITIES = [
  ...Object.entries(OPERATOR_ROUTES),
  ...Object.entries(OPERATOR_ROUTE_DETAILS)
].map(([identity]) => identity);

function makeEntryPointOperator(
  capability: GPUSpatialAnalysisCapability,
  exportName: string
): OperatorDescriptor {
  const detail = ENTRY_POINT_OPERATOR_DETAILS[exportName];
  if (!detail) throw new Error(`${exportName} has no entry in ENTRY_POINT_OPERATOR_DETAILS`);
  const kinds = GPU_ANALYSIS_OPERATOR_KIND_RECORDS[exportName];
  if (!kinds) throw new Error(`${exportName} has no per-operator kind record`);
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
    status: detail.status,
    readback: detail.readback ?? 'unknown',
    summary: detail.summary,
    role: 'operator',
    tier: 'unknown',
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
      ...GPU_SPATIAL_ANALYSIS_OPERATOR_DESCRIPTORS,
      ...REUSED_ANALYSIS_OPERATORS,
      ...ENTRY_POINT_CAPABILITIES.flatMap(capability =>
        capability.operators.map(exportName => makeEntryPointOperator(capability, exportName))
      ),
      // Operators of `gpu-spatial-analysis` added after the original 261; their families are original.
      ...SPATIOTEMPORAL_REVIEW_OPERATORS,
      ...FITTING_KERNEL_OPERATORS,
      ...BUILDING_TESSELLATION_OPERATORS,
      {
        ...GPU_SPATIAL_ANALYSIS_OPERATOR_DESCRIPTORS.find(
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
          contract: 'GPUNetworkVisibilityGraphCSRProps | GPUVisibilityGraphProps'
        }
      }
    ] satisfies OperatorDescriptor[]
  ).map(descriptor => {
    const operator = resolveOperatorDescriptor(descriptor);
    const key = `${operator.package}:${operator.exportName}`;
    const baseForms = SPATIAL_OPERATION_FORMS[key];
    const relationships =
      PREPARED_TILED_FORM_RELATIONSHIPS[key] ?? OPERATOR_FORM_RELATIONSHIPS[key];
    const recipeForms = RECIPE_OPERATION_FORMS[key];
    const declaredForms =
      baseForms || relationships || recipeForms
        ? {...baseForms, ...relationships, ...recipeForms}
        : undefined;
    const withContract = <Form extends GPUAnalysisOperatorIdentity>(form: Form) => ({
      ...form,
      contract: OPERATION_FORM_CONTRACTS[`${form.package}:${form.exportName}`]
    });
    const forms = declaredForms
      ? {
          ...declaredForms,
          ...(declaredForms.cpu ? {cpu: withContract(declaredForms.cpu)} : {}),
          ...(declaredForms.gpu ? {gpu: withContract(declaredForms.gpu)} : {}),
          ...(declaredForms.arrays ? {arrays: declaredForms.arrays.map(withContract)} : {}),
          ...(declaredForms.variants ? {variants: declaredForms.variants.map(withContract)} : {})
        }
      : undefined;
    const outputDeclarations = OPERATOR_OUTPUT_MODES[`${operator.package}:${operator.exportName}`];
    const routeDetails = OPERATOR_ROUTE_DETAILS[key];
    const routes =
      OPERATOR_ROUTES[key] || routeDetails
        ? [...(OPERATOR_ROUTES[key] ?? []), ...(routeDetails ?? [])]
        : undefined;
    return forms || outputDeclarations || ROUTED_OPERATOR_IDENTITIES.includes(key)
      ? {
          ...operator,
          ...(forms ? {forms} : {}),
          ...(routes ? {routes} : {}),
          ...(outputDeclarations ? {outputDeclarations} : {}),
          // Enrichment never changes identity, even when a static reader cannot evaluate
          // a source-owned planner descriptor or optional metadata branch.
          package: operator.package,
          exportName: operator.exportName
        }
      : operator;
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
 * Unaudited status is `unknown`; `[]` requires an affirmative audit of the qualified identity.
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

/** A task result retains its owner; upstream entries never join the analysis export census. */
export type GPUAnalysisTaskResult = Readonly<{
  questions: readonly GPUAnalysisQuestion[];
  score: number;
}> &
  (
    | Readonly<{owner: 'analysis'; identity: GPUAnalysisOperatorIdentity}>
    | Readonly<{owner: 'upstream'; identity: Readonly<{package: string; exportName: string}>}>
  );

const TASK_ALIASES: Readonly<Record<string, readonly string[]>> = freezeCatalog({
  GPURasterEmbeddingNearestPixels: ['find similar pixels', 'similar pixels', 'nearest pixels'],
  GPUPettittTest: ['change point', 'find a change point', 'single change point'],
  GPULocationChangePoints: ['change point', 'find a change point', 'multiple change points'],
  GPURasterNDVI: ['NDVI', 'normalized difference vegetation index'],
  GPURasterFillHoles: ['fillholes', 'fill holes']
});

function tokenizeTask(text: string): string[] {
  return text
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z])([A-Z][a-z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

/** Task discovery with approved GIS aliases, token matching and a question facet. */
export function searchGPUAnalysisTasks(
  task: string,
  options: {question?: GPUAnalysisQuestion} = {}
): readonly GPUAnalysisTaskResult[] {
  if (options.question !== undefined && !GPU_ANALYSIS_QUESTIONS.includes(options.question)) {
    throw new Error(
      `searchGPUAnalysisTasks question must be ${GPU_ANALYSIS_QUESTIONS.join(', ')}; received ${options.question}`
    );
  }
  const tokens = tokenizeTask(task);
  if (tokens.length === 0) return [];
  const normalized = tokens.join(' ');
  const results: GPUAnalysisTaskResult[] = [];
  const candidates = [
    ...queryGPUAnalysisOperators().map(operator => ({
      identity: {package: operator.package, exportName: operator.exportName},
      owner: 'analysis' as const,
      questions: getGPUAnalysisOperatorQuestions(operator),
      text: [
        operator.exportName,
        operator.summary ?? '',
        operator.primaryCapability,
        ...operator.tags
      ].join(' ')
    })),
    {
      identity: {package: '@luma.gl/graph/gpu-raster', exportName: 'GPURasterNDVI'},
      owner: 'upstream' as const,
      questions: ['what-is-at-each-place'] as readonly GPUAnalysisQuestion[],
      text: 'GPURasterNDVI NDVI normalized difference vegetation index'
    }
  ];
  for (const candidate of candidates) {
    if (options.question && !candidate.questions.includes(options.question)) continue;
    const aliases = TASK_ALIASES[candidate.identity.exportName] ?? [];
    const normalizedName = tokenizeTask(candidate.identity.exportName).join(' ');
    const compactName = normalizedName.replace(/ /g, '').replace(/^gpu/, '');
    const exact =
      normalizedName === normalized ||
      compactName === normalized.replace(/ /g, '') ||
      aliases.some(alias => tokenizeTask(alias).join(' ') === normalized);
    const indexed = new Set(tokenizeTask([candidate.text, ...aliases].join(' ')));
    if (!exact && !tokens.every(token => indexed.has(token))) continue;
    const {text: _text, ...result} = candidate;
    results.push({...result, questions: candidate.questions, score: exact ? 100 : 10});
  }
  return freezeCatalog(
    results.sort(
      (left, right) =>
        right.score - left.score ||
        left.identity.exportName.localeCompare(right.identity.exportName) ||
        left.identity.package.localeCompare(right.identity.package)
    )
  );
}
