// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUSpatialAnalysisCapability,
  GPUSpatialAnalysisOperator
} from './spatial-analysis-ontology';

const TEST_ROOT = 'modules/analysis/test/gpu-spatial-analysis';
const DOCUMENTATION_ROOT = 'docs/api/gpu-spatial-analysis';

/** Discovery keeps bounded numerical contracts separate from upstream API parity. */
export const KEPLER_ANALYSIS_CAPABILITIES: readonly GPUSpatialAnalysisCapability[] = [
  {
    id: 'quantile-local-association',
    title: 'Quantile local association',
    status: 'available',
    domain: 'spatial-statistics',
    stage: 'infer',
    summary:
      'Univariate or multivariate type-5 quantile intersection and conditional local join-count permutations.',
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'source-sized',
    operators: ['GPUQuantileLISA'],
    limitation:
      'Up to 64 variables with one selected class each; empty intersections stay empty; lower-inclusive boundaries, islands and missing observations retain explicit labels. Uses legacy directed pseudo p-values, which can be anti-conservative; no implicit multiple-testing correction.',
    evidence: {
      representativeSourceRows: 8,
      transientMemory: 'linear-source',
      dispatch: 'multi-pass',
      correctnessOracle: `${TEST_ROOT}/recipes/quantile-lisa.spec.ts`
    }
  },
  {
    id: 'dorling-cartogram',
    title: 'Dorling circle cartogram',
    status: 'available',
    domain: 'geometry',
    stage: 'transform',
    summary:
      'Bounded planar circle relaxation with stable source identities and convergence diagnostics.',
    inputs: ['points', 'attributes'],
    outputs: ['points', 'attributes', 'summary'],
    coordinateModels: ['planar'],
    outputBound: 'source-sized',
    operators: ['GPUDorlingCartogram'],
    limitation:
      'Heuristic layout, indexed rows device-binding bound up to 1,073,741,822 (neighborGrid auto from 32,768); all-pairs retains 65,536 rows. At most 256 iterations; neither route guarantees overlap removal or GeoDa coordinate parity.',
    evidence: {
      representativeSourceRows: 4,
      transientMemory: 'linear-source',
      dispatch: 'bounded-iterative',
      correctnessOracle: `${TEST_ROOT}/dorling-cartogram/gpu-dorling-cartogram.spec.ts`
    }
  },
  {
    id: 'rate-mapping',
    title: 'Excess risk and spatial rates',
    status: 'available',
    domain: 'workflow',
    stage: 'orchestrate',
    summary: 'Pooled expected-event excess risk and binary-neighborhood event/population rates.',
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'source-sized',
    operators: ['addExcessRiskRecipe', 'addSpatialRateRecipe'],
    limitation:
      'Binary CSR adjacency ignores numeric edge weights; islands are undefined by default with an explicit own-rate opt-in. These descriptive rates do not establish significance.',
    evidence: {
      representativeSourceRows: 4,
      transientMemory: 'linear-source',
      dispatch: 'multi-pass',
      correctnessOracle: `${TEST_ROOT}/rate-smoothing/rate-mapping-recipes.spec.ts`
    }
  }
];

type OperatorOverride = Partial<
  Pick<
    GPUSpatialAnalysisOperator,
    | 'semantics'
    | 'status'
    | 'deterministic'
    | 'topology'
    | 'cardinality'
    | 'dimensionality'
    | 'coordinateModels'
  >
> & {tests: string; documentation: string; contract: string; evidence: 'partial'};

export const KEPLER_ANALYSIS_OPERATOR_OVERRIDES: Readonly<Record<string, OperatorOverride>> = {
  GPUQuantileLISA: {
    semantics: 'stochastic',
    deterministic: true,
    topology: 'fixed',
    cardinality: 'source-sized',
    dimensionality: 'topology',
    coordinateModels: ['topological'],
    status: ['invalidCount', 'overflow'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/recipes/quantile-lisa.spec.ts`,
    documentation: 'docs/api/gpu-spatial-analysis/quantile-lisa.md',
    contract: 'GPUQuantileLISAProps'
  },
  GPUDorlingCartogram: {
    semantics: 'approximate',
    deterministic: true,
    topology: 'fixed',
    cardinality: 'source-sized',
    dimensionality: '2d',
    status: ['converged', 'iterationLimitReached', 'invalidCount'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/dorling-cartogram/gpu-dorling-cartogram.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/gpu-dorling-cartogram.md`,
    contract: 'GPUDorlingCartogramProps'
  },
  addExcessRiskRecipe: {
    semantics: 'model-based',
    deterministic: true,
    topology: 'fixed',
    cardinality: 'source-sized',
    dimensionality: '2d',
    status: [],
    evidence: 'partial',
    tests: `${TEST_ROOT}/rate-smoothing/rate-mapping-recipes.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/rate-mapping.md`,
    contract: 'GPUExcessRiskRecipeProps'
  },
  addSpatialRateRecipe: {
    semantics: 'model-based',
    deterministic: true,
    topology: 'fixed',
    cardinality: 'source-sized',
    dimensionality: '2d',
    status: ['overflow', 'requiredCount', 'candidateOverflow'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/rate-smoothing/rate-mapping-recipes.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/rate-mapping.md`,
    contract: 'GPUSpatialRateRecipeProps'
  }
};
