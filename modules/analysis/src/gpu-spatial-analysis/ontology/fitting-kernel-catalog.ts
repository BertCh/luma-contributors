// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisOperator, GPUSpatialAnalysisCapability} from './spatial-analysis-ontology';

const SOURCE = 'modules/analysis/src/analysis-core/kernels/';
const TEST = 'modules/analysis/test/analysis-core/fitting/fitting-kernels.spec.ts';

/** Shared fitting support, below every domain entry point. */
export const FITTING_KERNEL_CAPABILITY: GPUSpatialAnalysisCapability = {
  id: 'model-fitting-kernels',
  title: 'Model fitting kernels',
  status: 'available',
  domain: 'spatial-models',
  stage: 'infer',
  summary:
    'Fixed-order gradients, budgeted Adam steps, seeded row orders and spatial coordinate encodings.',
  inputs: ['attributes', 'points', 'parameters'],
  outputs: ['attributes', 'summary'],
  coordinateModels: ['none', 'planar', 'spherical'],
  outputBound: 'fixed-sized',
  operators: [
    'GPUFittingGradientSums',
    'GPUAdamStep',
    'GPUFittingMinibatchOrder',
    'GPUFittingCoordinateEncoder'
  ]
};

/** LRN-02 kernels: algorithm references and precision conventions are recorded in the linked page. */
export const FITTING_KERNEL_OPERATORS: readonly GPUAnalysisOperator[] = (
  [
    {
      exportName: 'GPUFittingGradientSums',
      summary: 'Fixed-order row tiles followed by a binary tree of gradient partials.',
      coordinateModels: ['none'],
      limits: [
        {
          name: 'parameterCount',
          maximum: 100_000,
          reason: 'Tier-zero fitted model parameter budget.',
          source: `${SOURCE}gpu-fitting-gradient-sums.ts`
        }
      ]
    },
    {
      exportName: 'GPUAdamStep',
      summary:
        'Bias-corrected Adam with caller-owned moments, loss history, iteration budget and loss plateau status.',
      coordinateModels: ['none'],
      limits: [
        {
          name: 'maximumIterations',
          maximum: 10_000,
          reason: 'Bounds the loss history and float32 iteration powers.',
          source: `${SOURCE}gpu-adam-step.ts`
        }
      ],
      status: ['converged', 'iterationLimitReached', 'invalidCount']
    },
    {
      exportName: 'GPUFittingMinibatchOrder',
      summary:
        'Repeatable Philox random-key row permutation using the published stable radix sort.',
      coordinateModels: ['none'],
      limits: [
        {
          name: 'rowCount',
          maximum: 2 ** 24,
          reason: 'Counts remain exact float32 integers for downstream gradient normalization.',
          source: `${SOURCE}gpu-fitting-minibatch-order.ts`
        }
      ]
    },
    {
      exportName: 'GPUFittingCoordinateEncoder',
      summary:
        'Caller-supplied Fourier frequencies or normalized real spherical harmonics through degree 39.',
      coordinateModels: ['planar', 'spherical'],
      limits: [
        {
          name: 'maximumDegree',
          maximum: 39,
          reason: 'Bounds per-coordinate recurrence work and supports the 1600-value L=40 basis.',
          source: `${SOURCE}gpu-fitting-coordinate-encoder.ts`
        }
      ]
    }
  ] satisfies readonly Partial<GPUAnalysisOperator>[]
).map(
  (detail): GPUAnalysisOperator => ({
    primaryCapability: FITTING_KERNEL_CAPABILITY.id,
    tags: [],
    package: '@luma-contributors/analysis/analysis-core',
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    dimensionality: 'mixed',
    topology: 'unspecified',
    cardinality: 'fixed-sized',
    semantics: 'unspecified',
    deterministic: true,
    status: [],
    readback: 'none',
    role: 'helper',
    tier: 'scale',
    evidence: {
      status: 'partial',
      tests: TEST,
      documentation: 'modules/analysis/docs/fitting-kernels.md',
      contract: 'GPUCommandNodeProducer; caller-supplied fitting state; no bundled weights'
    },
    ...detail
  })
);
