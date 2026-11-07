// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/** Bounded native GPU interpolation and conditional simulation contracts. */
export const INTERPOLATION_EXPANSION_OPERATORS: SpatialFamilyOperators = {
  GPULocalKriging: {
    family: 'density',
    descriptor: {scope: 'solver', scopeSecondary: 'focal', reduction: 'solve'},
    variables: {
      results: 'score',
      statuses: 'category',
      observations: 'score',
      supports: 'coordinate'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'query', label: 'prediction support', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'mode',
        stage: 'operator',
        alternatives: ['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'],
        sweepable: false
      },
      {name: 'neighborCount', stage: 'neighbourhood', sweepable: true},
      {name: 'supportCount', stage: 'operator', sweepable: false}
    ],
    formula: {
      id: 'gpu-spatial-analysis.local-kriging',
      derivation: 'curated',
      tex: String.raw`\begin{bmatrix}K+E&F\\F^T&0\end{bmatrix}\begin{bmatrix}w\\\lambda\end{bmatrix}=\begin{bmatrix}c\\f\end{bmatrix},\quad \term{results}{(\hat z,\sigma^2)}=(w^Tz, C_{BB}-w^Tc-\lambda^Tf)`,
      ascii:
        '[K+E F; F^T 0][w;lambda]=[c;f]; output.values=[prediction, blockVariance, pivotRatio, neighborCount]. Known-mean modes omit F and predict mean+w*(z-mean). Co-kriging uses a sum of PSD structures B_s=L_s*transpose(L_s), with known or separate unknown means.',
      symbols: [{id: 'results', role: 'output', binding: 'wgsl:results'}]
    },
    metadata: {
      maximumNeighborCount: 32,
      maximumSupportCount: 64,
      maximumDriftCount: 8,
      maximumVariableCount: 4,
      maximumCoregionalizationStructures: 4,
      packedValues: 'prediction, latentVariance, pivotRatio, retainedNeighborCount',
      coordinateModel: 'planar',
      observationErrorSeparateFromNugget: true
    }
  },
  GPULocalRBF: {
    family: 'density',
    descriptor: {scope: 'solver', scopeSecondary: 'focal', reduction: 'solve'},
    variables: {results: 'score', statuses: 'category', values: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'query', label: 'query point', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'kernel',
        stage: 'operator',
        alternatives: [
          'thin-plate-spline',
          'cubic',
          'gaussian',
          'linear',
          'quintic',
          'multiquadric',
          'inverse-multiquadric',
          'inverse-quadratic'
        ],
        sweepable: false
      },
      {name: 'smoothing', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'gpu-spatial-analysis.local-rbf',
      derivation: 'curated',
      tex: String.raw`\begin{bmatrix}K+\lambda I&P\\P^T&0\end{bmatrix}\begin{bmatrix}a\\b\end{bmatrix}=\begin{bmatrix}z\\0\end{bmatrix},\quad\term{results}{s(x)}=K(x,X)a+P(x)b`,
      ascii:
        'Augmented kernel/polynomial solve with nonnegative smoothing, local centered/scaled polynomial, at most 32 neighbors.',
      symbols: [{id: 'results', role: 'output', binding: 'wgsl:results'}]
    },
    metadata: {
      maximumNeighborCount: 32,
      permitsExtrapolation: true,
      packedValues: 'prediction, pivotRatio, retainedNeighborCount, reserved'
    }
  },
  GPUNaturalNeighbor: {
    family: 'density',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {results: 'score', statuses: 'category', weights: 'probability'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'query', label: 'query point', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{name: 'maximumVertexCount', stage: 'neighbourhood', sweepable: false}],
    formula: {
      id: 'gpu-spatial-analysis.natural-neighbor',
      derivation: 'curated',
      tex: String.raw`w_i=\frac{|V_x\cap V_i|}{|V_x|},\quad\term{results}{s(x)}=\sum_iw_i z_i`,
      ascii:
        'Sibson weights are stolen Voronoi areas; outside hull is invalid, hull edges use limiting linear interpolation.',
      symbols: [{id: 'results', role: 'output', binding: 'wgsl:results'}]
    },
    metadata: {
      maximumDirectSampleCount: 64,
      maximumIndexedSampleCount: 16777216,
      maximumCavityNeighborCount: 64,
      maximumVertexCount: 64,
      permitsExtrapolation: false,
      packedValues: 'prediction, queryVoronoiArea, weightSum, contributingCount'
    }
  },
  GPUConditionalGaussianSimulation: {
    family: 'density',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {values: 'score', moments: 'score', statuses: 'category'},
    stages: ['operator', 'inference'],
    datumKinds: [{kind: 'replicate', label: 'joint field realization', indexSpace: 'batch'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'seed', stage: 'inference', sweepable: true},
      {name: 'replicateCount', stage: 'inference', sweepable: true}
    ],
    formula: {
      id: 'gpu-spatial-analysis.conditional-gaussian-simulation',
      derivation: 'curated',
      tex: String.raw`\term{values}{Z_*\mid z}=m_*+C_{*o}(C_{oo}+E)^{-1}(z-m_o)+L_c\epsilon`,
      ascii:
        'Known-mean exact bounded Gaussian conditioning; Lc*transpose(Lc) is joint conditional covariance. Philox supplies independent standard normals.',
      symbols: [{id: 'values', role: 'output', binding: 'wgsl:values'}]
    },
    metadata: {
      maximumObservationCount: 128,
      maximumQueryCount: 128,
      independentReplicateStreams: true
    }
  }
};
