// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorEntry, RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
function entry(
  file: string | readonly string[],
  entry: Omit<RasterOperatorEntry, 'sources' | 'guards' | 'datumKinds'>
): RasterOperatorEntry {
  return {
    ...entry,
    guards: [],
    datumKinds: [CELL],
    sources: (typeof file === 'string' ? [file] : file).map(
      source => `modules/analysis/src/raster-analysis/remote-sensing/${source}.ts`
    )
  };
}

/** Explicit bounded remote-sensing models; each entry names its actual executed algorithm. */
export const RASTER_COVERAGE_EXPANSION_OPERATORS: RasterOperatorTable = {
  GPURasterConservativeRemapping: entry('gpu-raster-conservative-remapping', {
    family: 'remapping',
    descriptor: {scope: 'pairwise', reduction: 'weighted-mean'},
    variables: {values: 'measure', coveredArea: 'measure', status: 'category'},
    stages: ['representation', 'support', 'operator'],
    neighbourhood: 'window',
    choices: [
      {stage: 'support', name: 'weightMode', alternatives: ['prepare', 'reuse'], sweepable: false},
      {
        stage: 'operator',
        name: 'normalization',
        alternatives: ['destination-area', 'covered-area', 'extensive'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Exact Cartesian rectangle overlaps weight each valid source value; normalize by destination area, valid covered area, or source area for extensive redistribution.'
    }
  }),
  GPURasterTexture: entry('gpu-raster-texture', {
    family: 'texture',
    descriptor: {scope: 'focal', reduction: 'histogram'},
    variables: {features: 'score', pairCount: 'measure'},
    stages: ['representation', 'neighbourhood', 'operator'],
    neighbourhood: 'window',
    choices: [
      {stage: 'neighbourhood', name: 'radius', sweepable: true},
      {stage: 'operator', name: 'levels', sweepable: true},
      {stage: 'operator', name: 'symmetric', alternatives: [false, true], sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Quantized paired-valid observations form a direction-specific normalized GLCM; reduce contrast, dissimilarity, homogeneity, angular second moment, energy, correlation and natural-log entropy.'
    }
  }),
  GPURasterTemporal: entry('gpu-raster-temporal', {
    family: 'temporal',
    descriptor: {scope: 'focal', reduction: 'solve'},
    variables: {values: 'measure', counts: 'measure', chosenSlice: 'rank', status: 'category'},
    stages: ['representation', 'support', 'operator'],
    neighbourhood: 'window',
    choices: [
      {
        stage: 'operator',
        name: 'operation',
        alternatives: ['mean', 'minimum', 'maximum', 'median', 'savitzky-golay'],
        sweepable: false
      },
      {stage: 'neighbourhood', name: 'windowLength', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composite finite QA-approved observations or fit a local regular-time polynomial with reorthogonalized QR; interpolation edges shift the fitting window and missing observations retain original times.'
    }
  }),
  GPURasterRegistration: entry(['gpu-raster-registration', 'raster-masked-registration'], {
    family: 'registration',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {shift: 'score', peak: 'score', status: 'category', overlap: 'measure'},
    stages: ['representation', 'operator'],
    neighbourhood: 'none',
    choices: [
      {
        stage: 'operator',
        name: 'mode',
        alternatives: ['periodic', 'masked-linear'],
        sweepable: false
      },
      {stage: 'operator', name: 'upsampleFactor', sweepable: true},
      {stage: 'operator', name: 'ambiguityTolerance', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Periodic unmasked phase correlation normalizes reference times conjugate moving spectra, rejecting DC and products at or below max(1e-20, 1e-6 * maximumProductMagnitude). Parallel two-pass argmax reductions choose the lowest cell index on ties; a separable localized inverse DFT refines the inverse-FFT peak. Masked-linear mode uses six zero-padded FFT correlations for paired normalized correlation, then bounded bilinear refinement with explicit overlap requirements. Flat, unsupported and ambiguous inputs retain failure status.'
    }
  }),
  GPURasterRegionGraph: entry(['gpu-raster-region-graph', 'raster-region-tiles'], {
    family: 'regions',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {adjacency: 'measure', weights: 'score', mapping: 'category', status: 'category'},
    stages: ['representation', 'neighbourhood', 'operator', 'iteration'],
    neighbourhood: 'zone',
    choices: [
      {stage: 'operator', name: 'mergeMode', alternatives: ['fixed', 'adaptive'], sweepable: false},
      {stage: 'operator', name: 'mergeThreshold', sweepable: true},
      {stage: 'iteration', name: 'maximumIterations', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Count symmetric four-neighbor region boundaries, weight edges by original finite region mean differences and propagate minimum source IDs along thresholded edges in explicit bounded rounds. Adaptive mode merges the globally closest eligible pair per round and recomputes observation-count-weighted component means. Report another-round-needed status.'
    }
  }),
  GPURasterSpectralUnmixing: entry('gpu-raster-spectral-unmixing', {
    family: 'spectral-unmixing',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {
      abundances: 'score',
      residual: 'measure',
      status: 'category',
      iterationCounts: 'round'
    },
    stages: ['representation', 'support', 'operator', 'iteration'],
    neighbourhood: 'none',
    choices: [
      {
        stage: 'operator',
        name: 'constraint',
        alternatives: ['unconstrained', 'nonnegative', 'simplex'],
        sweepable: false
      },
      {stage: 'iteration', name: 'maximumIterations', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Fit supplied endmember spectra with reorthogonalized QR or projected gradient descent, projecting onto the nonnegative orthant or probability simplex. Keep residual, feasibility-preserving exhausted estimates and convergence status.'
    }
  }),
  GPURasterSeasonalBreak: entry(['gpu-raster-seasonal-break', 'seasonal-break-extended'], {
    family: 'seasonal-break',
    descriptor: {scope: 'solver', reduction: 'argmin'},
    variables: {
      coefficients: 'score',
      components: 'score',
      breakIndex: 'rank',
      improvement: 'score',
      status: 'category',
      diagnostics: 'score'
    },
    stages: ['representation', 'support', 'operator', 'iteration'],
    neighbourhood: 'none',
    choices: [
      {stage: 'operator', name: 'period', sweepable: true},
      {stage: 'operator', name: 'seasonality', sweepable: true},
      {stage: 'operator', name: 'fitMethod', sweepable: true},
      {stage: 'support', name: 'missingPolicy', sweepable: true},
      {stage: 'iteration', name: 'maximumIterations', sweepable: true},
      {stage: 'operator', name: 'monitoring', sweepable: true},
      {stage: 'support', name: 'minimumSegmentLength', sweepable: true},
      {stage: 'operator', name: 'minimumBICGain', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Harmonic or dummy-season trend regression compares eligible single level-shift boundaries using BIC. Joint QR or bounded cyclic decomposition preserves convergence status. Fixed-history residual monitoring requires persistent same-sign threshold crossings; missing-time policies preserve original acquisition slots.'
    }
  })
};
