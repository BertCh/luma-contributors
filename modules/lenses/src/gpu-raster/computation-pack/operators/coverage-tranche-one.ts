// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorEntry, RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
const ZONE = {kind: 'zone', label: 'source zone', indexSpace: '1d'} as const;
const SOURCE_ROOT = 'modules/analysis/src/gpu-raster';

function makeSpectralEntry(
  name: string,
  formula: string,
  description: string
): RasterOperatorEntry {
  return {
    family: 'spectral-indices',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {values: 'score', validity: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [{stage: 'operator', name: 'epsilon', sweepable: true}],
    formula: {
      id: `gpu-raster.${name}.rule`,
      derivation: 'curated',
      tex: formula,
      ascii: description,
      symbols: []
    },
    sources: [`${SOURCE_ROOT}/gpu-raster-spectral-indices.ts`]
  };
}

/** Bounded coverage summaries, paired focal statistics and calibrated spectral recipes. */
export const RASTER_COVERAGE_TRANCHE_ONE_OPERATORS: RasterOperatorTable = {
  GPURasterFractionalZonalDistribution: {
    family: 'fractional-zones',
    descriptor: {scope: 'zonal', reduction: 'scan-sort'},
    variables: {
      quantiles: 'measure',
      populationVariances: 'measure',
      sampleVariances: 'measure',
      effectiveSampleSizes: 'measure',
      coveredArea: 'measure',
      validArea: 'measure',
      validity: 'category'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [ZONE],
    neighbourhood: 'zone',
    guards: [],
    choices: [
      {
        stage: 'operator',
        name: 'quantileMethod',
        alternatives: ['exactextract', 'inverse-cdf'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Coverage times independent observation-weight centered moments retain distinct cell support; quantiles coalesce equal values and interpolate the exactextract 0.3.0 weighted plotting positions or select the inverse weighted CDF. Cell areas remain independent. Incomplete coverage and undefined support retain explicit status.'
    },
    sources: [`${SOURCE_ROOT}/gpu-raster-fractional-zonal-distribution.ts`]
  },
  GPURasterFractionalCategoricalStatistics: {
    family: 'fractional-zones',
    descriptor: {scope: 'zonal', reduction: 'histogram'},
    variables: {
      coverageSums: 'measure',
      classAreas: 'measure',
      proportions: 'fraction',
      majority: 'category',
      minority: 'category',
      missingArea: 'measure',
      validity: 'category'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [ZONE],
    neighbourhood: 'zone',
    guards: [],
    choices: [],
    formula: {
      id: 'gpu-raster.fractional-categorical-statistics.rule',
      derivation: 'curated',
      tex: String.raw`A_{zk}=|\det A|\sum_{c:C(c)=k}f_{zc},\quad p_{zk}=A_{zk}/\sum_j A_{zj}`,
      ascii:
        'Sum coverage fractions by zone and valid category, multiply by affine cell area, and normalize by valid category area. Missing area is separate; majority ties choose the largest category and minority ties the smallest present category.',
      symbols: []
    },
    sources: [`${SOURCE_ROOT}/gpu-raster-fractional-categorical-statistics.ts`]
  },
  GPURasterFocalPairs: {
    family: 'focal-pairs',
    descriptor: {scope: 'focal', reduction: 'solve'},
    variables: {
      pairedCount: 'measure',
      covariance: 'score',
      correlation: 'score',
      slope: 'score',
      intercept: 'score',
      rSquared: 'fraction',
      residualSumSquares: 'measure',
      status: 'category'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'radius', sweepable: true},
      {
        stage: 'operator',
        name: 'covarianceNormalization',
        alternatives: ['sample', 'population'],
        sweepable: false
      }
    ],
    formula: {
      id: 'gpu-raster.focal-pairs.rule',
      derivation: 'curated',
      tex: String.raw`b=S_{xy}/S_{xx},\quad a=\bar y-b\bar x,\quad r=S_{xy}/\sqrt{S_{xx}S_{yy}}`,
      ascii:
        'Centered moments use paired-valid observations in an explicit square window. Covariance uses the selected population or sample divisor; regression fits response = intercept + slope * predictor. Degenerate moments and insufficient support remain explicit.',
      symbols: []
    },
    sources: [`${SOURCE_ROOT}/focal-pairs/gpu-raster-focal-pairs.ts`]
  },
  GPURasterNDWI: makeSpectralEntry(
    'NDWI',
    String.raw`\mathrm{NDWI}_{Gao}=\frac{NIR-SWIR}{NIR+SWIR},\quad \mathrm{NDWI}_{McFeeters}=\frac{green-NIR}{green+NIR}`,
    'The explicit Gao or McFeeters method selects the calibrated band pair; invalid inputs and unstable denominators remain masked. No automatic wavelength inference or clamping.'
  ),
  GPURasterNBR: makeSpectralEntry(
    'NBR',
    String.raw`\mathrm{NBR}=\frac{NIR-SWIR2}{NIR+SWIR2}`,
    'Normalized burn ratio uses calibrated near-infrared and SWIR2 reflectance; preserve input masks and denominator validity.'
  ),
  GPURasterEVI: makeSpectralEntry(
    'EVI',
    String.raw`\mathrm{EVI}=2.5\frac{NIR-red}{NIR+6red-7.5blue+1}`,
    'Enhanced vegetation index uses gain 2.5, red coefficient 6, blue coefficient 7.5 and offset 1 on calibrated reflectance. Digital-number scaling precedes these constants; input and denominator masks remain explicit.'
  ),
  GPURasterSAVI: makeSpectralEntry(
    'SAVI',
    String.raw`\mathrm{SAVI}=(1+L)\frac{NIR-red}{NIR+red+L}`,
    'Soil-adjusted vegetation index applies the supplied soil adjustment to calibrated reflectance; invalid bands and unstable denominators remain masked.'
  )
};
