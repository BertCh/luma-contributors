// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SpectralEvidenceOptions} from './spectral-evidence.compute';

export default defineScene<SpectralEvidenceOptions>({
  id: 'spectral-evidence',
  title: 'What can six bands distinguish?',
  chapter: 'raster',
  order: 12,
  summary:
    'Compare six calibrated Sentinel-2 indices and fit measured spectral exemplars to each pixel. Indices emphasize different contrasts; mixture weights depend on the chosen model and are not validated material fractions.',
  contributors: [
    'GPURasterNDVI',
    'GPURasterNBR',
    'GPURasterEVI',
    'GPURasterSAVI',
    'GPURasterNDWI',
    'GPURasterSpectralUnmixing',
    'GPUGroupStatistics',
    'GPUChangeDetection'
  ],
  datasets: [
    {
      id: 'dixie-spectral',
      role: 'Six calibrated Sentinel-2 bands, strict per-date validity and measured model exemplars'
    }
  ],
  initialView: {longitude: -120.993, latitude: 40.193, zoom: 13.3},
  basemap: ground('relief'),
  annotations: [
    {kind: 'area', coordinate: [-121.02, 40.2], text: 'Dixie Fire landscape', size: 'small'}
  ],
  furniture: {
    title: {
      title: 'What can six bands distinguish?',
      subtitle: 'Sentinel-2 L2A · 60 m UTM reflectance grid · July and September 2021'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data 2021',
    caveat:
      'July 13 is the fire ignition date. Indices are spectral proxies, not validated burn severity. The exemplar names describe measured spectra only; fitted weights are model coefficients, not ground-truth land-cover percentages. Strict SCL screening is not manual cloud validation.'
  },
  options: [
    {
      kind: 'select',
      id: 'date',
      label: 'Acquisition',
      group: 'Observation',
      apply: 'param',
      default: 'after',
      help: 'Select the July 13 ignition-day or September 21 post-ignition image. Each scene uses unitless reflectance averaged to a 60 m UTM grid.',
      options: [
        {value: 'before', label: '13 Jul 2021 · ignition day'},
        {value: 'after', label: '21 Sep 2021 · post-ignition'}
      ]
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Map product',
      group: 'Spectral evidence',
      apply: 'param',
      default: 'ndvi',
      help: 'Index layers are computed from calibrated bands; SAVI L selects a GPU-computed soil adjustment. Mixture maps show a selected measured exemplar coefficient or squared spectral residual.',
      options: [
        {value: 'ndvi', label: 'NDVI · vegetation contrast'},
        {value: 'nbr', label: 'NBR · NIR / SWIR2 contrast'},
        {value: 'evi', label: 'EVI · blue-adjusted contrast'},
        {value: 'savi', label: 'SAVI · soil adjustment'},
        {value: 'ndwi-gao', label: 'Gao-style NIR / SWIR1'},
        {value: 'ndwi-mcfeeters', label: 'McFeeters green / NIR'},
        {value: 'abundance', label: 'Mixture weight'},
        {value: 'residual', label: 'Squared model residual'}
      ]
    },
    {
      kind: 'select',
      id: 'saviL',
      label: 'SAVI soil adjustment L',
      group: 'Spectral evidence',
      apply: 'param',
      default: '0.5',
      help: 'Selects among three SAVI rasters computed on the GPU from the same observation.',
      options: [
        {value: '0', label: 'L = 0'},
        {value: '0.5', label: 'L = 0.5'},
        {value: '1', label: 'L = 1'}
      ]
    },
    {
      kind: 'select',
      id: 'indexEpsilon',
      label: 'Minimum absolute denominator',
      group: 'Spectral evidence',
      apply: 'compile',
      default: '0.000001',
      help: 'Rejects a formula when its final absolute denominator is at or below this threshold. Larger settings are an explicit sensitivity experiment, not a cloud or uncertainty model.',
      options: [
        {value: '0.000001', label: '10⁻⁶ · numerical guard'},
        {value: '0.01', label: '0.01 · sensitivity'},
        {value: '0.1', label: '0.1 · sensitivity'}
      ]
    },
    {
      kind: 'select',
      id: 'indexClamp',
      label: 'Computed output clamp',
      group: 'Spectral evidence',
      apply: 'compile',
      default: 'none',
      help: 'Changes computed index values, independently of the fixed map color scale. The narrow range is deliberately restrictive to reveal information lost through clipping. It is not a recommended correction.',
      options: [
        {value: 'none', label: 'Unclipped computed values'},
        {value: 'unit-range', label: 'Clamp to [−1, 1]'},
        {value: 'sensitivity', label: 'Sensitivity cap [−0.25, 0.25]'}
      ]
    },
    {
      kind: 'select',
      id: 'constraint',
      label: 'Fit constraint',
      group: 'Spectral mixture',
      apply: 'param',
      default: 'simplex',
      help: 'Compare ordinary least squares, nonnegative coefficients and a nonnegative simplex fit whose weights sum to one. Iteration-limited constrained fits retain feasible coefficients and status.',
      options: [
        {value: 'unconstrained', label: 'Unconstrained least squares'},
        {value: 'nonnegative', label: 'Nonnegative'},
        {value: 'simplex', label: 'Simplex · sum to one'}
      ]
    },
    {
      kind: 'select',
      id: 'component',
      label: 'Displayed exemplar',
      group: 'Spectral mixture',
      apply: 'param',
      default: '0',
      help: 'Selects one coefficient map; all three weights remain visible in the model readout.',
      options: [
        {value: '0', label: 'Vegetation-like spectrum'},
        {value: '1', label: 'Dark low-NBR spectrum'},
        {value: '2', label: 'Bright low-NDVI spectrum'}
      ]
    },
    {
      kind: 'select',
      id: 'maximumIterations',
      label: 'Iteration budget',
      group: 'Solver',
      apply: 'compile',
      default: '64',
      help: 'Constrains both nonnegative solvers. Unconstrained least squares is solved directly; iteration counts and limit status are read from GPU outputs.',
      options: [
        {value: '64', label: '64 iterations'},
        {value: '256', label: '256 iterations'}
      ]
    },
    {
      kind: 'select',
      id: 'tolerance',
      label: 'Convergence tolerance',
      group: 'Solver',
      apply: 'compile',
      default: '0.001',
      help: 'Tolerance for the projected-gradient mapping in nonnegative and simplex fits. A reported iteration-limit result remains visible.',
      options: [
        {value: '0.001', label: '1 × 10⁻³'},
        {value: '0.0001', label: '1 × 10⁻⁴'}
      ]
    },
    {
      kind: 'select',
      id: 'regionView',
      label: 'Regional evidence',
      group: 'Whole-cell quadrants',
      apply: 'param',
      default: 'indices',
      help: 'Select the GPU date distributions or the paired two-date reflectance difference over shared valid cells.',
      options: [
        {value: 'indices', label: 'Spectral index or mixture'},
        {value: 'distribution', label: 'Date distributions'},
        {value: 'change', label: 'Paired difference'}
      ]
    },
    {
      kind: 'select',
      id: 'selectedRegion',
      label: 'Quadrant',
      group: 'Whole-cell quadrants',
      apply: 'param',
      default: '0',
      help: 'Select one analyst-defined 32 × 32 whole-cell quadrant in north-west, north-east, south-west, south-east order.',
      options: [
        {value: '0', label: 'North-west'},
        {value: '1', label: 'North-east'},
        {value: '2', label: 'South-west'},
        {value: '3', label: 'South-east'}
      ]
    },
    {
      kind: 'select',
      id: 'changeThreshold',
      label: 'Absolute change cutoff',
      group: 'Whole-cell quadrants',
      apply: 'param',
      default: '0.05',
      help: 'Changes the descriptive absolute reflectance-difference cutoff for changed area. This is not severity, significance or causal attribution.',
      options: [
        {value: '0.02', label: '0.02 reflectance'},
        {value: '0.05', label: '0.05 reflectance'},
        {value: '0.10', label: '0.10 reflectance'},
        {value: '0.20', label: '0.20 reflectance'}
      ]
    }
  ],
  readouts: [
    {id: 'sourceSupport', label: 'Source support'},
    {id: 'indexEvidence', label: 'Selected-pixel index values'},
    {id: 'indexChart', label: 'Index contrasts at selected pixel', kind: 'chart'},
    {id: 'exemplars', label: 'Measured exemplars'},
    {id: 'exemplarSpectra', label: 'Six-band exemplar spectra', kind: 'chart'},
    {id: 'modelEvidence', label: 'GPU mixture weights and fit status'},
    {id: 'fitStatus', label: 'GPU solver status across crop'},
    {id: 'pixelSpectra', label: 'Observed and modelled pixel spectrum', kind: 'chart'},
    {id: 'regionalEvidence', label: 'Quadrant support and paired change'},
    {id: 'regionalChart', label: 'Regional NIR distributions', kind: 'chart'}
  ],
  legends: state =>
    state.regionView === 'change'
      ? [
          {
            id: 'regionalChange',
            kind: 'ramp' as const,
            title: 'September − July NIR reflectance',
            ramp: 'vik' as const,
            extent: [-0.4, 0.4] as const,
            midpoint: 0,
            unit: 'reflectance'
          }
        ]
      : state.regionView === 'distribution'
        ? [
            {
              id: 'regionalDistribution',
              kind: 'ramp' as const,
              title: 'Quadrant median NIR reflectance',
              ramp: 'cividis' as const,
              extent: [0, 0.4] as const,
              unit: 'reflectance'
            }
          ]
        : state.product === 'abundance'
          ? [
              {
                kind: 'ramp',
                id: 'abundance',
                title: 'Selected mixture coefficient',
                ramp: state.constraint === 'unconstrained' ? 'vik' : 'cividis',
                extent: 'gpu',
                ...(state.constraint === 'unconstrained' ? {midpoint: 0} : {}),
                unit: 'model weight'
              }
            ]
          : state.product === 'residual'
            ? [
                {
                  id: 'residual',
                  kind: 'ramp',
                  title: 'Squared six-band residual',
                  ramp: 'cividis',
                  extent: 'gpu',
                  unit: 'reflectance²'
                }
              ]
            : [
                {
                  id: 'index',
                  kind: 'ramp',
                  title: 'Unitless index · fixed display −1 to 1',
                  ramp: 'vik',
                  extent: [-1, 1],
                  midpoint: 0,
                  unit: 'index'
                }
              ],
  story: [
    {
      id: 'calibrated-observations',
      headline: 'Six calibrated bands carry different evidence',
      textAlternative:
        'A 64 by 64 UTM crop maps a GPU index with the selected pixel and three measured spectral exemplars marked.',
      title: 'Begin with measured reflectance',
      body: 'This bounded crop pairs six calibrated unitless bands: blue, green, red, NIR, SWIR1 and SWIR2. Each scene is a separate observation; July 13 is the ignition date, while September 21 is post-ignition. Click a cell to inspect its six-band spectrum and source support.',
      evidence: '**{{sourceSupport}}**',
      caveat:
        'The strict mask requires all six bands and contributing SCL classes 4, 5 or 6. It is conservative screening, not hand-checked cloud removal.',
      options: {date: 'after', product: 'ndvi'},
      controls: ['date'],
      readouts: ['sourceSupport', 'pixelSpectra']
    },
    {
      id: 'index-contrasts',
      headline: 'A ratio emphasizes one band contrast at a time',
      textAlternative:
        'GPU NDVI, NBR, EVI, SAVI and two explicitly named NDWI conventions are compared at one pixel.',
      title: 'Change the contrast, not the observation',
      body: 'NDVI compares NIR and red; NBR compares NIR and SWIR2; EVI adds a blue correction; SAVI exposes L = 0, 0.5 or 1. The two water indices differ: Gao-style NIR/SWIR1 is a Sentinel-specific moisture adaptation, while McFeeters compares green and NIR. The chart uses separate bars for GPU values at the selected pixel.',
      evidence: '**{{indexEvidence}}**',
      caveat:
        'These are unitless spectral indices with denominator validity preserved. The denominator threshold and output clamp are sensitivity controls: a narrow clamp changes the computed values, while the map color domain only changes their display. SWIR1 is 1.61 µm and does not reproduce Gao’s original 1.24 µm band.',
      options: {product: 'nbr'},
      controls: ['product', 'saviL', 'date', 'indexEpsilon', 'indexClamp'],
      readouts: ['indexEvidence', 'indexChart']
    },
    {
      id: 'vegetation-emphasis',
      headline: 'Vegetation-oriented indices need not agree',
      textAlternative:
        'The map shows the actual selected index output and the chart compares all six GPU index results at a clicked cell.',
      title: 'Look at a pixel across formulas',
      body: 'The formulas reuse the same calibrated observation, but their wavelengths and adjustments change which contrast is strongest. Select any index and click another pixel to compare its values without treating them as a validated class.',
      evidence: '**{{indexEvidence}}**',
      caveat:
        'A large index response is not a material identification or a causal fire-impact estimate.',
      options: {product: 'evi'},
      controls: ['product', 'saviL', 'date'],
      readouts: ['indexEvidence', 'indexChart']
    },
    {
      id: 'measured-exemplars',
      headline: 'The mixture basis comes from measured pixels',
      textAlternative:
        'Three marked source cells contribute their real six-band spectra to the mixture model.',
      title: 'Inspect the exemplar spectra',
      body: 'Three September source pixels provide the displayed model basis: vegetation-like, dark low-NBR and bright low-NDVI spectra. Their six reflectances and row/column locations are plotted. The labels describe spectral signatures, not independently verified materials.',
      evidence: '**{{exemplars}}**',
      caveat:
        'The fitted basis is deliberately small and scene-specific. It cannot represent every surface or illumination condition.',
      options: {date: 'after', product: 'abundance', constraint: 'simplex'},
      controls: ['date', 'product'],
      readouts: ['exemplars', 'exemplarSpectra']
    },
    {
      id: 'constraint-changes-fit',
      headline: 'Constraints change what the coefficients mean',
      textAlternative:
        'GPU mixture weights, squared residual, solver status and iterations are shown for the clicked source pixel.',
      title: 'Compare feasible mixture models',
      body: 'Unconstrained least squares may assign negative coefficients. Nonnegative fitting clips the feasible domain through projected-gradient optimization; simplex fitting also enforces a unit sum. The map displays a selected coefficient, while the readout reports all three GPU weights, residual, status and iterations.',
      evidence: '**{{modelEvidence}}**',
      caveat:
        'Weights depend on the chosen three spectra and optimization constraints. They are not validated land-cover fractions.',
      options: {product: 'abundance', constraint: 'simplex', component: '0'},
      controls: ['constraint', 'component', 'maximumIterations', 'tolerance'],
      readouts: ['modelEvidence', 'fitStatus', 'pixelSpectra']
    },
    {
      id: 'residual-limits',
      headline: 'Regional summaries keep support visible',
      textAlternative:
        'Four fixed whole-cell quadrants compare date-specific NIR distributions and paired two-date changes over shared valid cells.',
      title: 'Compare distributions and paired change by quadrant',
      body: '**Regional evidence** below selects date quantiles or the paired September-minus-July difference. **Quadrant** selects one fixed whole-cell region. **Absolute change cutoff** changes the descriptive changed-area tally. The chart shows each acquisition’s quantiles separately from the paired common-support change. The six-band residual remains a local fit diagnostic for the selected exemplar basis.',
      evidence: '**{{regionalEvidence}}**',
      caveat:
        'July 13 is ignition day and the dates differ in season and QA support. Two acquisitions cannot identify a causal fire effect; the cutoff is not severity or significance. Each quadrant has equal whole-cell geometry, while valid support follows the strict per-date mask.',
      options: {regionView: 'distribution', selectedRegion: '0', changeThreshold: '0.05'},
      controls: ['regionView', 'selectedRegion', 'changeThreshold'],
      readouts: ['regionalEvidence', 'regionalChart']
    }
  ],
  create: async ctx => (await import('./spectral-evidence.compute')).createSpectralEvidence(ctx)
});
