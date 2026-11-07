// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SurveyFrameSensitivityOptions} from './survey-frame-sensitivity.types';

export default defineScene<SurveyFrameSensitivityOptions>({
  id: 'survey-frame-sensitivity',
  title: 'How much does the control area change the correction?',
  chapter: 'terrain',
  order: 10,
  summary:
    'Test a robust vertical median shift on two real, paired terrestrial-laser-scan surface grids while keeping the control mask and uncertainty assumptions explicit.',
  contributors: ['GPUTerrainDEMCoregistration'],
  datasets: [{id: 'ahk-rock-glacier', role: 'paired 2017 and 2018A mean-return surface cells'}],
  initialView: {
    longitude: 11.005414391236211,
    latitude: 46.83836570985789,
    zoom: 20.1,
    pitch: 0,
    bearing: 0
  },
  basemap: ground('night'),
  furniture: {title: true, credit: true, scaleBar: true},
  annotations: [
    {
      kind: 'area',
      coordinate: [11.005414391236211, 46.83836570985789],
      text: 'Äußeres Hochebenkar · 1 m cells',
      size: 'small'
    }
  ],
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Map result',
      group: 'Display',
      apply: 'param',
      default: 'raw',
      display: 'segmented',
      help: 'Switch between raw epoch difference, vertically corrected residual, and the assumed per-cell uncertainty.',
      options: [
        {value: 'raw', label: 'Raw difference'},
        {value: 'corrected', label: 'After vertical shift'},
        {value: 'uncertainty', label: 'Assumed uncertainty'}
      ]
    },
    {
      kind: 'select',
      id: 'controlRegion',
      label: 'Hypothetical control area',
      group: 'Vertical fit sensitivity',
      apply: 'param',
      default: 'west',
      display: 'chips',
      help: 'Selects paired cells only. These masks are study scenarios; the source provides no authoritative stable-ground mask or glacier polygon.',
      options: [
        {value: 'west', label: 'West'},
        {value: 'east', label: 'East'},
        {value: 'north', label: 'North'},
        {value: 'all-paired', label: 'All paired cells'}
      ]
    },
    {
      kind: 'slider',
      id: 'assumedDifferenceSd',
      label: 'Assumed raw-difference SD',
      group: 'Uncertainty assumptions',
      apply: 'compile',
      min: 0.05,
      max: 0.5,
      step: 0.05,
      default: 0.15,
      unit: 'm',
      help: 'A user-set per-cell standard deviation for the raw epoch difference; this is not a calibrated source uncertainty.'
    },
    {
      kind: 'slider',
      id: 'correlationLength',
      label: 'Assumed correlation length',
      group: 'Uncertainty assumptions',
      apply: 'compile',
      min: 0,
      max: 5,
      step: 1,
      default: 2,
      unit: 'm',
      marks: [
        {value: 0, label: 'Independent'},
        {value: 2, label: '2 m'},
        {value: 5, label: '5 m'}
      ],
      help: 'Sets an exponential spatial covariance range for the assumed raw-difference errors; it is a sensitivity assumption, not a measured correlation length.'
    }
  ],
  readouts: [
    {
      id: 'pairedCells',
      label: 'Paired observed cells',
      format: 'integer',
      emphasis: 'tile',
      help: 'Cells with at least three returns from the retained original-row sample in each epoch; no resampling or gap filling.'
    },
    {
      id: 'controlCells',
      label: 'Cells in hypothetical control area',
      format: 'integer',
      help: 'Paired observed cells selected for the vertical median fit.'
    },
    {
      id: 'rawBias',
      label: 'Raw median vertical offset',
      format: 'decimal',
      unit: 'm',
      help: 'Median of moving-minus-reference heights over the selected hypothetical control cells.'
    },
    {
      id: 'residualNmad',
      label: 'Control residual NMAD',
      format: 'decimal',
      unit: 'm',
      help: 'Robust dispersion of control-cell residuals after subtracting the median vertical shift.'
    },
    {
      id: 'biasStandardError',
      label: 'Estimated bias standard error',
      format: 'decimal',
      unit: 'm',
      help: 'Contributor estimate from residual NMAD and selected control-cell covariance. The user-set per-cell standard deviation contributes to the measurement and volume uncertainty model.'
    },
    {
      id: 'correctedVolume',
      label: 'Corrected vertical surface volume',
      format: 'decimal',
      unit: 'm³',
      help: 'Area-integrated corrected height difference across paired observed cells; this is not mass balance.'
    },
    {
      id: 'volumeStandardError',
      label: 'Volume standard error',
      format: 'decimal',
      unit: 'm³',
      help: 'Propagated from the assumed per-cell error and exponential spatial covariance; not source-calibrated.'
    },
    {
      id: 'validArea',
      label: 'Paired observed area',
      format: 'integer',
      unit: 'm²',
      help: 'Count of cells valid in both epochs multiplied by the 1 m² cell area.'
    },
    {
      id: 'rawVolume',
      label: 'Raw vertical surface volume',
      format: 'decimal',
      unit: 'm³',
      help: 'Area-integrated uncorrected moving-minus-reference difference over paired observed cells.'
    },
    {
      id: 'assumedDifferenceSd',
      label: 'Assumed raw-difference SD',
      format: 'decimal',
      unit: 'm',
      help: 'User-selected hypothetical per-cell standard deviation.'
    },
    {
      id: 'correlationLengthReadout',
      label: 'Assumed correlation length',
      format: 'decimal',
      unit: 'm',
      help: 'User-selected exponential covariance range.'
    },
    {id: 'maskVolumeChart', label: 'Volume by hypothetical control scenario', kind: 'chart'},
    {
      id: 'residualHistogram',
      label: 'Corrected paired-cell differences',
      kind: 'chart',
      help: 'Twenty fixed bins span -1 m to +1 m; corrected values outside that range are counted in the end bins.'
    }
  ],
  story: [
    {
      id: 'paired-grids',
      title: 'Compare paired surface cells',
      headline: 'Each mapped value comes from measured returns in both epochs',
      textAlternative:
        'A map compares unfilled one-metre mean-return cells from the 2017 and 2018A lidar epochs. Cells lacking three returns in either epoch are omitted.',
      body: 'The 38 × 37 grid stores mean heights from the retained original-row sample for 2017 and 2018A in each one-metre cell. The paired support therefore reflects that retained sample, not every full-resolution return. The analysis uses the native local vertical axis after the supplied survey preparation; it makes no horizontal fit and fills no gaps.',
      evidence: '**{{pairedCells}}** cells have at least three returns in each epoch.',
      caveat:
        'Paired support differs between epochs. A mean of returns in a cell is not a bare-earth DEM, and the cells are not an independent sample of the full scanner uncertainty.',
      options: {view: 'raw'},
      controls: ['view'],
      readouts: ['pairedCells']
    },
    {
      id: 'choose-control',
      title: 'Choose a hypothetical control region',
      headline: 'The vertical correction depends on which paired cells are treated as stable',
      textAlternative:
        'A control-region selector changes the mask used by a robust median vertical-offset fit; each option is explicitly hypothetical.',
      body: 'Select west, east, north, or all paired cells as a control scenario. The dataset does not identify stable terrain, so none of these masks represents observed ground truth.',
      evidence: '**{{controlCells}}** paired cells contribute to the selected vertical fit.',
      caveat:
        'A real stable-area mask requires external geomorphic or survey evidence. The fit estimates a vertical shift only; it cannot correct horizontal misregistration.',
      options: {view: 'raw'},
      controls: ['view', 'controlRegion'],
      readouts: ['controlCells', 'rawBias', 'residualNmad', 'maskVolumeChart']
    },
    {
      id: 'subtract-median',
      title: 'Subtract the vertical median',
      headline: 'One robust vertical shift can remove an offset without aligning the surfaces',
      textAlternative:
        'The corrected map subtracts the median moving-minus-reference offset from each paired cell; it does not move points or adjust horizontal position.',
      body: '`GPUTerrainDEMCoregistration` estimates the median of moving minus reference heights in the chosen control region, then subtracts that one value from all paired cells. The residual NMAD describes control residual spread after the shift.',
      evidence:
        'The fitted offset is **{{rawBias}}**, with a control residual NMAD of **{{residualNmad}}**.',
      caveat:
        'A control-region bias can contain real surface change. The residual NMAD is not a formal accuracy statement or a replacement for a survey error model.',
      options: {view: 'corrected'},
      controls: ['view', 'controlRegion'],
      readouts: ['rawBias', 'residualNmad', 'residualHistogram', 'maskVolumeChart']
    },
    {
      id: 'uncertainty-volume',
      title: 'Vary the uncertainty assumptions',
      headline:
        'Spatial correlation changes volume uncertainty even when the correction stays fixed',
      textAlternative:
        'An assumed per-cell uncertainty and exponential correlation length feed the GPU covariance calculation for corrected volume standard error.',
      body: 'The source subset does not include a calibrated raster error covariance. Set a hypothetical standard deviation for each raw cell difference and an exponential correlation length, then inspect how the propagated volume uncertainty responds.',
      evidence:
        'The corrected vertical surface volume is **{{correctedVolume}}** with an assumed-model standard error of **{{volumeStandardError}}** over **{{validArea}}**.',
      caveat:
        'These are sensitivity results under user-chosen assumptions, not measured annual mass balance. The two epochs are labelled 2017 and 2018A; no annual velocity is inferred.',
      options: {view: 'uncertainty'},
      controls: ['view', 'controlRegion', 'assumedDifferenceSd', 'correlationLength'],
      readouts: ['correctedVolume', 'volumeStandardError', 'validArea', 'maskVolumeChart']
    }
  ],
  pipeline: [
    {
      id: 'raster-coregistration',
      label: 'Vertical fit',
      detail:
        'Fits a robust vertical median shift from selected paired cells, applies it, and estimates uncertainty under explicit covariance assumptions.'
    }
  ],
  about: {
    what: 'A robust median vertical offset is fitted from selected paired cells, subtracted from the comparison surface, then uncertainty is propagated under an explicit spatial covariance model.',
    why: 'The chosen control area can contain real surface change. Comparing control masks shows how a vertical correction can change the sign and magnitude of an integrated surface-volume estimate.',
    howToRead:
      'The source contains 1 m cell means from original scanner returns in two campaigns, with no resampling or gap filling. The supplied preparation aligns the comparison epoch; this analysis applies no horizontal correction. The map locates cell centers and outlines the grid footprint; slight projection differences can make its north-up quad an approximation of the native UTM grid edges. All calculations use the exact native affine. Dataset source: Winiwarter et al. (2021), DOI 10.11588/data/XHYB10, CC BY 4.0.'
  },
  legends: state => [
    state.view === 'uncertainty'
      ? {
          kind: 'ramp',
          title: 'Assumed-model standard deviation',
          ramp: 'cividis',
          colors: UNCERTAINTY_COLORS,
          extent: [0, 0.6],
          unit: 'm',
          note: 'User-set per-cell uncertainty and covariance; not calibrated source error.'
        }
      : {
          kind: 'ramp',
          title: state.view === 'raw' ? 'Raw moving − reference' : 'After vertical median shift',
          ramp: 'vik',
          colors: DIFFERENCE_COLORS,
          extent: [-1, 1],
          unit: 'm',
          midpoint: 0,
          midpointLabel: 'zero',
          note: 'Fixed scale; cells without paired returns are hatched.'
        },
    {
      kind: 'categories',
      title: 'Selected hypothetical control boundary',
      entries: [{color: CONTROL_COLOR, label: FIT_LABELS[state.controlRegion]}],
      note: 'Scenario mask only; the source does not identify stable ground.'
    }
  ],
  snippet: `new GPUTerrainDEMCoregistration({
  grid: demGrid,
  reference,
  moving,
  stableMask: hypotheticalControlMask,
  differenceVariance,
  correlationLength,
  output: {fit, differences, validity, volume}
});`,
  create: async ctx =>
    (await import('./survey-frame-sensitivity.compute')).createSurveyFrameSensitivity(ctx)
});

const DIFFERENCE_COLORS = [
  [57, 116, 158],
  [152, 190, 194],
  [232, 226, 207],
  [218, 154, 123],
  [167, 75, 75]
] as const;
const UNCERTAINTY_COLORS = [
  [34, 66, 90],
  [81, 125, 135],
  [190, 185, 131],
  [239, 202, 126]
] as const;
const CONTROL_COLOR = [255, 205, 120, 235] as const;
const FIT_LABELS: Record<SurveyFrameSensitivityOptions['controlRegion'], string> = {
  west: 'West',
  east: 'East',
  north: 'North',
  'all-paired': 'All paired cells'
};
