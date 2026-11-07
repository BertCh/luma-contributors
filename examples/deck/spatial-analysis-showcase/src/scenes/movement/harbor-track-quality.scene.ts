// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {HarborTrackQualityOptions} from './harbor-track-quality.compute';
import {MOVEMENT_CREDITS} from './movement-style';

const UPPER_BAY = {longitude: -74.035, latitude: 40.672, zoom: 12.2};

export default defineScene<HarborTrackQualityOptions>({
  id: 'harbor-track-quality',
  title: 'When an AIS track has gaps',
  chapter: 'movement',
  order: 5,
  summary:
    'Inspect the sampling gaps and speed flags in a real harbor vessel track, then compare raw fixes with GPU Kalman filtering and smoothing.',
  contributors: [
    'addTrackQualityRecipe',
    'GPUTrajectoryMetrics',
    'GPUTrajectoryRestriction',
    'GPUTrajectoryKalman'
  ],
  datasets: [{id: 'ais-vessels', role: 'AIS observations in New York Harbor, 12 June 2024'}],
  initialView: UPPER_BAY,
  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: {
      subtitle: 'One vessel · observed fixes and model estimates',
      chips: ['Sampling gaps are not proven errors']
    },
    scaleBar: {units: 'nautical'},
    credit: joinCredits(MOVEMENT_CREDITS.harborAis),
    caveat:
      'AIS is sampled unevenly. The dataset thins stationary tracks to about one fix per five minutes; a threshold flag asks for review and does not prove a bad fix.'
  },
  options: [
    {
      kind: 'slider',
      id: 'maximumGapSeconds',
      label: 'Flag gaps longer than',
      group: 'Raw observation review',
      apply: 'param',
      min: 60,
      max: 900,
      step: 30,
      default: 300,
      unit: 's',
      help: 'Flags the later endpoint of an interval longer than this threshold. Stationary-vessel thinning can create long gaps by design.'
    },
    {
      kind: 'slider',
      id: 'maximumSpeedKnots',
      label: 'Flag speed above',
      group: 'Raw observation review',
      apply: 'param',
      min: 5,
      max: 35,
      step: 1,
      default: 20,
      unit: 'kn',
      help: 'Flags a derived step speed above this threshold. A flag is a review cue, not a measured error.'
    },
    {
      kind: 'slider',
      id: 'maximumModelGapSeconds',
      label: 'Break model across gaps',
      group: 'Motion model',
      apply: 'param',
      min: 60,
      max: 1200,
      step: 30,
      default: 300,
      unit: 's',
      help: 'The filter starts a new state after this time gap; restriction marks longer intervals as unknown and does not draw a straight bridge.'
    },
    {
      kind: 'slider',
      id: 'windowEndHour',
      label: 'Show observations through',
      group: 'Sample window',
      apply: 'param',
      min: 1,
      max: 21,
      step: 1,
      default: 21,
      unit: ':00 UTC',
      help: 'Clip supported path segments to the chosen UTC time. This selected track ends before midnight in the bundled snapshot.'
    },
    {
      kind: 'slider',
      id: 'processNoise',
      label: 'Acceleration spectral density',
      group: 'Motion model',
      apply: 'param',
      min: 0.01,
      max: 8,
      step: 0.1,
      default: 1.5,
      unit: 'm²/s³',
      help: 'Constant-velocity model acceleration spectral density. Higher values let the estimate respond more quickly to turns.'
    },
    {
      kind: 'slider',
      id: 'measurementVariance',
      label: 'Position variance',
      group: 'Motion model',
      apply: 'param',
      min: 100,
      max: 10000,
      step: 100,
      default: 2500,
      unit: 'm²',
      help: 'Assumed isotropic GPS position measurement variance; this is a model setting, not a claim about receiver accuracy.'
    },
    {
      kind: 'select',
      id: 'covarianceModel',
      label: 'Position covariance model',
      group: 'Motion model',
      apply: 'param',
      default: 'isotropic',
      options: [
        {value: 'isotropic', label: 'Isotropic covariance'},
        {value: 'full', label: 'Full correlated covariance'}
      ],
      help: 'Both GPU filters and RTS smoothers run from the same observations. Full covariance tracks x/y position and velocity cross-covariances; axis ratio 1 and correlation 0 should match isotropic mode within floating-point tolerance.'
    },
    {
      kind: 'slider',
      id: 'axisRatio',
      label: 'Assumed x/y variance ratio',
      group: 'Full covariance assumptions',
      apply: 'param',
      min: 1,
      max: 4,
      step: 0.25,
      default: 1,
      help: 'Sets qxx/qyy and rxx/ryy to reciprocal positive ratios. Initial velocity variance stays isotropic. These are model assumptions, not calibration from AIS truth.'
    },
    {
      kind: 'slider',
      id: 'axisCorrelation',
      label: 'Assumed x/y correlation ρ',
      group: 'Full covariance assumptions',
      apply: 'param',
      min: -0.8,
      max: 0.8,
      step: 0.1,
      default: 0,
      help: 'Applies the same bounded correlation to process and measurement covariance. The |ρ| ≤ 0.8 bound with positive diagonal variances keeps both 2×2 covariance matrices positive definite.'
    },
    {
      kind: 'toggle',
      id: 'showUncertainty',
      label: 'Show 1σ position ellipses',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws model-implied one-standard-deviation ellipses at every fourth valid estimate, clipped to the selected observation window.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Track view',
      group: 'Display',
      apply: 'param',
      default: 'smoothed',
      help: 'Compare observations, model estimates, gap-supported intervals, or a GPU-clipped study polygon with an even-odd hole and merged interval union.',
      options: [
        {value: 'observed', label: 'Observed fixes'},
        {value: 'filtered', label: 'Forward filtered'},
        {value: 'smoothed', label: 'Smoothed estimate'},
        {value: 'gaps', label: 'Supported intervals'},
        {value: 'zone', label: 'Polygon zone + merged intervals'}
      ]
    }
  ],
  story: [
    {
      id: 'sampled-path',
      title: 'One path is a sample of movement',
      headline: 'Observed fixes sample a route; gaps stay uncertain.',
      body: 'This vessel has **{{trackId}}**. Its observed fixes span **{{trackWindow}}** and represent the reporting pattern in the bundled AIS snapshot, not a continuous GPS recording. The window control clips the displayed route; length and time span below summarize the full selected track. Integer epoch subtraction preserves the original second-resolution times before GPU float32 upload; uploading large absolute epoch seconds directly would merge distinct report times.',
      evidence:
        'The full observed path has **{{trackLength}}** total length across **{{trackDuration}}**. Clock audit: **{{clockPrecision}}**.',
      caveat: 'AIS reporting intervals depend on vessel state and receiver coverage.',
      textAlternative: 'A real New York Harbor AIS path is drawn as observed line segments.',
      camera: {
        bounds: [-74.075, 40.639, -74.009, 40.703],
        padding: 56,
        zoom: 13.4,
        transitionMs: 1200
      },
      options: {display: 'observed'},
      controls: ['display', 'windowEndHour'],
      readouts: ['trackId', 'trackWindow', 'trackLength', 'trackDuration']
    },
    {
      id: 'flag-review',
      title: 'A threshold flag is a prompt to inspect',
      headline: 'Flags mark thresholds, not proven bad fixes.',
      body: 'The GPU recipe computes step speeds and flags long time gaps or speeds above the selected threshold. Adjust **Flag gaps longer than** and **Flag speed above** to see how sensitive the review is.',
      evidence:
        '**{{quality}}** under current thresholds; maximum derived step speed is **{{maximumSpeed}}**.',
      caveat:
        'The source manifest describes stationary tracks as thinned to about five minutes, so a long gap can be intentional sampling.',
      textAlternative:
        'The path is shown with threshold-based review counts and a bar chart of gap and speed flags.',
      options: {display: 'gaps'},
      controls: ['maximumGapSeconds', 'maximumSpeedKnots'],
      readouts: ['quality', 'maximumSpeed', 'flagsChart']
    },
    {
      id: 'filter',
      title: 'Filtering follows observations forward',
      headline: 'The filter restarts after unsupported gaps.',
      body: 'The constant-velocity Kalman filter combines each fix with a motion prediction. The model breaks across large gaps instead of pretending it knows what happened during the missing interval.',
      evidence: '**{{kalmanResets}}** mark independently initialized filter segments.',
      caveat:
        'Process noise and measurement variance are assumptions; the filtered line is an estimate, not validation of raw coordinates.',
      textAlternative:
        'A purple forward estimate follows the observed fixes and restarts after large gaps.',
      options: {display: 'filtered'},
      controls: ['maximumModelGapSeconds', 'processNoise', 'measurementVariance'],
      readouts: ['kalmanResets']
    },
    {
      id: 'smooth',
      title: 'Smoothing can erase turns',
      headline: 'Future fixes can flatten earlier maneuvers.',
      body: 'The Rauch–Tung–Striebel smoother uses later observations to revise earlier states. Compare filtered and smoothed paths in isotropic and full-covariance mode. In full mode, **Assumed x/y variance ratio** and **correlation ρ** shape both process and measurement matrices; axis ratio 1 with ρ = 0 is the parity check against the isotropic result.',
      evidence:
        'Fit residuals are **{{modelResidual}}**. At ratio 1 and ρ 0, the parallel covariance run differs by **{{covarianceParity}}**. The chart and map ellipses show **{{uncertaintySummary}}**.',
      caveat:
        'Residuals describe fit to the same observations used by the model, not independent accuracy. A smoother can flatten real maneuvers. Covariance and one-sigma ellipses are conditional on assumed noise, not calibrated accuracy or a verified vessel-position error bound.',
      textAlternative:
        'The orange smoothed model is overlaid on observations with RMS residuals to the same input fixes.',
      camera: {
        bounds: [-74.074, 40.64, -74.0685, 40.646],
        padding: 56,
        zoom: 15.8,
        transitionMs: 1200
      },
      options: {display: 'smoothed'},
      controls: ['covarianceModel', 'axisRatio', 'axisCorrelation', 'showUncertainty'],
      readouts: ['modelResidual', 'covarianceParity', 'uncertaintySummary', 'uncertaintyChart']
    },
    {
      id: 'unknown-support',
      title: 'Do not bridge unsupported time',
      headline: 'Long gaps are unknown support, not straight routes.',
      body: 'The GPU trajectory restriction clips intervals to the observed time window and treats intervals beyond **Break model across gaps** as unknown support. Choose **Supported intervals** to focus on segments whose time gap is within the assumption.',
      evidence: '**{{quality}}**; the gap chart separates long-gap endpoints from speed flags.',
      caveat:
        'Straight segments between fixes are only a display interpolation over supported intervals; they do not reveal the vessel’s exact route between reports.',
      textAlternative:
        'Only source intervals within the selected maximum-gap assumption are highlighted.',
      camera: {
        bounds: [-74.075, 40.639, -74.009, 40.703],
        padding: 56,
        zoom: 13.4,
        transitionMs: 1200
      },
      options: {display: 'gaps'},
      controls: ['maximumModelGapSeconds', 'display', 'windowEndHour'],
      readouts: ['quality', 'flagsChart']
    },
    {
      id: 'polygon-hole-union',
      title: 'A hand-drawn polygon can contain a hole',
      headline: 'Geometry exclusion and missing time mean different things.',
      body: 'The same observed AIS segments are clipped by a GPU even-odd polygon: the outer ring covers this track extent, while the inner ring removes a small study area. Adjacent included intervals are merged into a compact GPU output, and the map draws the resulting union with both ring boundaries.',
      evidence: '**{{zoneRestriction}}**. Compaction status: **{{zoneCompaction}}**.',
      caveat:
        'This is an illustrative analysis mask, not an official harbor boundary. Intervals that cross a gap above the selected threshold are unknown; a segment outside the polygon is simply excluded and does not imply absence of vessel activity.',
      textAlternative:
        'The map shows a hand-drawn outer polygon, an excluded inner hole, and merged supported AIS intervals.',
      camera: {
        bounds: [-74.075, 40.639, -74.009, 40.703],
        padding: 56,
        zoom: 13.4,
        transitionMs: 1200
      },
      options: {display: 'zone'},
      controls: ['maximumModelGapSeconds', 'windowEndHour', 'display'],
      readouts: ['zoneRestriction', 'zoneCompaction']
    }
  ],
  legends: state => [
    {
      kind: 'categories',
      title: 'Track interpretation',
      entries: [
        {color: [156, 177, 191, 180], label: 'Observed AIS segment'},
        {color: [134, 70, 201, 240], label: 'Forward filtered'},
        {color: [232, 112, 50, 245], label: 'Smoothed estimate'},
        {color: [45, 155, 183, 245], label: 'Gap within model support'},
        {color: [67, 223, 218, 170], label: 'Model-implied 1σ position ellipse'},
        {color: [99, 216, 202, 230], label: 'Hand-drawn polygon and hole rings'},
        {color: [236, 178, 74, 245], label: 'Merged supported intervals inside polygon'}
      ],
      note:
        state.display === 'gaps'
          ? 'Longer intervals remain unknown.'
          : 'Model lines are conditional estimates, not verified vessel positions.'
    }
  ],
  readouts: [
    {id: 'trackId', label: 'Selected vessel'},
    {id: 'trackWindow', label: 'Observed UTC window'},
    {id: 'trackLength', label: 'Observed path length'},
    {id: 'trackDuration', label: 'Observed span'},
    {id: 'clockPrecision', label: 'Epoch precision audit'},
    {id: 'zoneRestriction', label: 'Polygon clip and unknown gaps'},
    {id: 'zoneCompaction', label: 'GPU interval-union status'},
    {id: 'quality', label: 'Raw threshold review'},
    {id: 'maximumSpeed', label: 'Maximum step speed'},
    {id: 'kalmanResets', label: 'Filter starts'},
    {id: 'kalmanValidity', label: 'Selected model status'},
    {id: 'modelResidual', label: 'Fit residual to input fixes'},
    {id: 'covarianceParity', label: 'Isotropic/full covariance parity'},
    {id: 'uncertaintySummary', label: 'Model-implied position uncertainty'},
    {id: 'flagsChart', label: 'Raw flag counts', kind: 'chart'},
    {id: 'uncertaintyChart', label: 'Position uncertainty over time', kind: 'chart'}
  ],
  snippet: state =>
    `addTrackQualityRecipe(graph, { trajectory, thresholds }); // ${state.display} · ${state.maximumGapSeconds}s flag threshold`,
  about: {
    what: 'A GPU review of raw AIS speed and time gaps alongside Kalman-filtered and smoothed vessel positions.',
    why: 'Irregular reporting makes an AIS line look more certain than its observations support; this scene makes the sampling assumptions visible.',
    howToRead:
      'Observed fixes are grey, model estimates use purple or orange, and cyan marks intervals within the chosen maximum-gap assumption. Threshold flags and covariance describe the selected model, not truth.'
  },
  create: async ctx =>
    (await import('./harbor-track-quality.compute')).createHarborTrackQuality(ctx)
});
