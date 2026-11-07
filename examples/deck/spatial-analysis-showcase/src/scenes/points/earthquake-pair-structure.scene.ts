// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {EarthquakePairStructureOptions} from './earthquake-pair-structure.compute';

const JAPAN_VIEW = {longitude: 140.5, latitude: 36.5, zoom: 6.4};
const DEPTH_COLORS = [
  [30, 84, 118, 255],
  [45, 144, 151, 255],
  [204, 171, 103, 255],
  [171, 83, 78, 255]
] as const;
const INTENSITY_COLORS = [
  [199, 215, 201, 255],
  [61, 151, 145, 255],
  [211, 128, 76, 255]
] as const;

function legends(state: EarthquakePairStructureOptions): LegendSpec[] {
  const output: LegendSpec[] = [
    {
      id: 'reported-depth',
      kind: 'ramp',
      ramp: 'batlow',
      title: 'Reported earthquake depth',
      colors: DEPTH_COLORS,
      extent: 'gpu',
      unit: 'km',
      note: 'Input mark; exact 10 km outlines are a flagged sensitivity case.'
    }
  ];
  if (state.mode === 'inhomogeneous') {
    output.push({
      id: 'estimated-intensity',
      kind: 'ramp',
      ramp: 'mako',
      title: 'GPU intensity overlay',
      colors: INTENSITY_COLORS,
      extent: 'gpu',
      scale: 'log',
      unit: 'events/m²',
      note: 'GPU values stretch from the 5th–95th percentile on a log scale; tail colors clip.'
    });
  }
  return output;
}

/** GPU marked-pair and inhomogeneous intensity sensitivity for a fixed Japanese study window. */
export default defineScene<EarthquakePairStructureOptions>({
  id: 'earthquake-pair-structure',
  title: 'Do nearby earthquakes share reported depth?',
  chapter: 'points',
  order: 3,
  create: async context =>
    (await import('./earthquake-pair-structure.compute')).createEarthquakePairStructure(context),
  summary:
    'Follow depth marks and pair statistics for all archived ComCat earthquake rows inside one fixed projected rectangle; compare unweighted, constant-intensity and kernel-intensity assumptions without a hazard or causal claim.',
  contributors: [
    'GPUMarkedPointStatistics',
    'GPUInhomogeneousPointPattern',
    'GPUStudyWindowPreparation'
  ],
  datasets: [
    {
      id: 'poopdeck-earthquakes',
      role: '77,231-row archived ComCat selection, filtered to earthquake type and the fixed AEQD rectangle'
    }
  ],
  initialView: JAPAN_VIEW,
  basemap: ground('paperSheet'),
  furniture: {
    title: {
      title: 'Earthquake pair structure',
      subtitle: 'Reported depth marks · fixed azimuthal-equidistant window',
      chips: ['802 type-0 events', 'Not a hazard map']
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.usgs, 'poopdeck.gl archive'),
    caveat:
      'The archived selection ends 2024-12-30 and omits depth-uncertainty and fixed-depth flags. Exact 10 km values do not prove a fixed-depth solution. Events are time-pooled, dependent observations; this is not a causal tectonic or hazard analysis.'
  },
  legends,
  options: [
    {
      kind: 'select',
      id: 'mode',
      label: 'Pair analysis',
      group: 'Method',
      apply: 'compile',
      default: 'marked',
      options: [
        {value: 'marked', label: 'Marked point pairs'},
        {value: 'inhomogeneous', label: 'Inhomogeneous intensity'}
      ],
      help: 'Builds only the selected GPU pipeline on demand. Every graph uses the same full 802-event rectangle.'
    },
    {
      kind: 'select',
      id: 'depthPair',
      label: 'Depth groups',
      group: 'Marked pairs',
      apply: 'compile',
      default: 'all',
      disabledWhen: state => state.mode !== 'marked',
      options: [
        {value: 'all', label: 'All earthquakes'},
        {value: 'shallow-to-deep', label: 'Shallow → deep'},
        {value: 'deep-to-shallow', label: 'Deep → shallow'},
        {value: 'shallow-to-shallow', label: 'Shallow → shallow'}
      ],
      help: 'Shallow is reported depth <70 km; deep is ≥70 km. Cross-pair curves are directed, while the distance count should agree after reversing the groups.'
    },
    {
      kind: 'toggle',
      id: 'excludeExactTen',
      label: 'Exclude rows reported at exactly 10 km',
      group: 'Marked pairs',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.mode !== 'marked',
      help: 'A sensitivity assumption only. The archive lacks flags that distinguish fixed-depth solutions from rounded reported values; every source point remains mapped.'
    },
    {
      kind: 'select',
      id: 'markedCorrection',
      label: 'Rectangle edge weighting',
      group: 'Marked pairs',
      apply: 'compile',
      default: 'translation',
      disabledWhen: state => state.mode !== 'marked',
      options: [
        {value: 'translation', label: 'Translation correction'},
        {value: 'none', label: 'No edge correction'}
      ],
      help: 'Translation weights pairs by their rectangle-overlap area. None reports the explicitly uncorrected baseline.'
    },
    {
      kind: 'select',
      id: 'intensityModel',
      label: 'Assumed intensity',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 'constant',
      disabledWhen: state => state.mode !== 'inhomogeneous',
      options: [
        {value: 'constant', label: 'Constant n / rectangle area'},
        {value: 'epanechnikov', label: 'Estimated Epanechnikov intensity'}
      ],
      help: 'Compares supplied constant λ=N/area against a kernel estimate of λ(x). Both use the same 802 rows and fixed rectangle.'
    },
    {
      kind: 'slider',
      id: 'intensityBandwidth',
      label: 'Intensity bandwidth',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 150_000,
      min: 75_000,
      max: 300_000,
      step: 75_000,
      unit: 'm',
      format: value => `${value / 1000} km`,
      disabledWhen: state =>
        state.mode !== 'inhomogeneous' || state.intensityModel !== 'epanechnikov',
      help: 'Epanechnikov intensity bandwidth in projected metres. This controls the estimated λ(x), not the pair-distance bins.'
    },
    {
      kind: 'select',
      id: 'selfContribution',
      label: 'Kernel self contribution',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 'exclude',
      disabledWhen: state =>
        state.mode !== 'inhomogeneous' || state.intensityModel !== 'epanechnikov',
      options: [
        {value: 'exclude', label: 'Leave one event out'},
        {value: 'include', label: 'Include focal event'}
      ],
      help: 'Whether each event contributes itself to its own kernel intensity estimate.'
    },
    {
      kind: 'select',
      id: 'inhomogeneousCorrection',
      label: 'Window edge correction',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 'border',
      disabledWhen: state => state.mode !== 'inhomogeneous',
      options: [
        {value: 'border', label: 'Border erosion'},
        {value: 'translation', label: 'Translation overlap'}
      ],
      help: 'Border correction excludes focal points within r of the rectangle edge; translation correction weights overlap of the fixed rectangle.'
    },
    {
      kind: 'select',
      id: 'pairCorrelationEstimator',
      label: 'Pair-correlation estimate',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 'annular',
      disabledWhen: state =>
        state.mode !== 'inhomogeneous' || state.inhomogeneousCorrection !== 'translation',
      options: [
        {value: 'annular', label: 'Annular K differences'},
        {value: 'epanechnikov', label: 'Epanechnikov smoother'}
      ],
      help: 'The annular output is the default difference of K over ring area; the kernel smoother requires translation correction.'
    },
    {
      kind: 'slider',
      id: 'pairCorrelationBandwidth',
      label: 'Pair-correlation kernel half-width',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      min: 10_000,
      max: 25_000,
      step: 15_000,
      default: 10_000,
      unit: 'm',
      format: value => `${value / 1000} km`,
      disabledWhen: state =>
        state.mode !== 'inhomogeneous' ||
        state.inhomogeneousCorrection !== 'translation' ||
        state.pairCorrelationEstimator !== 'epanechnikov',
      help: 'Compact-support kernel half-width in metres; it is a smoothing scale, not a standard deviation.'
    },
    {
      kind: 'select',
      id: 'pairCorrelationDivisor',
      label: 'Kernel divisor',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 'radius',
      disabledWhen: state =>
        state.mode !== 'inhomogeneous' ||
        state.inhomogeneousCorrection !== 'translation' ||
        state.pairCorrelationEstimator !== 'epanechnikov',
      options: [
        {value: 'radius', label: 'Divide by radius'},
        {value: 'distance', label: 'Divide by distance'}
      ],
      help: 'Distance rejects coincident distinct events. The default radius divisor retains them.'
    },
    {
      kind: 'select',
      id: 'zeroCorrection',
      label: 'Kernel zero correction',
      group: 'Inhomogeneous intensity',
      apply: 'compile',
      default: 'none',
      disabledWhen: state =>
        state.mode !== 'inhomogeneous' ||
        state.inhomogeneousCorrection !== 'translation' ||
        state.pairCorrelationEstimator !== 'epanechnikov',
      options: [
        {value: 'none', label: 'None'},
        {value: 'reflection', label: 'Reflection'}
      ],
      help: 'Reflection adds the reflected kernel at the distance origin; none leaves the kernel unchanged.'
    }
  ],
  story: [
    {
      id: 'fixed-window',
      title: 'Start with a fixed support',
      headline: 'The rectangle defines which events count',
      textAlternative:
        'A visible azimuthal-equidistant rectangle around Japan contains every one of 802 selected type-0 earthquake records in archive order; 673 have reported depth below 70 km and 129 are at least 70 km. The rectangle is a study support, not a tectonic or hazard boundary.',
      body: `This map keeps all **802 type-0 earthquake rows** that fall inside one projected rectangle: x = −135 to +135 km and y = −166 to +166 km about [140.5°E, 36.5°N]. Its boundary is drawn explicitly. The projection is spherical azimuthal-equidistant on the mean Earth radius, so planar distances from the origin are useful locally but the rectangle's far edges carry increasing scale distortion.

The archive contains 77,231 rows and includes 16 non-earthquake seismic event types; this scene filters to type 0 before applying the fixed projected support. The snapshot ends 2024-12-30 and does not imply a complete or current catalog.`,
      camera: {...JAPAN_VIEW, transitionMs: 1000},
      options: {mode: 'marked'},
      controls: ['mode'],
      readouts: ['archiveCount', 'windowSupport', 'depthSplit', 'selectedDepth']
    },
    {
      id: 'all-marked-pairs',
      title: 'Give each event its reported depth mark',
      headline: 'Nearby pair structure changes with distance',
      textAlternative:
        'The GPU curve and pair-count chart summarize all events in the rectangle at 16 radii from 5 to 80 km. Reported depth enters the product-mark correlation; K, L and annular pair correlation describe event locations. Terminal-radius numerical outputs and status flags qualify the charts; this is descriptive, not inferential.',
      body: `**GPUMarkedPointStatistics** computes location-based K, L(r) − r, annular pair correlation and cumulative directed non-self pair counts at 5 km steps. Reported depth in kilometres enters the separate product-mark correlation, normalized by the product of the focus and neighbour mean depths.

For the all-event curve this implementation uses the documented **n²** same-type normalizer; pair counts exclude self-pairs. The depth colormap is a quantitative context, and clicking a point reports its archive row and depth in kilometres. Empty annuli appear as chart gaps, not as zero correlation.

The **rectangle edge weighting** control compares translation correction with no correction. No p-value is produced; these are descriptive curves for a time-pooled, dependent event catalog.`,
      options: {mode: 'marked', depthPair: 'all', markedCorrection: 'translation'},
      controls: ['depthPair', 'markedCorrection'],
      readouts: ['pairStatus', 'pairTerminal', 'markedLCurve', 'pairCounts']
    },
    {
      id: 'cross-depth-pairs',
      title: 'Ask whether shallow and deep events pair differently',
      headline: 'Cross-pair curves preserve direction',
      textAlternative:
        'The map retains all 802 event dots while GPU pair statistics use shallow-to-deep depth groups; a selectable reversal swaps focus and neighbour groups. The source has 673 events below 70 km and 129 at or above 70 km; pair curves describe spatial association, not a cause.',
      body: `The event marks split at **70 km**: shallow is reported depth <70 km; deep is ≥70 km. The GPU now estimates ordered shallow-to-deep pairs. Switch **Depth groups** to deep-to-shallow: because pair distances are symmetric, the cumulative pair counts should agree, while focus and neighbour means exchange in the product-mark normalization.

Try shallow-to-shallow to compare within-group structure. The cross K normalization uses n_focus × n_neighbor. Neither version accounts for time dependence, catalog completeness or tectonic mechanism.`,
      options: {mode: 'marked', depthPair: 'shallow-to-deep'},
      controls: ['depthPair', 'excludeExactTen'],
      readouts: ['pairTerminal', 'pairStatus', 'markedLCurve', 'markedMarkCorrelation']
    },
    {
      id: 'exact-depth-sensitivity',
      title: 'Treat exact 10 km values as an assumption',
      headline: 'A reported round depth needs provenance',
      textAlternative:
        'Exactly 10 km records are ringed in amber and remain visible when their optional GPU mask excludes them from marked-pair calculations. The archive has 17 such selected rows and no per-event fixed-depth or uncertainty flags, so this is sensitivity analysis only.',
      body: `There are **17 rows reported at exactly 10 km** in this window, and no coincident coordinate pairs. Their ring is a flag for a possible reporting convention, not evidence that any individual solution was fixed at that depth.

Toggle **Exclude rows reported at exactly 10 km**. The GPU mask changes the admitted marked-pair rows while every source point stays on the map. Compare the curve and pair counts with all rows retained; the archive has no uncertainty or fixed-depth flag to justify treating this as a correction.`,
      options: {mode: 'marked', depthPair: 'all', excludeExactTen: true},
      controls: ['excludeExactTen'],
      readouts: ['exactTenCount', 'pairStatus', 'markedLCurve', 'pairCounts']
    },
    {
      id: 'intensity-model',
      title: 'Change the assumed event intensity',
      headline: 'Constant and smoothed intensity reweight pairs',
      textAlternative:
        'The GPU estimates inhomogeneous K and L(r)−r inside the same prepared rectangle, showing either supplied constant intensity n/area or a kernel estimate. The map overlays valid GPU intensity on depth-colored points; a plug-in estimate describes spatial variation but cannot make dependent time-pooled events independent.',
      body: `Now hold the point set and rectangle fixed while changing the intensity assumption. The **constant baseline** supplies λ = N / rectangle area to **GPUInhomogeneousPointPattern**. The **Epanechnikov model** estimates λ(x) from the event locations, and its bandwidth controls how much spatial variation is treated as intensity rather than pair interaction.

The line chart comes from GPU K and is transformed to L(r) − r. The thin point outlines encode the GPU intensity estimate only when every radius status and every intensity is valid; point fills still show reported depth.`,
      options: {
        mode: 'inhomogeneous',
        intensityModel: 'constant',
        inhomogeneousCorrection: 'border'
      },
      controls: ['intensityModel', 'intensityBandwidth', 'selfContribution'],
      readouts: ['inhomStatus', 'inhomSummary', 'inhomLCurve', 'intensitySummary']
    },
    {
      id: 'correction-sensitivity',
      title: 'Compare border and pair-correlation choices',
      headline: 'An edge rule and smoother change the estimate',
      textAlternative:
        'The same 802 event rows and fixed rectangle feed the GPU inhomogeneous estimator while controls change border erosion versus translation overlap, annular versus Epanechnikov pair correlation, kernel width and divisor. Per-radius status gates every output; these are estimator sensitivities without p-values.',
      body: `**Border erosion** keeps only focal events farther than r from each rectangle edge; **Translation overlap** weights the overlap of the window with its shifted copy. The support-edge readout gives the geometry context for the largest radius.

The pair-correlation output can be the annular difference of K, or an Epanechnikov smoother under translation correction. Change its half-width, radius/distance divisor and zero correction to see how the output changes. Status codes for invalid input, empty erosion, exhausted work or unresolved overlap suppress the curve rather than disguising it.`,
      options: {
        mode: 'inhomogeneous',
        intensityModel: 'epanechnikov',
        intensityBandwidth: 150_000,
        inhomogeneousCorrection: 'translation',
        pairCorrelationEstimator: 'epanechnikov'
      },
      controls: [
        'inhomogeneousCorrection',
        'pairCorrelationEstimator',
        'pairCorrelationBandwidth',
        'pairCorrelationDivisor',
        'zeroCorrection'
      ],
      readouts: ['inhomStatus', 'supportEdge', 'inhomPairCorrelation', 'intensityHistogram']
    },
    {
      id: 'explore',
      title: 'Keep the study window and assumptions visible',
      headline: 'The statistic inherits every choice',
      textAlternative:
        'All 802 selected earthquake records remain mapped within the fixed rectangle, with reported depth, selected event depth, current GPU curves and status summaries available. Controls expose pair direction, exact-10-km sensitivity, intensity, edge correction and smoothing assumptions; no chart supplies a significance test or hazard estimate.',
      body: `Use the controls to move among all-event and directed depth pairs, the exact-10 km sensitivity, and the constant or smoothed intensity models. The same projected window and selected source rows persist throughout.

The catalog is archived and time-pooled, has no event uncertainty or fixed-depth flags, and contains dependent earthquake sequences. These curves describe the selected data and estimator; they do not infer causal tectonics, earthquake abundance or hazard.`,
      options: {mode: 'marked', depthPair: 'all'},
      controls: ['mode', 'depthPair', 'excludeExactTen', 'markedCorrection', 'intensityModel'],
      readouts: ['pairStatus', 'pairSummary', 'inhomStatus', 'inhomSummary']
    }
  ],
  readouts: [
    {id: 'archiveCount', label: 'Archive and source filter'},
    {id: 'windowSupport', label: 'Fixed projected support'},
    {id: 'depthSplit', label: 'Reported depth groups'},
    {id: 'exactTenCount', label: 'Exact 10 km and coordinate duplicates'},
    {id: 'selectedDepth', label: 'Selected event depth'},
    {id: 'pairStatus', label: 'Marked-pair GPU status'},
    {id: 'pairTerminal', label: 'GPU pair values at 80 km'},
    {id: 'pairSummary', label: 'Marked-pair estimator'},
    {id: 'markedLCurve', label: 'Pair L(r) − r', kind: 'chart'},
    {id: 'markedPairCorrelation', label: 'Marked annular pair correlation', kind: 'chart'},
    {id: 'markedMarkCorrelation', label: 'Product-mark correlation', kind: 'chart'},
    {id: 'pairCounts', label: 'Cumulative directed pair counts', kind: 'chart'},
    {id: 'inhomStatus', label: 'Inhomogeneous GPU status'},
    {id: 'inhomTerminal', label: 'GPU inhomogeneous values at 80 km'},
    {id: 'inhomSummary', label: 'Inhomogeneous estimator'},
    {id: 'intensitySummary', label: 'GPU intensity summary'},
    {id: 'supportEdge', label: 'Edge-corrected support'},
    {id: 'inhomLCurve', label: 'Inhomogeneous L(r) − r', kind: 'chart'},
    {id: 'inhomPairCorrelation', label: 'Inhomogeneous pair correlation', kind: 'chart'},
    {id: 'intensityHistogram', label: 'Estimated intensity distribution', kind: 'chart'}
  ]
});
