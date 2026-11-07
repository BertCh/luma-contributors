// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {GLOBAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene} from '../scene';
import type {EarthquakeStandardizationOptions} from './earthquake-standardization.compute';

function isCentered(method: EarthquakeStandardizationOptions['method']): boolean {
  return method === 'deviation-from-mean' || method === 'z-score' || method === 'mad';
}

function methodName(method: EarthquakeStandardizationOptions['method']): string {
  switch (method) {
    case 'deviation-from-mean':
      return 'Deviation from mean';
    case 'z-score':
      return 'Z-score';
    case 'mad':
      return 'Mean absolute deviation';
    case 'range-adjust':
      return 'Range-adjust';
    case 'range-standardize':
      return 'Range-standardize';
  }
}

export default defineScene<EarthquakeStandardizationOptions>({
  id: 'earthquake-standardization',
  title: 'What changes when the scale changes?',
  chapter: 'dataframe',
  order: 12,
  summary:
    'GPUVariableStandardization transforms real ComCat earthquake magnitude or reported depth under three explicit source-derived reference populations. GPUColumnProfile reports the original and transformed distributions; these are statistical transforms, not hazard measures.',
  contributors: ['GPUVariableStandardization', 'GPUColumnProfile'],
  datasets: [
    {
      id: 'poopdeck-earthquakes',
      role: 'Archived USGS ComCat selection of 77,231 M4+ catalog events, 2020 through Dec. 30, 2024'
    }
  ],
  initialView: {longitude: -20, latitude: 20, zoom: 1.25},
  basemap: ground('paperSheet'),
  furniture: {
    ...GLOBAL_FURNITURE,
    title: {
      title: 'What changes when the scale changes?',
      subtitle: 'Five GPU transformations · one explicit reference population',
      sample: '77,215 catalog rows typed as earthquakes',
      chips: ['ComCat archived selection', 'Not hazard analysis']
    },
    credit: 'USGS Earthquake Catalog (ComCat), via poopdeck.gl',
    caveat:
      'This fixed archive is not a completeness claim. Its files omit magnitude-type metadata, so a transform cannot establish cross-type magnitude comparability or earthquake hazard. Reported negative depths are retained.'
  },
  options: [
    {
      kind: 'select',
      id: 'field',
      label: 'Source field',
      group: 'Input',
      apply: 'param',
      default: 'magnitude',
      display: 'segmented',
      options: [
        {value: 'magnitude', label: 'Magnitude'},
        {value: 'depth', label: 'Reported depth'}
      ],
      help: 'Selects the original ComCat magnitude or reported depth in kilometres. This extract does not retain magnitude-type metadata.'
    },
    {
      kind: 'select',
      id: 'referencePopulation',
      label: 'Reference population',
      group: 'Input',
      apply: 'param',
      default: 'all-earthquakes',
      display: 'segmented',
      options: [
        {value: 'all-earthquakes', label: 'All earthquakes'},
        {value: 'magnitude-6-plus', label: 'Magnitude 6+'},
        {value: 'shallow-70-km', label: 'Depth ≤70 km'}
      ],
      help: 'Defines the rows used to estimate the transformation. Negative reported depths remain in the depth-limited population.'
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Transformation',
      group: 'Scale',
      apply: 'param',
      default: 'z-score',
      display: 'segmented',
      options: [
        {value: 'deviation-from-mean', label: 'Deviation'},
        {value: 'z-score', label: 'Z-score'},
        {value: 'mad', label: 'Mean absolute deviation'},
        {value: 'range-adjust', label: 'Range-adjust'},
        {value: 'range-standardize', label: 'Range-standardize'}
      ],
      help: 'Runs the selected transformation on the GPU. “MAD” here means mean absolute deviation about the mean, not median absolute deviation.'
    },
    {
      kind: 'select',
      id: 'variance',
      label: 'Z-score spread',
      group: 'Scale',
      apply: 'param',
      default: 'sample',
      display: 'segmented',
      disabledWhen: state => state.method !== 'z-score',
      options: [
        {value: 'sample', label: 'Sample'},
        {value: 'population', label: 'Population'}
      ],
      help: 'Chooses n−1 or n in the z-score standard deviation. Other transformations do not use this setting.'
    }
  ],
  legends: state => [
    {
      id: 'standardized-values',
      kind: 'ramp',
      title: `${methodName(state.method)} output`,
      ramp: isCentered(state.method) ? 'diverging' : 'viridis',
      extent: 'gpu',
      ...(isCentered(state.method) ? {midpoint: 0, midpointLabel: 'mean'} : {}),
      unit:
        state.method === 'z-score' || state.method === 'mad'
          ? 'scale-relative units'
          : state.method === 'range-standardize'
            ? 'unitless [0, 1]'
            : state.method === 'range-adjust'
              ? 'unitless x / range'
              : 'transformed field units'
    }
  ],
  readouts: [
    {id: 'referenceCount', label: 'Reference rows'},
    {id: 'originalSummary', label: 'Original field', layout: 'block'},
    {id: 'transformedSummary', label: 'Transformed field', layout: 'block'},
    {id: 'activeFormula', label: 'Formula', layout: 'block'},
    {id: 'probeEvent', label: 'GPU probe event', layout: 'block'},
    {id: 'originalDistribution', label: 'Original distribution', kind: 'chart'},
    {id: 'standardizedDistribution', label: 'Transformed distribution', kind: 'chart'}
  ],
  story: [
    {
      id: 'catalog-population',
      title: 'Choose the comparison population',
      headline: 'A transformation depends on which rows define its scale',
      body: 'The default reference is the 77,215 archived records whose event type is “earthquake.” Choose all of them, the 626 magnitude-6-and-larger earthquakes, or the 54,188 earthquakes with reported depth at or below 70 km. The population mask comes from source columns; the transformed output and both histograms are computed on the GPU.',
      textAlternative:
        'A world map shows the archived earthquake catalog with the selected reference population colored by its GPU-transformed value. Readouts and the original and transformed histograms report the same selected rows.',
      evidence: '**{{referenceCount}}**; {{originalSummary}}',
      caveat:
        'The archive is an M4+ selection through Dec. 30, 2024, not a complete or current earthquake catalog. The source contains other seismic event types, which are outside all three reference populations.',
      controls: ['referencePopulation', 'field'],
      readouts: [
        'referenceCount',
        'originalSummary',
        'originalDistribution',
        'standardizedDistribution'
      ],
      camera: {longitude: -20, latitude: 20, zoom: 1.25, transitionMs: 1200}
    },
    {
      id: 'five-scales',
      title: 'Inspect the transformation',
      headline: 'Five named recipes change units in different ways',
      body: 'Compare deviation from mean, z-score, mean absolute deviation, range-adjust, and range-standardize. Here **MAD** is the mean absolute deviation about the mean; it is not the median-based robust statistic often given the same abbreviation. Range-adjust divides x by max−min; range-standardize subtracts the minimum before dividing, placing a nonconstant field on [0, 1]. These are positive affine transforms: they change location and units, not the distribution’s shape or normality.',
      textAlternative:
        'One histogram shows the original selected field and another shows its GPU-transformed values. A world map colors only rows inside the same reference population; a diverging ramp is centered on zero for deviation-based transformations. The transforms change the axis scale but do not make the distribution more normal.',
      evidence: '**{{activeFormula}}**; {{transformedSummary}}',
      caveat:
        'These transformations describe the selected sample distribution. They do not turn magnitudes or depths into calibrated hazard scores.',
      controls: ['method', 'variance'],
      readouts: [
        'activeFormula',
        'transformedSummary',
        'originalDistribution',
        'standardizedDistribution'
      ]
    },
    {
      id: 'reference-changes',
      title: 'Change the reference, keep the same event',
      headline: 'A z-score is relative to its selected population',
      body: 'Keep **Z-score** selected and switch between all earthquakes and magnitude 6+. The marked probe is one fixed source event that belongs to all three reference populations; its source value stays the same while its transformed value is copied from GPU buffers after each population change. The GPU recomputes the reference mean and spread from a different population. Toggle **Sample** and **Population** to compare the n−1 and n variance conventions; the readouts show each output’s observed range and population standard deviation.',
      textAlternative:
        'A dark marker identifies one fixed source row included in every reference population. Its original value stays constant, while its transformed value is copied from GPU buffers and changes with the selected reference mean and spread. Profile summaries update; the selected z-score uses the chosen sample or population convention.',
      evidence: '**{{originalSummary}}**; after transformation **{{transformedSummary}}**.',
      caveat:
        'The selected magnitude-6 population is small relative to the archive. A standardized value is conditional on this extract and the selected reference rows.',
      controls: ['referencePopulation', 'method', 'variance'],
      readouts: ['referenceCount', 'originalSummary', 'transformedSummary', 'probeEvent']
    },
    {
      id: 'limits',
      title: 'Keep the source meaning attached',
      headline: 'Changing scale does not add information to the catalog',
      body: 'Magnitude-type metadata is absent from this bundled extract, so the displayed transform cannot establish that all magnitude values share a scale. Reported negative depths remain in the input and in the shallow-depth reference; they are source values, not rows silently relabeled as missing. The global catalog backdrop is context, while colored points are only members of the selected reference population.',
      textAlternative:
        'The map distinguishes faint archive context from the selected transformed population. A source note states that magnitude-type metadata is unavailable and negative reported depths were retained.',
      evidence: '{{activeFormula}} · **{{referenceCount}}**',
      caveat:
        'This is a descriptive comparison of an archived catalog selection, not an earthquake forecast, hazard assessment or completeness audit.',
      controls: ['field', 'referencePopulation', 'method'],
      readouts: ['referenceCount', 'activeFormula', 'probeEvent']
    }
  ],
  snippet: state => `graph.add(new GPUVariableStandardization({
  values: ${state.field},
  mask: referencePopulation,
  method: '${state.method}',
  variance: '${state.variance}',
  output: transformed
}));
graph.add(new GPUColumnProfile({
  columns: [{values: ${state.field}}, {values: transformed}],
  mask: referencePopulation,
  histogramBinCount: 28,
  output: {statistics, histograms}
}));`,
  about: {
    what: 'Five GPU transformations and GPU-generated profiles/histograms over real ComCat magnitude or reported depth values.',
    why: 'Standardized values depend on both the chosen formula and the rows that define its reference scale.',
    howToRead:
      'The faint map is the full archived context. Colored events belong to the selected earthquake population; the histograms and summaries are GPU outputs for original and transformed values. A dark marker and probe readout identify one included source row. Negative reported depths remain included when they satisfy the source-derived population rule.'
  },
  create: async ctx =>
    (await import('./earthquake-standardization.compute')).createEarthquakeStandardization(ctx)
});
