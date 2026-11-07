// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSchemeColors} from './choropleth-classes.style';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CountyPopulationClassificationOptions} from './county-population-classification.compute';

type PopulationClassLegend = {classCount: number; breaks: number[]; counts: number[]};

const CLASS_COUNT = 5;
const CLASS_COLORS = getSchemeColors('YlOrBr', CLASS_COUNT, 'light');
const countFormat = new Intl.NumberFormat('en-US', {maximumFractionDigits: 0});

export default defineScene<CountyPopulationClassificationOptions>({
  id: 'county-population-classification',
  title: 'How do population counts and Jenks classes tell different stories?',
  chapter: 'statistics',
  order: 2,
  summary:
    'A bounded New Jersey example compares the same 1990 population counts with their GPU Fisher–Jenks class IDs, including exact and approximate Jenks variants over 21 historical county units.',
  contributors: ['GPUClassBreaks', 'GPUClassAssignment', 'GPUColorScale'],
  datasets: [
    {
      id: 'ncovr-new-jersey',
      role: '21 historical county polygons, source FIPS and 1990 population counts'
    }
  ],
  initialView: {longitude: -74.55, latitude: 40.05, zoom: 7.1},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'New Jersey population and classes',
      subtitle: '1990 population counts · bounded 21-unit Jenks comparison',
      chips: ['Observed counts', 'Computed classes']
    },
    scaleBar: {units: 'metric', ticks: [50_000]},
    northArrow: 'always',
    credit: 'NCOVR; Messner et al. (2000); GeoDa archive',
    caveat:
      'Population counts and class membership are different map quantities. The exact Jenks dynamic program is used only for this 21-row support.'
  },
  options: [
    {
      kind: 'select',
      id: 'representation',
      label: 'Map quantity',
      group: 'Display',
      apply: 'param',
      default: 'population-counts',
      display: 'segmented',
      help: 'Population counts use GPUColorScale integerValues:numeric with five equal-width colour bins. Jenks classes use GPUColorScale integerValues:ordinal to look up the actual GPU class ID in the same five-color palette.',
      options: [
        {value: 'population-counts', label: 'Population counts'},
        {value: 'jenks-classes', label: 'Jenks class IDs'}
      ]
    },
    {
      kind: 'select',
      id: 'naturalBreaksMode',
      label: 'Jenks calculation',
      group: 'Jenks',
      apply: 'compile',
      default: 'approximate',
      display: 'segmented',
      help: 'Switches between GPUClassBreaks approximate and exact Fisher–Jenks implementations on these 21 rows. Exact supports at most 1,024 rows; this example is below that cap. Results are compared, not ranked in advance.',
      options: [
        {value: 'approximate', label: 'Approximate'},
        {value: 'exact', label: 'Exact'}
      ]
    }
  ],
  story: [
    {
      id: 'observed-population',
      title: 'Where did New Jersey’s population live in 1990?',
      headline: 'The map begins with measured integer population counts',
      textAlternative:
        'Twenty-one New Jersey historical county units are shaded by their observed 1990 population counts.',
      body: 'The source records 1990 population for all 21 units in GeoDa’s historical New Jersey common-support geography. The count values are integers below 2²⁴, so the GPU numeric color scale can represent every input exactly as float32 after its explicit uint32-to-number conversion. Five equal-width color bins retain the count scale; they are not fitted Jenks classes.',
      options: {representation: 'population-counts', naturalBreaksMode: 'approximate'},
      optionsMode: 'fresh',
      controls: ['representation'],
      readouts: ['sourceSummary', 'colorSummary'],
      caveat:
        'These are source population counts from 1990, not current population estimates or population density.'
    },
    {
      id: 'jenks-classes',
      title: 'How does Jenks group the same 21 counts?',
      headline: 'GPU Jenks assigns each unit to a count interval',
      textAlternative:
        'The map colors each New Jersey county unit by its GPU-assigned Fisher–Jenks class ID; a chart shows the returned number of units in each class.',
      body: '**`GPUClassBreaks`** computes five natural-break edges from the 21 source count values. **`GPUClassAssignment`** writes one class ID per unit. In this view, **`GPUColorScale`** consumes those integer IDs as ordinal palette indices; the displayed classes are not population counts. Compare exact and approximate Jenks to inspect the actual returned thresholds and assignments.',
      options: {representation: 'jenks-classes', naturalBreaksMode: 'exact'},
      optionsMode: 'fresh',
      controls: ['naturalBreaksMode'],
      readouts: ['jenksSummary', 'assignmentSummary', 'classCounts'],
      caveat:
        'Jenks is a descriptive partition of these values. A lower within-class sum of squares does not make one classification universally preferable or inferential.'
    },
    {
      id: 'two-map-quantities',
      title: 'Which question should the colors answer?',
      headline: 'Counts and classes share a palette but not an estimand',
      textAlternative:
        'A control switches between the numeric 1990 population count and its GPU-assigned Jenks class membership.',
      body: 'The **Map quantity** control switches between two intentionally different data types. Numeric mode sends the observed uint32 counts directly to `GPUColorScale` with `integerValues: numeric`, where the scale converts integers to exact float32 values and colors five equal-width intervals. Class mode sends the GPU-returned uint32 class IDs with `integerValues: ordinal`, where each code indexes one palette entry. Neither representation is an alternative estimate of the other.',
      options: {representation: 'jenks-classes', naturalBreaksMode: 'exact'},
      optionsMode: 'fresh',
      controls: ['representation', 'naturalBreaksMode'],
      readouts: ['sourceSummary', 'jenksSummary', 'assignmentSummary', 'classCounts'],
      caveat:
        'The exact-versus-approximate comparison concerns only the Jenks break calculation. The population-count view remains a separate numeric scale.'
    }
  ],
  legends: (state, data) => {
    const populationLegend = data['population-counts'] as
      | {minimumPopulation: number; maximumPopulation: number}
      | undefined;
    if (!populationLegend) {
      return [
        {
          kind: 'categories',
          title: '1990 population',
          entries: [],
          note: 'Loading source counts...'
        }
      ];
    }
    const extent = [
      populationLegend.minimumPopulation,
      populationLegend.maximumPopulation
    ] as const;
    if (state.representation === 'population-counts') {
      const breaks = Array.from(
        {length: CLASS_COUNT - 1},
        (_value, index) => extent[0] + ((extent[1] - extent[0]) * (index + 1)) / CLASS_COUNT
      );
      return [
        {
          kind: 'classes',
          title: '1990 population · numeric uint32 values',
          unit: 'people',
          colors: CLASS_COLORS,
          breaks,
          extent,
          layout: 'bar',
          format: value => countFormat.format(value),
          note: 'Five equal-width numeric bins; colors come from the same YlOrBr palette as class membership.'
        }
      ];
    }
    const legend = data['population-classes'] as PopulationClassLegend | undefined;
    return [
      {
        kind: 'classes',
        title: 'GPU Jenks class IDs · ordinal uint32 values',
        unit: 'people',
        colors: CLASS_COLORS,
        breaks: legend?.breaks ?? [],
        extent,
        counts: legend?.counts,
        layout: 'list',
        format: value => countFormat.format(value),
        note: legend
          ? `${state.naturalBreaksMode} Fisher–Jenks on 21 rows; each palette index is a computed class ID.`
          : 'GPU Jenks breaks and class counts are pending.'
      }
    ];
  },
  readouts: [
    {id: 'sourceSummary', label: 'Population source support'},
    {id: 'colorSummary', label: 'Integer color-scale contract'},
    {id: 'jenksSummary', label: 'GPU Jenks edges'},
    {id: 'assignmentSummary', label: 'GPU class assignments'},
    {id: 'assignmentEvidence', label: 'Per-unit IDs and packed GPU colors'},
    {id: 'classCounts', label: 'Units by class', kind: 'chart'}
  ],
  about: {
    what: 'A 21-unit New Jersey historical county set maps observed 1990 population counts and GPU Fisher–Jenks class membership.',
    why: 'The scene distinguishes numeric integer color scaling from ordinal integer class lookup and makes exact Jenks feasible on a bounded support.',
    howToRead:
      'Count mode maps actual uint32 population numbers through a numeric scale. Class mode maps GPUClassAssignment output IDs as ordinal palette indices. Exact Jenks is bounded to 21 rows here; the scene does not claim that the exact partition is the preferred interpretation.'
  },
  create: async ctx =>
    (
      await import('./county-population-classification.compute')
    ).createCountyPopulationClassification(ctx)
});
