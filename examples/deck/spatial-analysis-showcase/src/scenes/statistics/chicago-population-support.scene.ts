// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {ChicagoPopulationSupportOptions} from './chicago-population-support.compute';

export default defineScene<ChicagoPopulationSupportOptions>({
  id: 'chicago-population-support',
  title: 'How do support assumptions change a tract population estimate?',
  chapter: 'statistics',
  order: 118,
  summary:
    'Follow 12 observed Hyde Park tract counts through exact ID alignment, two 500 m allocation assumptions, and a mapped tree-cover summary.',
  contributors: [
    'addPopulationSupportAlignmentRecipe',
    'addDasymetricRedistributionRecipe',
    'addExposureByZoneRecipe'
  ],
  datasets: [
    {id: 'chicago-tracts', role: 'source Census tract geometries and SVI 2022 count context'},
    {
      id: 'chicago-population-support',
      role: '12 tract population groups and fractional WorldCover indicators'
    }
  ],
  initialView: {longitude: -87.591, latitude: 41.794, zoom: 12.7},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Hyde Park population support',
      subtitle: 'Observed tract counts · modeled 500 m allocations',
      sample: '12 whole tracts · SVI 2022 / WorldCover 2021',
      chips: ['Allocation assumptions', 'Not a measured population surface']
    },
    scaleBar: {units: 'metric', ticks: [500]},
    northArrow: 'always',
    credit:
      'US Census Bureau (public domain); CDC/ATSDR SVI 2022 (public domain); © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data 2021 processed by WorldCover consortium',
    caveat:
      'Population counts are tract estimates. Built-up cover does not identify housing; mapped tree cover does not measure access, heat, or health.'
  },
  options: [
    {
      id: 'display',
      label: 'Map display',
      kind: 'select',
      display: 'segmented',
      default: 'source-counts',
      apply: 'param',
      options: [
        {value: 'source-counts', label: 'Observed tract counts'},
        {value: 'allocation', label: 'Modeled allocation'},
        {value: 'built-up-fraction', label: 'Input built-up fraction'},
        {value: 'tree-cover', label: 'Tree-cover summary'}
      ],
      help: 'Choose a map layer: observed tract counts, modeled allocations, the actual 64 × 64 WorldCover built-up input, or mapped tree cover.'
    },
    {
      id: 'support',
      label: 'Allocation support',
      kind: 'select',
      display: 'segmented',
      default: 'built-up',
      apply: 'param',
      options: [
        {value: 'uniform-area', label: 'Uniform area'},
        {value: 'built-up', label: 'Built-up fraction'}
      ],
      help: 'This choice controls the ancillary weights used to compute allocations and tree-cover population summaries.'
    },
    {
      id: 'treeThreshold',
      label: 'Low tree-cover threshold',
      kind: 'select',
      display: 'segmented',
      default: '25',
      apply: 'param',
      options: [
        {value: '10', label: 'At most 10%'},
        {value: '25', label: 'At most 25%'},
        {value: '50', label: 'At most 50%'}
      ],
      help: 'Counts modeled residents in 500 m cells at or below this mapped WorldCover tree-cover fraction.'
    }
  ],
  story: [
    {
      id: 'source-counts',
      title: 'What do the twelve whole source tracts report?',
      headline: 'Observed counts stay at the tract scale first',
      body: 'The map and chart show observed SVI 2022 population counts for all 12 whole tracts assigned to Hyde Park by representative point. Their total is 29,559; the count colors use one fixed scale.',
      textAlternative:
        'Twelve complete tract polygons are colored by observed population count, before any cell allocation is shown.',
      evidence:
        'The observed source sum is **{{sourcePopulation}}** across **{{tractCount}}** tracts.',
      caveat: 'The source values are tract estimates, not individual locations.',
      controls: ['display'],
      readouts: ['tractCount', 'sourcePopulation', 'targetCells']
    },
    {
      id: 'identifier-alignment',
      title: 'Do shuffled count rows still match the tract geometries?',
      headline: 'Exact IDs restore the source geometry order',
      body: 'Two count groups—non-Hispanic Black residents and everyone else—are shuffled, then joined by the six-digit tract code inside Cook County namespace 17031. The full GEOID remains in this aligned ledger.',
      textAlternative:
        'The compact ledger lists aligned group counts in geometry order and reports the exact identifier join status.',
      evidence:
        'The GPU join reports **{{alignmentStatus}}**; the aligned groups sum to **{{sourcePopulation}}**.',
      caveat: 'Race counts use the CDC/ATSDR SVI 2022 ACS 2018–2022 five-year source.',
      controls: ['display'],
      options: {display: 'source-counts'},
      readouts: ['alignmentStatus', 'alignedCounts', 'recipeStatus']
    },
    {
      id: 'uniform-area',
      title: 'What if each tract count follows uniform area support?',
      headline: 'Uniform weights include every mapped area cell',
      body: 'Choose **Allocation support** below to use uniform weights across every valid cell, including mapped water. Choose **Modeled allocation** in Map display to inspect returned GPU counts in complete 500 m cells.',
      textAlternative:
        'A cividis raster displays returned GPU counts per 500 m cell with tract boundaries as outlines only.',
      evidence:
        'The GPU ledger allocates **{{allocatedPopulation}}**, retains **{{unallocatedPopulation}}**, and sums **{{targetPopulation}}** in target cells.',
      caveat:
        'Uniform area gives mapped water the same weight as other cells; the result is a deliberately simple baseline.',
      controls: ['display', 'support'],
      options: {display: 'allocation', support: 'uniform-area'},
      readouts: ['supportChoice', 'targetCells', 'unallocatedPopulation', 'massBalance']
    },
    {
      id: 'built-up-support',
      title: 'How does mapped built-up support change the allocation?',
      headline: 'Built-up fractions shift modeled cell totals',
      body: 'Allocation support uses the actual WorldCover class-50 fraction measured on 62.5 m cells. Choose **Input built-up fraction** to inspect those 64 × 64 weights on a 0–1 scale, or **Modeled allocation** to inspect returned GPU target counts on a fixed 0–3,500 scale.',
      textAlternative:
        'A cividis raster shows only modeled target counts; source tract boundaries remain as outlines.',
      evidence:
        'The mass ledger reports **{{targetPopulation}}** target residents and **{{coverageStatus}}** coverage.',
      caveat:
        'Built-up cover does not establish residential use or occupancy. A tract with zero eligible support would retain its entire mass.',
      controls: ['display', 'support'],
      options: {display: 'built-up-fraction', support: 'built-up'},
      readouts: ['supportChoice', 'targetCells', 'coverageStatus', 'supportLedger']
    },
    {
      id: 'tree-cover-summary',
      title: 'How many modeled residents fall in low tree-cover cells?',
      headline: 'Tree-cover counts inherit the population support assumption',
      body: 'Choose **Allocation support** to compare uniform-area and built-up-weighted population estimates. The map shows actual 500 m tree-cover fractions with quadrant boundaries; the threshold counts returned modeled residents in cells at or below the selected fraction.',
      textAlternative:
        'The map shows mapped tree fraction; the chart reports GPU population-weighted low-tree-cover counts and coverage by quadrant.',
      evidence:
        'The GPU exposure recipe reports **{{lowTreePopulation}}** with **{{treeCoverage}}** observed hazard coverage.',
      caveat:
        'This is population allocated to a land-cover class, not heat, health risk, public greenspace, or access.',
      controls: ['display', 'support', 'treeThreshold'],
      options: {display: 'tree-cover', support: 'built-up', treeThreshold: '25'},
      readouts: ['supportChoice', 'targetCells', 'lowTreePopulation', 'treeCoverage']
    },
    {
      id: 'limits',
      title: 'Which limits matter before interpreting the fine cells?',
      headline: 'A complete ledger does not turn allocations into observations',
      body: 'Compare the observed tract map, two support assumptions, and tree-cover summary. The compact status and mass ledger make identifier, coverage, and capacity checks visible.',
      textAlternative:
        'The complete GPU ledger and declared source limitations summarize how far the modeled estimates can be interpreted.',
      evidence:
        'The final run reports **{{recipeStatus}}** with a source-minus-allocation residual of **{{massBalance}}**.',
      caveat:
        'Population estimates span 2018–2022; WorldCover is a 2021 classification with its own error. Fractional support is not occupancy.',
      controls: ['display', 'support'],
      options: {display: 'tree-cover', support: 'built-up'},
      readouts: ['recipeStatus', 'massBalance', 'alignmentStatus', 'lowTreePopulation']
    }
  ],
  readouts: [
    {id: 'tractCount', label: 'Whole source tracts', format: 'integer'},
    {id: 'sourcePopulation', label: 'Observed tract residents', format: 'integer'},
    {id: 'alignmentStatus', label: 'Exact ID alignment', format: 'text'},
    {
      id: 'alignedCounts',
      label: 'Aligned Black / other counts by full GEOID',
      format: 'text',
      layout: 'block'
    },
    {id: 'allocatedPopulation', label: 'GPU allocated source residents', format: 'integer'},
    {
      id: 'unallocatedPopulation',
      label: 'GPU retained without eligible support',
      format: 'integer'
    },
    {id: 'targetPopulation', label: 'GPU target-cell resident sum', format: 'integer'},
    {id: 'supportChoice', label: 'Ancillary weights used', format: 'text'},
    {id: 'massBalance', label: 'Source minus allocated and retained', format: 'decimal'},
    {id: 'validSourceCount', label: 'Sources with positive eligible support', format: 'integer'},
    {id: 'supportLedger', label: 'Eligible support and transferred source mass', format: 'text'},
    {id: 'coverageStatus', label: 'Coverage tuples / required · overflow', format: 'text'},
    {id: 'pairStatus', label: 'Required source-target pairs · overflow', format: 'text'},
    {
      id: 'lowTreePopulation',
      label: 'Estimated residents in low mapped tree-cover cells',
      format: 'text'
    },
    {id: 'treeCoverage', label: 'Tree-cover observation coverage by quadrant', format: 'text'},
    {id: 'recipeStatus', label: 'GPU result status', format: 'text'},
    {
      id: 'targetCells',
      label: 'Observed counts, modeled allocations, or tree-cover summary',
      kind: 'chart'
    }
  ],
  legends: state =>
    state.display === 'source-counts'
      ? [
          {
            kind: 'ramp',
            id: 'population',
            title: 'Observed residents per source tract',
            ramp: 'cividis',
            extent: [0, 4000],
            unit: 'residents'
          }
        ]
      : state.display === 'tree-cover' || state.display === 'built-up-fraction'
        ? [
            {
              kind: 'ramp',
              id: 'population',
              title:
                state.display === 'tree-cover'
                  ? 'Mapped WorldCover tree-cover fraction'
                  : 'Input WorldCover built-up fraction · 62.5 m cells',
              ramp: 'cividis',
              extent: [0, 1],
              unit: 'fraction'
            }
          ]
        : [
            {
              kind: 'ramp',
              id: 'population',
              title: 'Modeled residents per 500 m cell',
              ramp: 'cividis',
              extent: [0, 3500],
              unit: 'residents',
              note: 'Returned GPU counts; source outlines are shown without source fills.'
            }
          ],
  snippet: state => `const aligned = addPopulationSupportAlignmentRecipe(graph, {
  populations: shuffledBlackAndOtherCounts,
  sourceIds: shuffledSixDigitTractIds,
  targetIds: geometryOrderTractIds
});
const result = addDasymetricRedistributionRecipe(graph, {
  sourceTotals: sumAlignedGroups(aligned.outputs.populations),
  ancillary: ${state.support === 'built-up' ? 'worldCoverBuiltUpFraction' : 'uniformOneWeights'}
});`,
  about: {
    what: 'Aligns actual tract count groups, redistributes tract totals under explicit fine-support weights, then summarizes modeled counts against mapped tree cover.',
    why: 'The sequence separates observed source counts from assumptions used to represent them at 500 m support.',
    howToRead:
      'Cividis uses distinct fixed scales for observed tract counts, modeled cell counts, and mapped tree fractions. The exposure summary is an estimate for low mapped tree-cover cells only.'
  },
  create: async context =>
    (await import('./chicago-population-support.compute')).createChicagoPopulationSupport(context)
});
