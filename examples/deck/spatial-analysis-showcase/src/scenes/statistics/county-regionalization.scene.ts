// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {ground} from '../../cartography/grounds';
import type {CountyRegionalizationOptions} from './county-regionalization.compute';

const REGION_PALETTE = [
  [0, 114, 178, 230],
  [213, 94, 0, 230],
  [0, 158, 115, 230],
  [204, 121, 167, 230],
  [230, 159, 0, 230],
  [86, 180, 233, 230],
  [51, 51, 51, 230],
  [0, 127, 127, 230],
  [231, 111, 81, 230],
  [38, 70, 83, 230],
  [138, 154, 60, 230],
  [106, 76, 147, 230],
  [196, 69, 105, 230],
  [140, 86, 75, 230],
  [49, 163, 84, 230]
] as const;

export default defineScene<CountyRegionalizationOptions>({
  id: 'county-regionalization',
  title: 'Which neighboring counties form similar groups?',
  chapter: 'statistics',
  order: 90,
  summary:
    'Compare seeded AZP, sequential and mutual-nearest constrained Ward, and population-threshold Max-P partitions on measured 1990 attributes for the 21 NCOVR New Jersey county units.',
  contributors: ['GPUAZPRegions', 'GPUWardRegions', 'GPUMaxPRegions'],
  datasets: [
    {
      id: 'ncovr-new-jersey',
      role: 'common-support county polygons, measured attributes and 1990 population counts'
    }
  ],
  initialView: {longitude: -74.55, latitude: 40.05, zoom: 7.1},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'New Jersey county regionalization',
      subtitle: 'Algorithm-authored partitions of measured attributes',
      sample: '21 common-support units · 1990 attributes',
      chips: ['Exploratory']
    },
    scaleBar: {units: 'metric'},
    credit: 'NCOVR; Messner et al. (2000); GeoDa archive',
    caveat: 'Partitions summarize selected attributes; they are not administrative recommendations.'
  },
  options: [
    {
      kind: 'select',
      id: 'algorithm',
      label: 'Regionalization method',
      apply: 'param',
      default: 'azp',
      group: 'Partition',
      options: [
        {value: 'azp', label: 'AZP · local improvement'},
        {value: 'ward-sequential', label: 'Ward · sequential merges'},
        {value: 'ward-mutual-nearest', label: 'Ward · mutual-nearest rounds'},
        {value: 'max-p', label: 'Max-P · population floor'}
      ],
      help: 'AZP refines six connected seed regions, Ward starts from county singletons and merges adjacent regions to six, and Max-P grows as many regions as can meet the extensive population floor.'
    },
    {
      kind: 'slider',
      id: 'minimumPopulation',
      label: 'Minimum population per region',
      apply: 'param',
      min: 250000,
      max: 1000000,
      step: 50000,
      default: 500000,
      unit: 'people',
      group: 'Max-P constraint',
      help: 'Max-P uses actual 1990 county population counts as an extensive sum. Raising the floor can reduce feasible region count or leave under-threshold regions.'
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Tie-order seed',
      apply: 'param',
      min: 0,
      max: 999,
      step: 1,
      default: 17,
      group: 'Reproducibility',
      help: 'Changes deterministic tie order for AZP, Ward and Max-P. A different seed may change a heuristic solution; it does not turn the result into an optimum.'
    }
  ],
  story: [
    {
      id: 'measured-attributes',
      title: 'The inputs are measured, comparable county attributes',
      headline: 'Three source features are standardized before clustering',
      body: 'Each county unit is described by 1990 unemployment, 1989 family poverty and 1990 source log population density. These three columns are standardized across the 21 units before measuring within-region similarity.',
      textAlternative:
        'A map of 21 historical New Jersey county units colored by GPU-produced nominal region labels from the selected spatial regionalization method.',
      evidence:
        'The fill shows GPU-produced region labels; hover a county unit for its source rates and population.',
      caveat:
        'Standardized similarity gives each selected feature equal scale. It does not encode population weighting or imply a social boundary.',
      controls: [],
      readouts: ['regionCount']
    },
    {
      id: 'azp-seeded-refinement',
      title: 'AZP improves a connected six-region starting partition',
      headline: 'AZP makes contiguous local moves from six seed regions',
      body: 'The **Regionalization method** control below selects **GPUAZPRegions**, which starts from six connected seed groups and applies local contiguous moves that reduce within-region squared deviation. Change the **Tie-order seed** below to inspect heuristic sensitivity.',
      textAlternative:
        'Six connected seed regions are colored by the AZP label output after local moves; a readout reports the number of labels and convergence status.',
      evidence:
        'The readout combines GPU labels, objective and status words with checks against source queen adjacency and population counts.',
      caveat:
        'AZP returns a local solution under its move rules; it is not a global optimum, administrative recommendation or ground-truth map.',
      options: {algorithm: 'azp'},
      controls: ['algorithm', 'seed'],
      readouts: ['regionCount', 'partitionQuality', 'partitionCertificate', 'minimumStatus']
    },
    {
      id: 'ward-merge-schedules',
      title: 'Ward schedules can stop at different partitions',
      headline: 'Sequential and batched merge schedules may differ',
      body: '**GPUWardRegions** begins with one region per county and merges adjacent pairs by their Ward increase until six remain. Choose **Ward · sequential merges** or **Ward · mutual-nearest rounds** under **Regionalization method** below.',
      textAlternative:
        'Twenty-one initial singleton polygons are joined into six contiguous regions under one selected Ward merge schedule; the map and status reflect that schedule.',
      evidence:
        'Each variant reports its own GPU labels, within-region objective, full status and connected-component check.',
      caveat:
        'Mutual-nearest is a batched schedule. With contiguity constraints and a fixed cut, it can return a different valid partition than sequential merging.',
      options: {algorithm: 'ward-sequential'},
      controls: ['algorithm'],
      readouts: ['regionCount', 'partitionQuality', 'partitionCertificate', 'minimumStatus']
    },
    {
      id: 'population-feasibility',
      title: 'Max-P makes population mass an explicit constraint',
      headline: 'Every Max-P region must meet a resident-count floor',
      body: '**GPUMaxPRegions** grows contiguous regions until each reaches a minimum sum of 1990 residents. Adjust **Minimum population per region** below to see how a substantive extensive constraint affects the partition.',
      textAlternative:
        'County units are colored by Max-P growth-region labels; the status distinguishes a fully feasible population floor from regions that remain below it.',
      evidence:
        'The GPU status reports count, threshold feasibility, below-threshold regions, invalid units, selected start and improvement moves; readback sums actual 1990 population for each returned label.',
      caveat:
        'A threshold can be infeasible on a given adjacency graph. Under-threshold or unassigned mass must remain visible as a status limitation, not be read as a successful region.',
      options: {algorithm: 'max-p'},
      controls: ['algorithm', 'minimumPopulation', 'seed'],
      readouts: ['regionCount', 'partitionQuality', 'partitionCertificate', 'minimumStatus']
    },
    {
      id: 'algorithm-authored',
      title: 'Treat these boundaries as algorithm-authored partitions',
      headline: 'These clusters are exploratory summaries, not policy borders',
      body: 'Compare the partitions as alternative summaries of measured attributes. The **Regionalization method** and **Tie-order seed** controls below let you test method and tie-order sensitivity.',
      textAlternative:
        'The selected algorithm’s nominal region labels fill the historical county polygons; they are shown as exploratory groups rather than recommended service areas.',
      evidence:
        'Each map is a current contributor output over the same 21 source units and queen-contiguous graph.',
      caveat:
        'No algorithm here optimizes public services, political representation or a stated administrative objective. These are exploratory partitions, not recommendations.',
      controls: ['algorithm', 'seed'],
      readouts: ['regionCount', 'partitionQuality', 'partitionCertificate', 'minimumStatus']
    }
  ],
  legends: (state, data) => [
    {
      kind: 'categories',
      title: 'Current returned region labels',
      entries: (Array.isArray(data['regionLabels']) ? data['regionLabels'] : [])
        .filter((label): label is number => typeof label === 'number')
        .map(label => ({
          color: REGION_PALETTE[label % REGION_PALETTE.length],
          label: `Label ${label + 1}`
        })),
      note:
        state.algorithm === 'max-p'
          ? 'Max-P labels are construction IDs; the population floor status determines whether all regions are feasible.'
          : 'Each currently returned label has a distinct palette color. Labels identify an algorithmic partition, not ranked classes.'
    }
  ],
  readouts: [
    {
      id: 'regionCount',
      label: 'Returned region labels',
      format: 'integer',
      help: 'Count of distinct labels read from the selected contributor output.'
    },
    {
      id: 'partitionQuality',
      label: 'Within-region objective',
      format: 'decimal',
      help: 'GPU-reported within-region sum of squared deviations in standardized feature units.'
    },
    {
      id: 'partitionCertificate',
      label: 'Partition check',
      format: 'text',
      help: 'Readback checks assigned rows, connected components in the exact queen graph, and actual 1990 population minima and maxima.'
    },
    {
      id: 'minimumStatus',
      label: 'Partition status',
      format: 'text',
      help: 'AZP/Ward words are iterations, converged, iteration-limit flag and invalid count; Ward adds merge rounds. Max-P words are region count, feasibility, infeasible-region count, invalid count, selected start and improvement moves. Convergence is not proof of a global optimum.'
    }
  ],
  snippet:
    state => `// All three measured attributes are standardized; population remains an extensive count.
new ${state.algorithm === 'max-p' ? 'GPUMaxPRegions' : state.algorithm.startsWith('ward') ? 'GPUWardRegions' : 'GPUAZPRegions'}({
  weights: historicalQueenContiguity,
  values: [unemployment1990, familyPoverty1989, logDensity1990],
  ${state.algorithm === 'max-p' ? `extensiveValues: population1990, threshold: ${state.minimumPopulation},` : ''}
  parameters: {seed: ${state.seed}}
});`,
  about: {
    what: 'Four GPU regionalization outputs compare seeded AZP refinement, two Ward merge schedules and multi-start Max-P constrained by actual 1990 population mass.',
    why: 'The algorithms expose different construction logic: local moves from contiguous seeds, agglomerative contiguity-constrained Ward merges, and growth subject to a minimum extensive total.',
    howToRead:
      'Each county fill is colored by the selected output label; unassigned rows are nodata. Objective and partition certificate come from GPU outputs plus readback checks. GPURegionalization standardizes the three input features once. 1990 population is used only for Max-P feasibility. The map uses archive historical common-support units and queen adjacency. Convergence does not establish a global optimum.'
  },
  create: async ctx =>
    (await import('./county-regionalization.compute')).createCountyRegionalization(ctx)
});
