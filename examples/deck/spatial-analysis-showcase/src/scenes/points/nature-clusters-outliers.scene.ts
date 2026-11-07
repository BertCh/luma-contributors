// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {formatCount} from '../../cartography/live-text';
import {defineScene, type LegendSpec} from '../scene';
import {formatCategory} from './b1-nature-data';
import {POINTS_CREDITS, pointsCartouche} from './b1-points-look';
import {NATURE_OUTLIER_SAMPLE_LIMIT} from './nature-clusters-outliers-analysis';
import type {NatureOutlierOptions} from './nature-clusters-outliers.compute';

const CATEGORIES = [
  'Birds',
  'Plants',
  'Insects',
  'Fungi',
  'Mammals',
  'Spiders and kin',
  'Amphibians and reptiles'
];

export default defineScene<NatureOutlierOptions>({
  id: 'nature-clusters-outliers',
  title: 'Which records sit in locally sparse pockets?',
  chapter: 'points',
  order: 3,
  summary:
    'Local Outlier Factor compares each iNaturalist record with its selected k nearest records. A high score marks a locally sparse pocket in the reporting pattern, not an unusual animal or a statistically significant outlier.',
  contributors: ['GPUNeighborSearch', 'GPULocalOutlierFactor'],
  datasets: [{id: 'chicago-nature', role: 'iNaturalist observations (2023), one group at a time'}],
  initialView: {...CITY_FRAMES.chicago},
  options: [
    {
      kind: 'select',
      id: 'category',
      label: 'Group',
      group: 'Data',
      apply: 'compile',
      default: 'Birds',
      help: `Every selected group is sampled at evenly spaced source rows, up to ${formatCount(NATURE_OUTLIER_SAMPLE_LIMIT)} records. The deterministic bounded sample keeps this exact GPU k-nearest-neighbour example tractable.`,
      options: CATEGORIES.map(name => ({value: name, label: formatCategory(name)}))
    },
    {
      kind: 'select',
      id: 'view',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      display: 'segmented',
      default: 'score',
      help: 'Show the selected raw records or their GPU Local Outlier Factor scores. The analysis still runs either way.',
      options: [
        {value: 'records', label: 'Records'},
        {value: 'score', label: 'LOF score'}
      ]
    },
    {
      kind: 'slider',
      id: 'neighborCount',
      label: 'Nearest records (k)',
      group: 'Local comparison',
      apply: 'compile',
      min: 10,
      max: 30,
      step: 1,
      default: 10,
      format: value => `k = ${value}`,
      help: 'LOF compares local reachability density across this many nearest records. The GPU kNN table and LOF graph are rebuilt when k changes; compare scores at the same k. If a group has fewer records than k, all available neighbours are used. Equal distances at the k-th neighbour are resolved by source-row order.'
    },
    {
      kind: 'slider',
      id: 'threshold',
      label: 'Flag above LOF',
      group: 'Local comparison',
      apply: 'param',
      min: 1.1,
      max: 4,
      step: 0.1,
      default: 1.5,
      format: value => value.toFixed(1),
      help: 'The map keeps showing the continuous GPU score. This optional cut only counts rows above it; LOF has no null distribution, so the cut is exploratory, not a significance threshold.'
    }
  ],
  readouts: [
    {id: 'sampleSize', label: 'Records sampled'},
    {id: 'categoryLabel', label: 'Observation group'},
    {id: 'neighborCount', label: 'Nearest records used'},
    {id: 'flagged', label: 'Records above the chosen score'},
    {id: 'medianScore', label: 'Median local outlier factor'},
    {id: 'maximumScore', label: 'Highest sampled score'},
    {
      id: 'neighborOverflow',
      label: 'Neighbour table capacity',
      help: 'The output table reserves k slots per sampled record. Any truncation is shown here; equal distances at the k-th neighbour are resolved by source-row order.'
    },
    {
      id: 'scoreDistribution',
      label: 'Local outlier factor scores',
      kind: 'chart',
      help: 'GPU LOF scores across the deterministic sample. The marker is the optional exploratory cut; the last bin includes scores of 4 or more.'
    }
  ],
  legends: state =>
    state.view === 'records'
      ? []
      : [
          {
            kind: 'ramp',
            id: 'lof-score',
            title: 'Local outlier factor',
            ramp: 'oranges',
            extent: [0, 4],
            labels: ['0', '4+'],
            unit: 'score'
          } satisfies LegendSpec
        ],
  about: {
    what: '`GPUNeighborSearch` builds a k-nearest-neighbour table, then `GPULocalOutlierFactor` compares each record’s local reachability density with those of its neighbours. Scores near 1 have similar local density; higher scores are locally sparser.',
    why: 'A single DBSCAN radius asks where records form connected dense groups; Local Outlier Factor asks which records sit in a sparse pocket relative to nearby records. It is a different description of the observation process, not a claim about wildlife abundance or unusual organisms.',
    howToRead:
      'The map and histogram show continuous scores for a deterministic, evenly spaced sample of at most 1,200 source rows. A 10 m density floor keeps coincident coordinates finite. The threshold only counts scores above a user-chosen value. Observers choose where and what to record, so a high score may reflect reporting coverage, access or identification practices.'
  },
  pipeline: [
    {
      id: 'sample',
      label: 'Sample',
      detail: 'Choose evenly spaced source rows within one observation group'
    },
    {
      id: 'neighbors',
      label: 'Neighbours',
      detail: 'Find the selected k closest records for each sample row'
    },
    {
      id: 'density',
      label: 'Local density',
      detail: 'Compare reachability density with each row’s neighbours'
    },
    {id: 'score', label: 'LOF score', detail: 'Write one local sparsity ratio per sampled record'}
  ],
  basemap: ground('paperCity'),
  furniture: {
    title: pointsCartouche(
      'Where is the reporting pattern locally sparse?',
      'LOF over iNaturalist records'
    ),
    scaleBar: {units: 'metric'},
    credit: POINTS_CREDITS.nature,
    caveat: 'A sparse pocket in the records is not evidence of sparse wildlife.'
  },
  annotations: labelsFor(CHICAGO, ['lake-michigan', 'loop', 'montrose-point']),
  create: async ctx =>
    (await import('./nature-clusters-outliers.compute')).createNatureOutliers(ctx),
  story: [
    {
      id: 'sample',
      title: 'A bounded sample',
      headline: 'A sample of records sketches observer coverage',
      textAlternative:
        'Chicago map of evenly spaced source rows from one nature observation group, drawn as small blue ink dots.',
      body: 'This is a bounded, deterministic sample of **{{sampleSize}}** **{{categoryLabel}}** records. It chooses evenly spaced source rows so an exact GPU k-nearest-neighbour calculation can compare each record with the surrounding reporting pattern. The dots follow observers and access; they are not a count of animals.',
      optionsMode: 'fresh',
      options: {category: 'Birds', view: 'records', neighborCount: 10, threshold: 1.5},
      controls: ['category'],
      readouts: ['sampleSize', 'categoryLabel'],
      stage: 'sample',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {
        title: pointsCartouche('Which records sit in locally sparse pockets?', 'Observation sample')
      }
    },
    {
      id: 'local-density-score',
      title: 'Compare local density',
      headline: 'LOF compares a record with nearby records',
      textAlternative:
        'The sampled records now use an orange score ramp and a histogram of Local Outlier Factor values.',
      body: '`GPULocalOutlierFactor` compares each record with **{{neighborCount}}** nearest records. A score near **1** has similar local reachability density; a higher score is locally sparser. The sample median is **{{medianScore}}**. Equal distances at the k-th neighbour are resolved by source-row order. The map is a readout of local reporting density, not animal abundance or biological rarity.',
      optionsMode: 'fresh',
      options: {category: 'Birds', view: 'score', neighborCount: 10, threshold: 1.5},
      controls: ['neighborCount'],
      readouts: ['sampleSize', 'neighborCount', 'medianScore', 'scoreDistribution'],
      stage: 'density',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1000}
    },
    {
      id: 'neighborhood-scale',
      title: 'Change the neighbourhood',
      headline: 'The score depends on the neighbourhood scale',
      textAlternative:
        'The same sampled map and histogram update after the number of nearest records changes.',
      body: 'Change **Nearest records (k)** and the graph rebuilds because k sets the k-nearest-neighbour table. Scores can shift as the comparison neighbourhood grows; compare runs at the same k. This is the local perspective alongside DBSCAN’s connected hot spots and k-means’ forced partition.',
      optionsMode: 'fresh',
      options: {category: 'Birds', view: 'score', neighborCount: 20, threshold: 1.5},
      controls: ['neighborCount'],
      readouts: ['sampleSize', 'neighborCount', 'medianScore', 'scoreDistribution'],
      stage: 'neighbors',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1000}
    },
    {
      id: 'no-universal-cut',
      title: 'There is no universal cut',
      headline: 'A chosen score is a reading aid, not a test',
      textAlternative:
        'The continuous score map remains unchanged while a threshold marker moves across the histogram and the flagged count updates.',
      body: 'Move **Flag above LOF**: **{{flagged}}** records clear the selected score. This changes a parameter buffer and recomputes the GPU count; the continuous score map itself stays the same. LOF has no null distribution, so a high score is not a p-value or a statistically significant outlier. In records shaped by observer effort, a sparse pocket can reflect reporting coverage, access or identification practices. [Compare connected hot spots](#/story/nature-clusters) to see the different question DBSCAN asks.',
      optionsMode: 'fresh',
      options: {category: 'Birds', view: 'score', neighborCount: 10, threshold: 3.5},
      controls: ['threshold'],
      readouts: ['sampleSize', 'neighborCount', 'flagged', 'scoreDistribution'],
      stage: 'neighbors',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1000}
    }
  ]
});
