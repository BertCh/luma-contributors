// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {GLOBAL_FURNITURE} from '../../cartography/projection-notes';
import {defineScene, type LegendSpec} from '../scene';
import type {EarthquakeQueryLabOptions} from './earthquake-query-lab.compute';

export default defineScene<EarthquakeQueryLabOptions>({
  id: 'earthquake-query-lab',
  title: 'Earthquake catalog: what survives the query?',
  chapter: 'dataframe',
  order: 8,
  summary:
    'A GPUDataFrame runs live magnitude and year filters, a magnitude histogram, global and per-year top-K rankings, and type-dimension lookup over 77,231 archived USGS ComCat M4+ events. The archive ends Dec. 30, 2024, contains non-earthquake event types, and is not a complete hazard catalog.',
  contributors: [
    'GPUDataFrame',
    'GPUDataFrameHistogramQuery',
    'GPUDataFrameAggregationQuery',
    'GPUDataFrameGlobalSortQuery',
    'GPUDataFrameSortQuery',
    'GPUDataFrameLookupQuery',
    'GPUDataFrameJoinQuery'
  ],
  datasets: [{id: 'poopdeck-earthquakes', role: 'USGS ComCat archive, M4+, 2020–2024'}],
  initialView: {longitude: 0, latitude: 15, zoom: 1.1},
  basemap: ground('abyss'),
  furniture: {
    ...GLOBAL_FURNITURE,
    title: {
      title: 'Earthquake catalog query lab',
      subtitle: '77,231 archived M4+ events, 2020–2024',
      sample: 'USGS ComCat archive via poopdeck.gl',
      chips: ['Catalog archive']
    },
    credit: 'US Geological Survey (public domain)',
    caveat:
      'M4+ archive only; the event window ends Dec. 30, 2024. This is not a completeness, risk, or prediction map.'
  },
  options: [
    {
      kind: 'slider',
      id: 'minimumMagnitude',
      label: 'Minimum magnitude',
      group: 'Query',
      apply: 'param',
      min: 4,
      max: 7,
      step: 0.1,
      default: 4,
      unit: 'M',
      help: 'The archive already contains only events at approximately M4.0 or above. Raise the threshold to narrow the live GPU query.'
    },
    {
      kind: 'slider',
      id: 'startYear',
      label: 'First year',
      group: 'Query',
      apply: 'param',
      min: 2020,
      max: 2024,
      step: 1,
      default: 2020,
      help: 'First UTC year included in the query.'
    },
    {
      kind: 'slider',
      id: 'endYear',
      label: 'Last year',
      group: 'Query',
      apply: 'param',
      min: 2020,
      max: 2024,
      step: 1,
      default: 2024,
      help: 'Last UTC year included in the query. The archive stops on Dec. 30, 2024.'
    },
    {
      kind: 'slider',
      id: 'minimumDepth',
      label: 'Minimum reported depth',
      group: 'Query',
      apply: 'param',
      min: -100,
      max: 100,
      step: 1,
      default: -100,
      unit: 'km',
      help: 'Lower depth bound. The negative default includes the catalog’s negative depth values.'
    },
    {
      kind: 'slider',
      id: 'maximumDepth',
      label: 'Maximum reported depth',
      group: 'Query',
      apply: 'param',
      min: 0,
      max: 700,
      step: 5,
      default: 700,
      unit: 'km',
      help: 'Upper depth bound for the same live dataframe filter.'
    },
    {
      kind: 'select',
      id: 'rankCount',
      label: 'Ranking size',
      group: 'Ranking',
      apply: 'param',
      default: '50',
      display: 'segmented',
      options: [
        {value: '50', label: 'Top 50'},
        {value: '250', label: 'Top 250'}
      ],
      help: 'Select a precompiled top-K result size for both global and year-partitioned ranking.'
    },
    {
      kind: 'select',
      id: 'rankDirection',
      label: 'Rank magnitude',
      group: 'Ranking',
      apply: 'param',
      default: 'descending',
      display: 'segmented',
      options: [
        {value: 'descending', label: 'Largest first'},
        {value: 'ascending', label: 'Smallest first'}
      ],
      help: 'Switch between the largest and smallest magnitudes in the current filter.'
    },
    {
      kind: 'toggle',
      id: 'includeOtherTypes',
      label: 'Include non-earthquake event types',
      group: 'Query',
      apply: 'param',
      default: false,
      help: 'USGS event type is not guaranteed to be earthquake. Enable this to include other archived types in the filtered outputs.'
    },
    {
      kind: 'toggle',
      id: 'showContext',
      label: 'Show all archived events',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'The map shows catalog coordinates colored by magnitude. It is global context; the chart and readouts show the current GPU query results.'
    }
  ],
  legends: (_state): LegendSpec[] => [
    {
      id: 'magnitude',
      title: 'Magnitude',
      kind: 'ramp',
      ramp: 'inferno',
      extent: [4, 8],
      unit: 'M',
      note: 'The dim context colors the archive. Values above 8 use the top color.'
    },
    {
      id: 'query-output',
      kind: 'categories',
      title: 'Live query output',
      entries: [
        {color: [255, 178, 78, 240], label: 'Rows satisfying the query'},
        {color: [255, 255, 255, 255], label: 'Stars: selected global magnitude ranking'}
      ]
    }
  ],
  readouts: [
    {id: 'catalogRows', label: 'Source rows', unit: 'events'},
    {id: 'catalogTypes', label: 'Type dictionary'},
    {id: 'filteredRows', label: 'Rows after query', unit: 'events', emphasis: 'tile'},
    {id: 'meanMagnitude', label: 'Mean magnitude', unit: 'M'},
    {id: 'maximumMagnitude', label: 'Maximum magnitude', unit: 'M'},
    {id: 'minimumDepth', label: 'Minimum reported depth', unit: 'km'},
    {id: 'topKRows', label: 'Global top-K'},
    {id: 'batchTopKRows', label: 'Per-year top-K'},
    {id: 'typeJoin', label: 'Type-dimension lookup'},
    {id: 'otherTypeRows', label: 'Non-earthquake types'},
    {id: 'joinCapacity', label: 'Join capacity'},
    {
      id: 'magnitudeHistogram',
      label: 'Magnitude histogram',
      kind: 'chart',
      placement: 'map'
    }
  ],
  story: [
    {
      id: 'catalog',
      title: 'Start with the archive',
      headline: 'The archive is a bounded slice',
      body: 'The map places each archived record at its catalog coordinate and colors it by magnitude. **Show all archived events** below toggles the geographic context. The table contains **{{catalogRows}}** from the USGS ComCat archive; the source is already filtered to roughly M4+.',
      textAlternative:
        'A global point map shows the 77,231 archived event locations, colored by magnitude.',
      evidence: '**{{catalogRows}}** are loaded from the bundled table.',
      caveat:
        'This archive stops on Dec. 30, 2024. Completeness varies by magnitude, place, and time; a blank area is not evidence of no earthquakes.',
      controls: ['showContext'],
      readouts: ['catalogRows', 'catalogTypes'],
      camera: {longitude: 0, latitude: 15, zoom: 1.1, transitionMs: 1400},
      options: {showContext: true}
    },
    {
      id: 'filter',
      title: 'Filter before interpreting',
      headline: 'Each filter changes the denominator',
      body: 'The compiled `GPUDataFrame` applies **Minimum magnitude**, **First year**, **Last year**, **Minimum reported depth**, and **Maximum reported depth** below. The GPU reduction reports rows that survive; the bright map overlay is the query’s GPU selection mask.',
      textAlternative:
        'A bright overlay marks locations whose magnitude, year, depth and event type pass the GPU filter.',
      evidence: '**{{filteredRows}}** rows currently satisfy the query.',
      caveat:
        'The starting catalog is M4+ already, so raising the slider cannot recover smaller events that were never archived here.',
      controls: ['minimumMagnitude', 'startYear', 'endYear', 'minimumDepth', 'maximumDepth'],
      readouts: ['filteredRows', 'meanMagnitude', 'maximumMagnitude'],
      camera: {longitude: 0, latitude: 15, zoom: 1.1, transitionMs: 1200}
    },
    {
      id: 'histogram',
      title: 'Inspect the magnitude distribution',
      headline: 'The fixed bins expose the filtered tail',
      body: 'The histogram bins magnitudes from 4 through 9 on the GPU. **Include non-earthquake event types** below changes the same filter feeding the chart and summaries.',
      textAlternative:
        'A histogram shows the magnitude counts for rows satisfying the current GPU query.',
      evidence: 'The chart is recomputed from the live filtered rows.',
      caveat:
        'The histogram domain is fixed at M4–9; values beyond that range are omitted from its bins.',
      controls: ['includeOtherTypes'],
      readouts: ['magnitudeHistogram', 'filteredRows'],
      options: {includeOtherTypes: false}
    },
    {
      id: 'ranking',
      title: 'Ask what “top” means',
      headline: 'Global and yearly rankings answer different questions',
      body: '`topKGlobal` selects across the whole archive. `topK` selects independently inside explicit calendar-year partitions, so its combined result can contain more rows. Choose **Ranking size** and **Rank magnitude** below to compare largest or smallest values.',
      textAlternative:
        'A bright query overlay marks matching rows, while white star markers show the currently selected largest or smallest magnitudes from a global GPU top-K result. Two live counts compare global ranking with five calendar-year partitions.',
      evidence: '**{{topKRows}}** globally; **{{batchTopKRows}}** across the year partitions.',
      caveat:
        'Magnitude ranks describe this archived slice and do not measure future likelihood or hazard.',
      controls: ['rankCount', 'rankDirection'],
      readouts: ['topKRows', 'batchTopKRows', 'maximumMagnitude']
    },
    {
      id: 'types',
      title: 'Keep event type semantics visible',
      headline: 'The type dictionary changes what survives',
      body: 'A unique-key type dimension drives archive-wide GPU lookup. The anti join isolates every archived row whose event type is not `earthquake`; those rows can be included in the active query with **Include non-earthquake event types** below.',
      textAlternative:
        'The lookup reports archive-wide matched and missing type keys with hash-probe counts; an anti join counts all archived records whose type is not earthquake.',
      evidence:
        '**{{otherTypeRows}}** non-earthquake rows are found by the archive-wide anti join; {{joinCapacity}}.',
      caveat:
        'Type labels classify catalog records; they do not certify that every record is a tectonic earthquake.',
      controls: ['includeOtherTypes'],
      readouts: ['typeJoin', 'otherTypeRows', 'joinCapacity'],
      options: {includeOtherTypes: false}
    },
    {
      id: 'explore',
      title: 'Explore the query boundary',
      headline: 'Change the query and read the new result',
      body: 'Change **Minimum magnitude**, **Minimum reported depth**, and **Maximum reported depth** below to update the filter, histogram and summary. The year and event-type controls remain available in the earlier steps; the map stays as whole-archive context.',
      textAlternative:
        'The map retains all archived points while the bright subset and summary outputs respond to query controls.',
      evidence: '**{{filteredRows}}** rows remain under the current settings.',
      caveat:
        'Negative depth values can occur in the source and are reported as recorded; they are not automatically invalid.',
      controls: ['minimumMagnitude', 'minimumDepth', 'maximumDepth'],
      readouts: ['filteredRows', 'minimumDepth', 'magnitudeHistogram']
    }
  ],
  snippet: state => `const query = frame
  .filter(column('magnitude').greaterThanOrEqual(parameter('minimumMagnitude', ${state.minimumMagnitude})))
  .filter(column('year').greaterThanOrEqual(parameter('startYear', ${state.startYear})))
  .filter(column('depth').greaterThanOrEqual(parameter('minimumDepth', ${state.minimumDepth})))
  .topKGlobal('magnitude', ${state.rankCount}, {direction: '${state.rankDirection}'});

const globalRanking = query.compile(globalRankingGraph);
const yearLocalRanking = filtered.topK('magnitude', ${state.rankCount}, {direction: '${state.rankDirection}'}).compile(yearLocalGraph);
const eventTypeLookup = filtered.lookup(typeDimension, {
  leftOn: 'eventType', rightOn: 'typeCode'
});`,
  about: {
    what: 'An interactive query notebook over a bundled USGS ComCat event archive.',
    why: 'Filtering, grouping, ranking, and categorical joins are everyday dataframe operations; their semantics affect counts and interpretation.',
    howToRead:
      'The global map is context. The chart and GPU readouts report query outputs. Yearly record-batch boundaries make per-batch top-K visibly different from one global top-K.'
  },
  create: async ctx =>
    (await import('./earthquake-query-lab.compute')).createEarthquakeQueryLab(ctx)
});
