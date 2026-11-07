// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {HarborSimilarityOptions} from './harbor-track-similarity.compute';
import {HARBOR_AUDITED_SOURCE_PATHS} from './harbor-audited-window';

export default defineScene<HarborSimilarityOptions>({
  id: 'harbor-track-similarity',
  title: 'Which harbor tracks resemble each other?',
  chapter: 'movement',
  order: 8,
  summary:
    'Compare three GPU trajectory similarity measures on eight real, support-audited passenger-vessel paths in a shared UTC hour.',
  contributors: ['GPUTrackSimilaritySearch'],
  datasets: [{id: 'ais-vessels', role: 'NY/NJ Harbor AIS tracks, 12 June 2024'}],
  initialView: {longitude: -74.015, latitude: 40.695, zoom: 11.2},
  basemap: ground('night', {labels: 'above'}),
  furniture: {
    title: {
      subtitle: 'Eight observed paths · 14:00–15:00 UTC',
      chips: ['Exact bounded GPU comparisons']
    },
    scaleBar: {units: 'metric', latitude: 40.69},
    credit: 'NOAA Office for Coastal Management / U.S. Coast Guard Navigation Center · CC0 1.0',
    caveat:
      'Similarity describes these sampled paths under a planar metre projection. It does not establish shared intent, identity, or behavior between fixes.'
  },
  options: [
    {
      kind: 'select',
      id: 'queryTrack',
      label: 'Query path',
      group: 'Search',
      apply: 'param',
      default: '1',
      options: HARBOR_AUDITED_SOURCE_PATHS.map((path, index) => ({
        value: String(index),
        label: `Path ${path}`
      })),
      help: 'Compare this actual AIS path with each of the other seven paths in the same hour.'
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Similarity measure',
      group: 'Search',
      apply: 'param',
      default: 'frechet',
      options: [
        {value: 'frechet', label: 'Discrete Fréchet'},
        {value: 'dynamic-time-warping', label: 'DTW accumulated distance'},
        {value: 'longest-common-subsequence', label: 'LCSS similarity'}
      ],
      help: 'Fréchet measures the maximum leash distance, DTW sums the minimum alignment distances, and LCSS normalizes ordered matches within 150 m.'
    },
    {
      kind: 'slider',
      id: 'frechetThresholdMeters',
      label: 'Fréchet limit',
      group: 'Threshold',
      apply: 'param',
      min: 1000,
      max: 40000,
      step: 1000,
      default: 10000,
      unit: 'm',
      help: 'Maximum discrete Fréchet leash distance in projected metres.'
    },
    {
      kind: 'slider',
      id: 'dtwAccumulatedThresholdMeters',
      label: 'DTW accumulated limit',
      group: 'Threshold',
      apply: 'param',
      min: 50000,
      max: 1000000,
      step: 50000,
      default: 300000,
      unit: 'm',
      help: 'Maximum sum of local alignment distances in projected metres. This is an accumulated path cost, not average separation.'
    },
    {
      kind: 'slider',
      id: 'lcssSimilarityThreshold',
      label: 'LCSS similarity minimum',
      group: 'Threshold',
      apply: 'param',
      min: 0,
      max: 0.2,
      step: 0.01,
      default: 0.05,
      help: 'Normalized ordered-match fraction for LCSS, using a fixed 150 m positional tolerance.'
    }
  ],
  story: [
    {
      id: 'bounded-sample',
      headline: 'Observed time support limits which path comparisons are defensible.',
      title: 'Compare paths with audited support',
      body: 'The search uses source path rows **221, 370, 509, 562, 595, 712, 726 and 735**, clipped to their observed fixes from 14:00 to 15:00 UTC. It tests all directed query–candidate pairs and never extends a track past its recorded support.',
      evidence: '**{{windowAudit}}** · **{{samplingAudit}}**',
      caveat:
        'The selected paths are a sample. AIS gaps remain unobserved time, even when two tracks receive a similar score.',
      textAlternative:
        'Eight real passenger-vessel paths are drawn over the New York Harbor basemap.',
      controls: [],
      readouts: ['windowAudit', 'samplingAudit']
    },
    {
      id: 'measurements-disagree',
      headline: 'Fréchet, warping and ordered matches can choose different nearest paths.',
      title: 'Similarity depends on the question',
      body: 'Discrete Fréchet emphasizes the worst separation along an ordered path alignment. Dynamic time warping accumulates local distances, while LCSS counts ordered matches within a 150 m tolerance and normalizes by the shorter track. For query path 370, the bundled hour ranks path 595 closest by Fréchet distance, while DTW and LCSS rank path 735 first. Change the measure and its separate threshold to compare those answers.',
      evidence: '**{{similarityMetric}}** · **{{searchStatus}}**',
      caveat:
        'DTW values are accumulated metres and therefore grow with path length; they are not an average separation. LCSS is a normalized similarity where larger scores indicate more matches. The chart lists the best threshold-matched candidates, up to three.',
      textAlternative:
        'A bar chart ranks threshold-matched candidates for the selected query under the active measure.',
      controls: [
        'metric',
        'frechetThresholdMeters',
        'dtwAccumulatedThresholdMeters',
        'lcssSimilarityThreshold'
      ],
      readouts: ['similarityMetric', 'searchStatus', 'candidateStatus', 'similarityTopK']
    },
    {
      id: 'threshold-pruning',
      headline: 'A strict threshold can legitimately return no matching candidate.',
      title: 'Distinguish no match from zero distance',
      body: 'Set a 1,000 m Fréchet threshold for path 370. None of the seven other observed paths meets that bound in this hour. The contributor prunes those candidates and reports no qualifying match; it does not replace a pruned score with zero or imply identical routes.',
      evidence: '**{{bestMatches}}** · **{{candidateStatus}}**',
      caveat:
        'A threshold defines this search result. It is not an independently estimated boundary between similar and dissimilar vessel behavior.',
      textAlternative:
        'The query remains visible, while the candidate ranking is empty when no path meets the chosen Fréchet bound.',
      options: {queryTrack: '1', metric: 'frechet', frechetThresholdMeters: 1000},
      controls: ['frechetThresholdMeters'],
      readouts: ['bestMatches', 'candidateStatus', 'searchStatus']
    },
    {
      id: 'inspect-matches',
      headline: 'The nearest shape match does not imply that two vessels travelled together.',
      title: 'Inspect the nearest candidate',
      body: 'The gold overlay shows the selected query; cyan shows its top ranked candidate. Choose another query path; hover a track to inspect its computed score, source MMSI and observed support interval.',
      evidence: '**{{queryPath}}** · **{{candidateCount}}**',
      caveat:
        'These shape comparisons do not account for vessel type, speed, navigational constraints, or behavior between fixes.',
      textAlternative: 'The selected query is gold and its best-ranked observed path is cyan.',
      options: {frechetThresholdMeters: 10000},
      controls: ['queryTrack'],
      readouts: ['queryPath', 'candidateCount', 'bestMatches', 'candidateStatus']
    }
  ],
  legends: () => [
    {
      kind: 'line',
      title: 'Observed paths and active match',
      entries: [
        {color: [114, 131, 144, 70], widthPixels: 2, label: 'Other audited paths'},
        {color: [250, 194, 71, 245], widthPixels: 4, label: 'Query path'},
        {color: [41, 218, 204, 245], widthPixels: 4, label: 'Best threshold match'}
      ],
      note: 'Match overlay is hidden until top-K search status is complete.'
    }
  ],
  readouts: [
    {id: 'windowAudit', label: 'Audited sample'},
    {id: 'samplingAudit', label: 'Observed interval range'},
    {id: 'queryPath', label: 'Query path'},
    {id: 'candidateCount', label: 'Comparison set'},
    {id: 'similarityMetric', label: 'Active score'},
    {id: 'searchStatus', label: 'Search capacity'},
    {id: 'candidateStatus', label: 'Candidate status'},
    {id: 'bestMatches', label: 'Ranked matches'},
    {id: 'similarityTopK', label: 'Closest candidates'}
  ],
  snippet: state =>
    `new GPUTrackSimilaritySearch({metric: '${state.metric}', topK: 3, survivorCapacity: 56});`,
  about: {
    what: 'Bounded GPU similarity search across eight real passenger-vessel AIS paths.',
    why: 'Different trajectory measures rank paths according to different geometric questions, so a nearest match depends on the chosen measure and threshold.',
    howToRead:
      'Choose a query and metric, inspect the top three chart entries, and compare the gold route pair. Every result uses recorded fixes in the selected hour only.'
  },
  create: async ctx =>
    (await import('./harbor-track-similarity.compute')).createHarborTrackSimilarity(ctx)
});
