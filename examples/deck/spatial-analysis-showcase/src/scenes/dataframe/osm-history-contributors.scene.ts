// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {OsmContributorsOptions} from './osm-history-contributors.compute';

const FIRST_YEAR = 2007.5;
const LAST_YEAR = 2026.4;

export default defineScene<OsmContributorsOptions>({
  id: 'osm-history-contributors',
  title: 'New York OSM nodes by contributor concentration',
  chapter: 'dataframe',
  order: 3,
  summary:
    'GPU group statistics describe contributor concentration and sampled creation-day moments in New York OpenStreetMap node history. The outputs include top contributors, a Lorenz curve, a Gini coefficient, skewness, excess kurtosis and within-contributor z-scores; sampling thins occasional contributors.',
  contributors: ['GPUGroupStatistics'],
  datasets: [
    {
      id: 'poopdeck-osm-nyc',
      role: 'node creations 2007-2026 (400k sample), © OpenStreetMap contributors'
    }
  ],
  initialView: {longitude: -73.97, latitude: 40.71, zoom: 10.4},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'New York OSM contributor concentration',
      subtitle: 'Node shares, ranked contributors and Lorenz curve'
    },
    scaleBar: {units: 'metric'},
    credit: '© OpenStreetMap contributors (ODbL)',
    caveat: 'The displayed map is a uniform sample of node history.'
  },

  options: [
    {
      kind: 'slider',
      id: 'topN',
      label: 'Top N contributors',
      group: 'Highlight',
      apply: 'param',
      min: 1,
      max: 500,
      step: 1,
      default: 10,
      format: value => `top ${value}`,
      help: 'Contributors are ranked by nodes over the full history (0 is the most active, no names or ids are used). Nodes from the top N are drawn in orange, the rest grey. Only the threshold changes: it is a parameter write.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color nodes by',
      group: 'Highlight',
      apply: 'param',
      default: 'highlight',
      help: 'Show the top-N highlight, contributor rank, or sampled creation-day z-scores within each contributor group. Z-scores are shown only for groups with at least five sampled nodes.',
      options: [
        {value: 'highlight', label: 'Top N against the rest'},
        {value: 'rank', label: 'Contributor rank'},
        {value: 'within-group-z-score', label: 'Within-contributor z-score'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Rank ramp',
      group: 'Highlight',
      apply: 'param',
      default: 'magma',
      disabledWhen: state => state.colorBy !== 'rank',
      help: 'Ramp used for the contributor rank. Bright is a prolific contributor.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'range',
      id: 'years',
      label: 'Creation years',
      group: 'Window',
      apply: 'param',
      min: FIRST_YEAR,
      max: LAST_YEAR,
      step: 0.25,
      default: [FIRST_YEAR, LAST_YEAR],
      format: value => value.toFixed(2),
      help: 'Group only the nodes created in this interval. A mask kernel marks them; GPUGroupStatistics skips the rest. No recompile.'
    },
    {
      kind: 'select',
      id: 'kind',
      label: 'Kind of node',
      group: 'Window',
      apply: 'param',
      default: 'all',
      help: 'Group only one kind of node (poopdeck.gl grouping of tags).',
      options: [
        {value: 'all', label: 'All kinds'},
        {value: '0', label: 'Land and water'},
        {value: '1', label: 'Other tagged nodes'},
        {value: '2', label: 'Transport'},
        {value: '3', label: 'Infrastructure'},
        {value: '4', label: 'Places and amenities'},
        {value: '5', label: 'Buildings'}
      ]
    },
    {
      kind: 'slider',
      id: 'inspect',
      label: 'Inspect contributor',
      group: 'Statistics',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      format: value => `#${value}`,
      help: 'Rank of the contributor whose count and first, median and last creation month are read out. The statistics come from the GPU group table.'
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Node size',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.1,
      default: 1.2,
      unit: 'px',
      help: 'Disc radius in pixels.'
    },
    {
      kind: 'slider',
      id: 'contextOpacity',
      label: 'Opacity of the rest',
      group: 'Display',
      apply: 'param',
      min: 0.02,
      max: 0.8,
      step: 0.02,
      default: 0.2,
      disabledWhen: state => state.colorBy !== 'highlight',
      help: 'Opacity of the nodes outside the top N.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'Did a crowd draw New York, or a handful?',
      headline: 'Top contributors account for a large node share',
      textAlternative:
        'Orange points mark nodes created by the highest-ranked contributors across New York, while gray points show the remaining sampled history.',
      body: 'The orange nodes are everything created by the **top 10 contributors** out of 11,361 (anonymous ranks over the full history). The map is a **400,000-node sample** of all 901,827 nodes. Raise **Top N contributors** below and watch the orange spread.',
      camera: {
        longitude: -73.97,
        latitude: 40.71,
        zoom: 10.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1200
      },
      controls: ['topN'],
      readouts: ['window', 'topShare', 'fullShare']
    },
    {
      id: 'lorenz',
      title: 'Read the concentration curve',
      headline: 'The contributor distribution is strongly concentrated',
      textAlternative:
        'The Lorenz curve stays below the equality diagonal before rising sharply among the most active contributors; the Gini readout quantifies the gap.',
      body: '`GPUGroupStatistics` groups the nodes by contributor on the GPU, one dense row per contributor. The Lorenz curve sorts contributors from least to most active: if everyone made equal shares it would follow the dashed diagonal. It hugs the floor and shoots up at the right: the vertical rule marks where your **Top N contributors** start, and the Gini coefficient measures the bend (0 equal, 1 one person).',
      controls: ['topN'],
      readouts: ['lorenz', 'gini', 'active']
    },
    {
      id: 'leaders',
      title: 'Meet the leaders by rank',
      headline: 'Leading contributors differ in duration and volume',
      textAlternative:
        'A ranked bar chart and contributor readout compare node totals and the first, median and last creation dates for the selected contributor.',
      body: 'Use **Inspect contributor** to step down the ranking. Each readout comes from the same group table: how many nodes that contributor made, and the minimum, median and maximum creation day of those nodes. Some leaders worked for years, others in one short burst.',
      options: {inspect: 1},
      controls: ['inspect'],
      readouts: ['inspected', 'topBars']
    },
    {
      id: 'surge',
      title: 'Zoom into the 2013 to 2014 surge',
      headline: 'Five accounts dominate the selected creation window',
      textAlternative:
        'Orange nodes from the five leading contributors dominate the 2013 to 2014 selection, accompanied by concentration and Lorenz readouts.',
      body: 'Set **Creation years** to 2013.5 to 2014.5 and **Kind of node** to Other tagged nodes. In this window five accounts make over half of the sampled nodes (about 54%, against 27% for the whole history): the top share jumps. The next scene examines these concentrated creation windows; anonymous account ranks cannot establish import provenance or the number of people involved.',
      options: {years: [2013.5, 2014.5], kind: '1', topN: 5},
      controls: ['years', 'kind'],
      readouts: ['topShare', 'gini', 'lorenz', 'inspected']
    },
    {
      id: 'group-moments',
      title: 'Compare creation-time profiles within groups',
      headline: 'Per-contributor moments describe sampled creation dates',
      textAlternative:
        'The inspected contributor’s sample count, mean creation month, sample standard deviation in days, Fisher-Pearson skewness and excess kurtosis are read from GPU group statistics. A diverging map colors each included node by its within-contributor creation-day z-score; a dark marker and bounded GPU probe identify one source row. Groups below five sampled nodes are transparent.',
      body: '`GPUGroupStatistics` computes each contributor’s sample mean and sample standard deviation, Fisher-Pearson **g₁ skewness**, **g₂ excess kurtosis**, and a row-aligned z-score for every sampled node. The g₁ and excess g₂ formulas use population central moments without small-sample bias correction; the selected sample variance convention applies to standard deviation and z-scores. Use **Inspect contributor** to examine a group; set **Color nodes by** to *Within-contributor z-score* to map how early or late each selected node is relative to that contributor’s sampled creation-day mean. The bounded GPU probe reports one included source row’s actual row-aligned z-score. A small-group gate hides z-scores and moments for groups with fewer than five sampled nodes. That cutoff is an analyst display rule, not a statistical theorem. Constant-date groups have undefined skewness and excess kurtosis; those values display as unavailable.',
      caveat:
        'The 400,000-node uniform sample represents 901,827 source nodes imperfectly and thins the contributor long tail. Creation-day moments describe sampled node histories only; they do not establish contributor intent, quality, identity, bot status or import provenance.',
      controls: ['inspect', 'colorBy', 'years', 'kind'],
      readouts: ['groupTable', 'moments', 'zScoreProbe', 'inspected'],
      options: {colorBy: 'within-group-z-score'}
    },
    {
      id: 'honest',
      title: 'How far to trust a sample',
      headline: 'Sampling estimates shares but thins occasional contributors',
      textAlternative:
        'A rank-colored node map is paired with sampled and exact top-share readouts, exposing agreement for prolific contributors and missing low-volume accounts.',
      body: 'A uniform sample estimates contributor shares with sampling variation, and the **long tail of occasional contributors is thinned**: with 44% of the nodes, an account represented by 3 nodes is missing about one time in six. Set **Creation years** back to the whole range and compare the sampled top share with the exact full-history share in the readouts. Try **Rank ramp**: color by rank.',
      options: {topN: 100, colorBy: 'rank'},
      controls: ['colorBy', 'topN'],
      readouts: ['topShare', 'fullShare', 'gini']
    }
  ],

  legends: state =>
    (
      [
        state.colorBy === 'highlight'
          ? {
              kind: 'categories',
              title: 'Nodes in the window',
              entries: [
                {color: [255, 70, 130, 255], label: 'Most active contributor (#1)'},
                {color: [255, 170, 50, 255], label: `Top ${state.topN} contributors`},
                {color: [150, 155, 170, 160], label: 'Everyone else'}
              ],
              note: 'Rank is by nodes over the full history; anonymous.'
            }
          : state.colorBy === 'rank'
            ? {
                kind: 'ramp',
                title: 'Contributor rank (0 = most active)',
                ramp: state.ramp,
                extent: [0, 1000],
                sqrtScale: true,
                unit: 'rank'
              }
            : {
                kind: 'ramp',
                title: 'Creation day within contributor',
                ramp: 'diverging',
                extent: [-3, 3],
                midpoint: 0,
                midpointLabel: 'group mean',
                unit: 'sample SD',
                note: 'Colors saturate beyond ±3 sample SD. Groups with fewer than five sampled records are hidden.'
              }
      ] as LegendSpec[]
    ).filter(Boolean),

  readouts: [
    {
      id: 'rows',
      label: 'Nodes in the sample',
      help: 'A seeded uniform sample of the full history.'
    },
    {
      id: 'window',
      label: 'Nodes in the window (estimate)',
      help: 'Grouped rows divided by the sample fraction (0.44).'
    },
    {
      id: 'active',
      label: 'Contributors in the window',
      help: 'Contributors with at least one sampled node in the window. The sample misses some occasional contributors, so this is a lower bound.'
    },
    {
      id: 'topShare',
      label: 'Share of the top N (sample)',
      help: 'Nodes made by the N top-ranked contributors, as a share of the window.'
    },
    {
      id: 'fullShare',
      label: 'Share of the top N (exact)',
      help: 'Computed over all 901,827 nodes at build time; only defined for the whole history.'
    },
    {
      id: 'gini',
      label: 'Gini coefficient',
      help: 'Inequality of nodes per contributor: 0 is equal, 1 is one contributor.'
    },
    {
      id: 'inspected',
      label: 'Inspected contributor',
      layout: 'block',
      help: 'Count, first, median and last creation month from GPUGroupStatistics.'
    },
    {
      id: 'groupTable',
      label: 'GPU group table status',
      help: 'Dense table capacity, returned row count and overflow status from GPUGroupStatistics.'
    },
    {
      id: 'moments',
      label: 'Inspected group moments',
      layout: 'block',
      help: 'For groups with at least five finite sampled creation days: mean creation month, sample standard deviation in days, Fisher-Pearson g1 and excess g2 from population central moments. Constant groups have undefined g1 and g2.'
    },
    {
      id: 'zScoreProbe',
      label: 'GPU z-score probe',
      help: 'One sample row from the inspected contributor is selected from source columns; a bounded GPU pass reads its row-aligned z-score.'
    },
    {id: 'lorenz', label: 'Lorenz curve', kind: 'chart'},
    {id: 'topBars', label: 'Ten most active contributors', kind: 'chart'}
  ],

  snippet: () => `graph.add(
  new GPUGroupStatistics({
    keys: contributorRank,            // uint32 per node, dense in [0, 11361)
    mask: windowMask,                 // years and kind, written by a kernel
    keyCount: 11361,                  // dense: row k is contributor k
    variance: 'sample',               // n - 1 SD for group summaries and row z-scores
    columns: [{
      values: days,
      statistics: ['minimum', 'maximum', 'mean', 'median', 'standardDeviation', 'skewness', 'kurtosis', 'zScore'],
      output: {minimums, maximums, means, medians, standardDeviations, skewness, kurtosis, zScores}
    }],
    output: {keys, counts, count, overflow}
  })
);
// after readback: sort counts ascending on the CPU -> Lorenz curve, Gini`,

  about: {
    what: 'A GPU group-by of 400,000 sampled OpenStreetMap node creations in New York City by anonymous contributor rank. GPUGroupStatistics returns counts, creation-day moments and row-aligned within-group z-scores; the CPU only sorts 11,361 counts for the Lorenz curve.',
    why: 'A map built by many people and a map built by a few accounts need different care: who corrects errors, whose conventions win, and what disappears if one contributor leaves. Concentration is the first number to know.',
    howToRead:
      'Orange nodes belong to the top N contributors (by full-history rank), grey nodes to everyone else. In within-contributor z-score mode, the diverging ramp compares a node’s sampled creation day with that contributor’s mean; groups below five sampled nodes are hidden. The Lorenz curve plots the cumulative share of nodes against contributors sorted from least to most active; a curve far below the diagonal means concentration. Shares and Gini from the sample are close to, but not equal to, the exact full-history values shown beside them, because the sample thins the long tail. Moments describe sampled creation times, not contributor intent, quality, identity, bot status or imports. No user names or ids are shipped or shown; ranks are anonymous. Data: © OpenStreetMap contributors, ODbL.'
  },

  create: async ctx =>
    (await import('./osm-history-contributors.compute')).createOsmContributors(ctx)
});
