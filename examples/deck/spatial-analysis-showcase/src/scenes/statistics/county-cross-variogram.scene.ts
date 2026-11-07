// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CountyCrossVariogramOptions} from './county-cross-variogram.compute';

export default defineScene<CountyCrossVariogramOptions>({
  id: 'county-cross-variogram',
  title: 'How do unemployment and poverty vary across New Jersey counties?',
  chapter: 'statistics',
  order: 92,
  summary:
    'GPU pair-bin semivariograms summarize observed unemployment and family-poverty differences across New Jersey’s historical common-support county units and decennial observations.',
  contributors: ['GPUCrossVariogram', 'GPUSpaceTimeVariogram'],
  datasets: [
    {
      id: 'ncovr-new-jersey',
      role: 'historical county polygons, source FIPS and observed decennial unemployment and family-poverty rates'
    }
  ],
  initialView: {longitude: -74.55, latitude: 40.05, zoom: 7.1},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'New Jersey county pair summaries',
      subtitle: 'Observed rates · descriptive lag bins',
      sample: 'Historical common-support units · 1970 / 1980 / 1990',
      chips: ['Pair-bin means', 'Not a fitted model']
    },
    scaleBar: {units: 'metric', ticks: [50_000]},
    northArrow: 'always',
    credit: 'NCOVR; Messner et al. (2000); GeoDa archive',
    caveat:
      'Polygon-vertex-mean anchors are distance proxies for areal units, not stations or true centroids. Pair summaries are descriptive; three decennial observations do not identify a dynamic process.'
  },
  options: [
    {
      id: 'analysis',
      label: 'Pair summary',
      kind: 'select',
      display: 'segmented',
      default: 'single-period',
      apply: 'compile',
      options: [
        {value: 'single-period', label: 'One period'},
        {value: 'space-time', label: 'Space + time'}
      ],
      help: 'One-period mode uses 21 units from the selected census observation. Space + time uses all 63 observed unit-period records, with time values 0, 10 and 20 years.'
    },
    {
      id: 'mode',
      label: 'Rate comparison',
      kind: 'select',
      display: 'segmented',
      default: 'unemployment',
      apply: 'compile',
      options: [
        {value: 'unemployment', label: 'Unemployment'},
        {value: 'unemployment-poverty', label: 'Unemployment × poverty'}
      ],
      help: 'Unemployment computes an ordinary empirical semivariogram. The second choice computes signed cross-semivariance from paired unemployment and family-poverty rates; it is not a correlation.'
    },
    {
      id: 'period',
      label: 'Census observation',
      kind: 'select',
      default: '1970',
      apply: 'compile',
      disabledWhen: state => state.analysis === 'space-time',
      options: [
        {value: '1970', label: '1970'},
        {value: '1980', label: '1980'},
        {value: '1990', label: '1990'}
      ],
      help: 'Select an observed decennial slice for the one-period summary and source map. Space-time mode keeps the most recently selected map period.'
    },
    {
      id: 'mapField',
      label: 'Mapped source rate',
      kind: 'select',
      default: 'unemployment',
      apply: 'param',
      options: [
        {value: 'unemployment', label: 'Unemployment'},
        {value: 'poverty', label: 'Family poverty'}
      ],
      help: 'Color the source county polygons by the observed unemployment rate or family-poverty rate for the selected census observation. Poverty fields are measured in the preceding year.'
    },
    {
      id: 'cutoff',
      label: 'Maximum distance',
      kind: 'slider',
      default: 200000,
      apply: 'param',
      min: 100000,
      max: 300000,
      step: 100000,
      format: value => `${value / 1000} km`,
      help: 'Include pairs up to this distance and divide it into six equal-width left-closed, right-open bins; the final endpoint is included.'
    },
    {
      id: 'spatialLag',
      label: 'Linked distance band',
      kind: 'slider',
      default: 1,
      apply: 'param',
      min: 0,
      max: 5,
      step: 1,
      format: value => `Band ${value + 1}`,
      help: 'Show directionless links between the authored county display anchors that fall into this spatial distance band.'
    },
    {
      id: 'temporalLag',
      label: 'Linked time band',
      kind: 'slider',
      default: 1,
      apply: 'param',
      disabledWhen: state => state.analysis !== 'space-time',
      min: 0,
      max: 2,
      step: 1,
      format: value => ['0–6.667 years', '6.667–13.333 years', '13.333–20 years'][value]!,
      help: 'Select the temporal interval for the map’s anchor-link evidence. The available actual time lags are zero, ten and twenty years.'
    }
  ],
  story: [
    {
      id: 'observed-county-rates',
      title: 'Start with the observed county rates',
      headline: 'The map shows archived values on historical polygons',
      body: 'Use **Mapped source rate** to view source unemployment or family-poverty percentages for one observed census period. Poverty columns are FP69, FP79 and FP89: each measures the preceding year. The records contain decennial snapshots only; no annual values are interpolated.',
      textAlternative:
        'Historical New Jersey county polygons are filled by the selected observed unemployment or preceding-year family-poverty percentage, with a fixed sequential scale and a scale bar.',
      evidence:
        'The map uses the selected source column; the pair chart remains a GPU summary of the chosen unemployment mode.',
      caveat:
        'These are NCOVR historical common-support units. Their vertex-mean anchors are explicit distance proxies for areal units, not observation stations or true centroids.',
      controls: ['period', 'mapField'],
      readouts: ['sourceObservation', 'pairSummaryStatus']
    },
    {
      id: 'single-period-pairs',
      title: 'Compare pairs in one observed period',
      headline: 'Distance bins summarize pairwise rate differences',
      body: '**One period** sends the selected 21 unemployment observations to **GPUCrossVariogram**. With **Unemployment × poverty**, it also supplies that period’s family-poverty observations and computes signed cross-semivariance. Each unordered pair contributes once; the chart and readout show GPU bin means, pair counts and status.',
      textAlternative:
        'A six-bin chart displays GPU means for a selected one-period unemployment semivariogram or signed unemployment–poverty cross-semivariogram; empty bins are gaps, with per-bin counts and statuses listed.',
      evidence:
        'Map links connect the selected distance-band anchors. Chart values and counts are read from GPU output buffers.',
      caveat:
        'Cross-semivariance may be negative and is not correlation. The signed chart uses a symmetric zero-centered axis; units are percentage-points squared.',
      options: {analysis: 'single-period', mode: 'unemployment-poverty'},
      controls: ['period', 'mode', 'cutoff', 'spatialLag'],
      readouts: ['pairSummaryStatus', 'variogram', 'pairSupport', 'pairBinDetails']
    },
    {
      id: 'three-observation-space-time',
      title: 'Add the observed decennial time lags',
      headline: 'Three snapshots produce only three actual time lags',
      body: '**Space + time** sends all observed unemployment records for the three decennial periods to **GPUSpaceTimeVariogram**. Time values are 0, 10 and 20 years, grouped into the configured intervals shown in the chart. The same-county pairs at positive time lag have zero spatial separation and remain valid pair support.',
      textAlternative:
        'Three chart series show GPU semivariance means over six spatial distance bands for each configured temporal interval; the readout exposes all 18 means, counts and statuses.',
      evidence:
        'The space-time chart displays all spatial-by-temporal bins, including empty-bin gaps. Select a temporal band to see its anchor links on the polygons.',
      caveat:
        'Three decennial observations cannot identify a continuous or dynamic process. Interval labels are bin boundaries; the actual observed time lags are zero, ten and twenty years.',
      options: {analysis: 'space-time', mode: 'unemployment'},
      controls: ['analysis', 'mode', 'cutoff', 'spatialLag', 'temporalLag'],
      readouts: ['pairSummaryStatus', 'variogram', 'pairSupport', 'pairBinDetails']
    },
    {
      id: 'cutoff-and-sparse-bins',
      title: 'Change the cutoff and inspect sparse support',
      headline: 'Equal-width bins change when the distance cutoff changes',
      body: 'The **Maximum distance** control sets the spatial cutoff and therefore the width of all six distance intervals. Hover a chart point for its GPU mean; use **Show values** or the per-bin readout for the corresponding count and status. Empty bins remain gaps rather than zero estimates.',
      textAlternative:
        'The selected distance cutoff changes six equal-width spatial intervals; chart gaps, pair counts and per-bin status make sparse support visible.',
      evidence:
        'The linked map band follows the same cutoff and interval assignment as the selected pair support; GPU counts are retained even for zero-length same-county temporal links.',
      caveat:
        'These are descriptive pair-bin summaries. They do not fit a range, sill, covariance model or causal relationship.',
      controls: ['cutoff', 'spatialLag', 'temporalLag'],
      readouts: ['pairSummaryStatus', 'variogram', 'pairSupport', 'pairBinDetails']
    },
    {
      id: 'explore-pair-summaries',
      title: 'Explore the observed pair summaries',
      headline: 'Keep the data, support and intervals in view',
      body: 'Switch between a one-period summary and all observed space-time records, choose unemployment or its signed cross-semivariance with poverty, and inspect the GPU means beside their pair counts. The map shows the source rates; selected links show geometric pair support between authored county anchors.',
      textAlternative:
        'The final interactive view pairs a fixed-scale source-rate choropleth and selected anchor links with the currently selected GPU semivariogram chart and detailed bin readout.',
      evidence:
        'The snippet records exact source columns, bin counts, cutoff and observed times for the selected mode.',
      caveat:
        'Pair locations stand in for areal county support, and the small decennial panel describes these archived units only.',
      controls: ['analysis', 'mode', 'period', 'mapField', 'cutoff'],
      readouts: ['sourceObservation', 'pairSummaryStatus', 'variogram', 'pairBinDetails']
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      title:
        state.mapField === 'unemployment'
          ? 'Observed unemployment rate'
          : 'Observed family-poverty rate',
      ramp: 'cividis',
      extent: state.mapField === 'unemployment' ? [0, 20] : [0, 50],
      unit: 'percent',
      note:
        state.mapField === 'poverty'
          ? 'Source FP column for the year preceding the selected census period'
          : `Source UE${state.period.slice(-2)} census-year rate`
    }
  ],
  readouts: [
    {id: 'sourceObservation', label: 'Mapped source column', format: 'text'},
    {id: 'pairSummaryStatus', label: 'GPU bins and unordered-pair total', format: 'text'},
    {id: 'variogram', label: 'Distance and time pair summary', kind: 'chart'},
    {
      id: 'pairBinDetails',
      label: 'Bin means, pair counts and statuses',
      format: 'text',
      layout: 'block'
    },
    {id: 'pairSupport', label: 'Selected map-link support', format: 'text'}
  ],
  snippet: state => {
    const contributor =
      state.analysis === 'space-time' ? 'GPUSpaceTimeVariogram' : 'GPUCrossVariogram';
    const sourceYear = state.period;
    const povertyYear = String(Number(sourceYear) - 1);
    const unemploymentColumns =
      state.analysis === 'space-time'
        ? '[unemployment1970, unemployment1980, unemployment1990]'
        : `unemployment${sourceYear}`;
    const povertyColumns =
      state.analysis === 'space-time'
        ? '[familyPoverty1969, familyPoverty1979, familyPoverty1989]'
        : `familyPoverty${povertyYear}`;
    return `// NCOVR observed columns; no annual interpolation.
// UE${sourceYear.slice(-2)} = unemployment; FP${povertyYear.slice(-2)} = preceding-year family poverty.
new ${contributor}({
  positions: authoredCountyVertexMeanAnchorsProjectedAEQD,
  values: ${unemploymentColumns},
  ${state.mode === 'unemployment-poverty' ? `secondValues: ${povertyColumns},` : '// omit secondValues for the univariate semivariogram'}
  ${state.analysis === 'space-time' ? 'times: [0, 10, 20] years, temporalLagCount: 3,' : ''}
  parameters: [${state.cutoff}, ${state.analysis === 'space-time' ? '20' : 'unused'}], lagCount: 6
});`;
  },
  about: {
    what: 'A descriptive GPU study of pairwise unemployment semivariance and signed unemployment–poverty cross-semivariance across the NCOVR New Jersey historical county units.',
    why: 'Distance and time intervals make the observed pair support visible alongside a county map, while GPU means, counts and status distinguish supported bins from empty ones.',
    howToRead:
      'The one-period mode uses the actual unemployment1970/1980/1990 fields and, for cross-semivariance, familyPoverty1969/1979/1989. Space-time uses all 63 source rows at times 0, 10 and 20 years; its configured intervals contain actual lags 0, 10 and 20. Each unordered i<j pair contributes once. County locations are authored polygon-vertex-mean anchors projected into AEQD [-74.55, 40.05], a reproducible areal-support proxy rather than a station or true centroid. Empty bins remain gaps. No variogram model is fitted, and the summaries do not imply prediction, causation or a continuous dynamic process. Source: [GeoDa NCOVR archive](https://geodacenter.github.io/data-and-lab/ncovr/); interval and pair-bin conventions follow the primary [gstat variogram](https://r-spatial.github.io/gstat/reference/variogram.html) and [space-time variogram](https://r-spatial.github.io/gstat/reference/variogramST.html) documentation.'
  },
  create: async ctx =>
    (await import('./county-cross-variogram.compute')).createCountyCrossVariogram(ctx)
});
