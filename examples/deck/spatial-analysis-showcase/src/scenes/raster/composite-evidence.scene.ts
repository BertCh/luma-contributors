// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CompositeEvidenceOptions} from './composite-evidence.compute';
import {COMPOSITE_DATES, SOURCE_DATE_COLORS} from './composite-evidence-data';

export default defineScene<CompositeEvidenceOptions>({
  id: 'composite-evidence',
  title: 'What survives a cloud-screened composite?',
  chapter: 'raster',
  order: 4,
  summary:
    'Compare GPU mean, minimum, maximum and median NDVI composites from 16 irregular Sentinel-2 acquisitions across the Dixie Fire and its recovery. Support counts and source-date attribution show where a composite summarizes enough observations and which acquisition controls an extreme.',
  contributors: ['GPURasterTemporal', 'GPURasterConditional'],
  datasets: [{id: 'ndvi-timeseries', role: 'Sentinel-2 NDVI time stack (100 m)'}],
  initialView: {longitude: -121.0, latitude: 40.17, zoom: 10.6},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'Dixie Fire NDVI composites',
      subtitle: 'Observed Sentinel-2 acquisitions, 2017–2024'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data 2017–2024',
    caveat:
      'Sparse seasonal observations are descriptive; SCL screening retains some shadow and unclassified pixels.'
  },
  options: [
    {
      kind: 'select',
      id: 'operation',
      label: 'Composite statistic',
      group: 'GPURasterTemporal',
      apply: 'compile',
      default: 'median',
      help: 'Selects one of four precompiled mean, minimum, maximum or median graphs. Extreme composites retain the date of the chosen observation; an even-sample median and a mean have no single source date.',
      options: [
        {
          value: 'mean',
          label: 'Mean',
          help: 'Arithmetic mean of valid observations; no individual acquisition is the source.'
        },
        {
          value: 'minimum',
          label: 'Minimum',
          help: 'Lowest valid NDVI and its acquisition date; may retain residual cloud or shadow contamination.'
        },
        {
          value: 'maximum',
          label: 'Maximum',
          help: 'Highest valid NDVI and its acquisition date; can select pre-fire canopy and hide a disturbance.'
        },
        {
          value: 'median',
          label: 'Median',
          help: 'Middle valid NDVI; a single source date exists only when the valid sample count is odd.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Evidence layers',
      apply: 'param',
      default: 'composite',
      help: 'Switch among composite NDVI, valid-observation count and the source acquisition for extrema (or odd-count medians).',
      options: [
        {value: 'composite', label: 'NDVI composite'},
        {value: 'support', label: 'Valid observations'},
        {value: 'source-date', label: 'Source acquisition date'}
      ]
    },
    {
      kind: 'range',
      id: 'dateWindow',
      label: 'Acquisition window',
      group: 'Evidence window',
      apply: 'param',
      min: 0,
      max: 15,
      step: 1,
      default: [0, 15],
      format: value => COMPOSITE_DATES[Math.round(value)] ?? String(value),
      help: 'Only these actual acquisition indices contribute. Dates outside the selected range are missing; the temporal operator does not interpolate between acquisitions.'
    },
    {
      kind: 'slider',
      id: 'minimumCount',
      label: 'Minimum valid observations',
      group: 'Evidence window',
      apply: 'param',
      min: 1,
      max: 16,
      step: 1,
      default: 4,
      help: 'Cells with fewer valid acquisitions than this threshold are made transparent in the NDVI composite. The support map and click probe still report the actual observed count.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      help: 'Changes the continuous color ramp for NDVI or support counts.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'},
        {value: 'grayscale', label: 'Grayscale'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Reveals more hillshade and roads beneath the raster.'
    }
  ],
  readouts: [
    {
      id: 'screening',
      label: 'Acquisition contract',
      help: 'Dates, pixel size and sensor for the sampled stack.'
    },
    {id: 'window', label: 'Dates in this composite'},
    {
      id: 'support',
      label: 'Composite support',
      help: 'Valid observations per pixel after the date window, plus the mean and maximum support.'
    },
    {
      id: 'masked',
      label: 'Minimum-count mask',
      help: 'Cells below the selected support threshold are transparent only on the composite layer.'
    },
    {id: 'pixelChart', label: 'Clicked pixel NDVI by acquisition', kind: 'chart'},
    {
      id: 'pixel',
      label: 'Clicked composite summary',
      layout: 'block',
      help: 'Composite value, support, status and the source date where this operator selects one.'
    }
  ],
  legends: (state, data) =>
    state.layer === 'source-date'
      ? [
          {
            kind: 'categories',
            title: 'Source acquisition',
            entries: ((data.dates as string[] | undefined) ?? [...COMPOSITE_DATES]).map(
              (date, index) => ({color: SOURCE_DATE_COLORS[index], label: date})
            ),
            note:
              state.operation === 'mean'
                ? 'Mean has no unique source date; transparent cells are shown as no data.'
                : 'Source date is unavailable for an even-sample median.'
          }
        ]
      : [
          {
            kind: 'ramp',
            title: state.layer === 'support' ? 'Valid acquisitions' : 'Composite NDVI',
            ramp: state.ramp,
            extent:
              state.layer === 'support'
                ? [0, Math.max(1, Math.abs(state.dateWindow[1] - state.dateWindow[0]) + 1)]
                : [-0.1, 0.9],
            unit: state.layer === 'support' ? 'acquisitions' : 'NDVI'
          }
        ],
  story: [
    {
      id: 'observations',
      headline: 'A composite summarizes acquisitions that were actually observed',
      textAlternative: 'The map starts with a median NDVI across the 16 sampled Sentinel-2 dates.',
      title: 'Sixteen summer snapshots are not a continuous record',
      body: 'This stack contains **16 Sentinel-2 scenes** from **2017–2024**, sampled mostly in the warm season at irregular intervals. `GPURasterTemporal` reduces the valid values at each 100 m pixel; it does not fill dates between scenes. A missing pixel-date stays missing rather than becoming zero.\n\nBegin with **Composite statistic** below, then switch **Map shows** to support. The map colors are NDVI; the support map counts real valid observations in the chosen window.',
      evidence: '**{{screening}}** · **{{support}}**',
      caveat:
        'The source builder uses Sentinel-2 scene classification screening but retains some shadow and unclassified pixels, so “valid” does not mean perfectly cloud-free.',
      options: {operation: 'median', layer: 'composite', dateWindow: [0, 15], minimumCount: 4},
      controls: ['operation', 'layer'],
      readouts: ['screening', 'support']
    },
    {
      id: 'fire-shock',
      headline: 'A median depends on which observations survive',
      textAlternative:
        'A median from dates spanning the Dixie Fire summarizes the middle retained vegetation observation.',
      title: 'Place the date window around the disturbance',
      body: 'Use **Acquisition window** below to keep **2021-06-03 through 2022-08-07**. The median takes the middle valid observation at each pixel, reducing the influence of a single extreme value. Because this window mixes observations before and after the fire, its median cannot locate a change in time; the acquisition chart preserves that sequence.\n\nUse **Minimum valid observations** below to hide thinly supported composites. Then click a pixel: the line chart shows its actual NDVI at each acquisition, with gaps left missing.',
      evidence: '**{{support}}** · **{{masked}}**',
      caveat:
        'Six acquisitions fall in this window: four in 2021 and two in 2022. July 13, 2021 is the fire ignition date; September observations follow it. A composite is not a trajectory estimate.',
      options: {operation: 'median', layer: 'composite', dateWindow: [6, 11], minimumCount: 2},
      controls: ['dateWindow', 'minimumCount'],
      readouts: ['support', 'masked', 'pixelChart', 'pixel']
    },
    {
      id: 'maximum-hides-shock',
      headline: 'Maximum NDVI can select pre-fire canopy and hide the shock',
      textAlternative:
        'The maximum composite chooses the greenest valid acquisition, often before the fire.',
      title: 'Ask what the maximum remembers',
      body: 'Switch **Composite statistic** below to **Maximum**, then **Map shows** to **Source acquisition date**. A bright NDVI value can be real and still answer a different question: which sampled scene had the greenest pixel? In the burn scar, that may be a pre-fire observation.\n\nThe source-date map identifies the acquisition chosen for each extreme. Return **Map shows** to NDVI to compare the selected value with the median.',
      evidence:
        '**{{pixel}}** reports the chosen acquisition for the clicked pixel; mean and even-sample median have no unique source.',
      caveat:
        'Maximum composites are sensitive to residual contamination and date-specific phenology; they do not represent typical conditions.',
      options: {operation: 'maximum', layer: 'source-date', dateWindow: [0, 15]},
      controls: ['operation', 'layer'],
      readouts: ['pixelChart', 'pixel']
    },
    {
      id: 'robustness-check',
      headline: 'Change the window to see which conclusions depend on sampling',
      textAlternative:
        'The mean and minimum composites provide contrasting summaries over the same observations.',
      title: 'Try the mean, minimum and a later recovery window',
      body: 'Compare **Mean**, **Minimum** and **Median** below over **2021-09-21 to 2024-08-01**. Mean has no single source date; minimum highlights the lowest sampled value; median tracks the middle observation. The count layer reveals whether those summaries rest on two observations or many.\n\nTreat every result as a summary of sampled dates. `GPURasterTemporal` offers Savitzky–Golay for regularly spaced sequences, but these acquisitions have large and unequal time gaps, so smoothing would misstate their timing.',
      evidence:
        '**{{window}}** · **{{support}}**. Click a location and inspect its actual acquisition series before interpreting the raster.',
      caveat:
        'A seasonal or annual trend needs an explicit time model and comparable seasonal sampling; these composite operators do not estimate one.',
      options: {operation: 'mean', layer: 'composite', dateWindow: [9, 15], minimumCount: 2},
      controls: ['operation', 'dateWindow'],
      readouts: ['window', 'support', 'pixel']
    }
  ],
  about: {
    what: '`GPURasterTemporal` computes per-pixel mean, minimum, maximum or median across valid NDVI observations; it also writes support count, status and chosen source-slice index. `GPURasterConditional` applies the minimum-support display mask on the GPU.',
    why: 'Composite choice changes the question a raster answers. Maximum values can select pre-disturbance canopy, while a median summarizes the middle observed value; the support and source-date layers expose those choices.',
    howToRead:
      'NDVI uses a fixed −0.1 to 0.9 ramp. Support is the number of valid acquisitions inside the selected window. Source-date colors identify the chosen observation for minima and maxima, and for odd-count medians; a mean and even-count median have no unique source.'
  },
  create: async ctx => (await import('./composite-evidence.compute')).createCompositeEvidence(ctx)
});
