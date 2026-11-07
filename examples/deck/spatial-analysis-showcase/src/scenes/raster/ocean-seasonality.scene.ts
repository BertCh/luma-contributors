// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {OceanSeasonalityOptions} from './ocean-seasonality.compute';
import {OCEAN_MODEL_STATUS_COLORS} from './ocean-seasonality-data';

const DATES = Array.from({length: 72}, (_, index) => {
  const year = 2012 + Math.floor(index / 12);
  const month = String((index % 12) + 1).padStart(2, '0');
  return `${year}-${month}`;
});

export default defineScene<OceanSeasonalityOptions>({
  id: 'ocean-seasonality',
  title: 'When the Pacific stays warm',
  chapter: 'raster',
  order: 5,
  summary:
    'Inspect NOAA ERSSTv5 monthly reconstructed sea-surface temperature and anomalies over the northeast Pacific. GPU seasonal smoothing and an offline single-break model describe this coarse 2012–2017 record; they do not classify marine heatwaves or attribute causes.',
  contributors: ['GPURasterTemporal', 'GPURasterSeasonalBreak'],
  datasets: [
    {id: 'northeast-pacific-sst', role: 'NOAA ERSSTv5 monthly SST and supplied anomaly'},
    {id: 'natural-earth', role: 'Natural Earth land, coastline and geographic context'}
  ],
  initialView: {longitude: -135, latitude: 40, zoom: 2.55},
  basemap: ground('abyss'),
  annotations: [
    {kind: 'water', coordinate: [-145, 56], text: 'Gulf of Alaska', size: 'small', tone: 'water'},
    {kind: 'area', coordinate: [-150, 61], text: 'Alaska', size: 'small'},
    {kind: 'area', coordinate: [-126, 51], text: 'British Columbia', size: 'small'},
    {kind: 'area', coordinate: [-123, 40], text: 'California', size: 'small'}
  ],
  furniture: {
    title: {title: 'When the Pacific stays warm', subtitle: 'NOAA ERSSTv5 · monthly 2012–2017'},
    scaleBar: {units: 'metric'},
    credit: 'NOAA National Centers for Environmental Information; NOAA CoastWatch ERDDAP',
    caveat:
      'Two-degree monthly reconstructions cannot resolve daily marine heatwaves or establish causes.'
  },
  options: [
    {
      kind: 'select',
      id: 'series',
      label: 'Series',
      group: 'Monthly data',
      apply: 'param',
      default: 'sst',
      help: 'Choose NOAA’s reconstructed SST or its supplied anomaly relative to the 1971–2000 climatology. The anomaly is not a residual from the fitted six-year model.',
      options: [
        {value: 'anomaly', label: 'NOAA supplied anomaly (°C)'},
        {value: 'sst', label: 'Reconstructed SST (°C)'}
      ]
    },
    {
      kind: 'slider',
      id: 'month',
      label: 'Month on map',
      group: 'Monthly data',
      apply: 'param',
      min: 0,
      max: 71,
      step: 1,
      default: 42,
      format: value => DATES[Math.round(value)] ?? String(value),
      help: 'Selects one of the 72 actual monthly samples for observed, smoothed and component maps.'
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Map shows',
      group: 'Map output',
      apply: 'param',
      default: 'seasonality',
      help: 'Inspect a monthly series or one seasonal-break model output. Components refer to the selected monthly slot.',
      options: [
        {value: 'observed', label: 'Observed'},
        {value: 'smoothed', label: 'Savitzky–Golay smooth'},
        {value: 'trend', label: 'Trend + selected jump'},
        {value: 'seasonality', label: 'Annual seasonal component'},
        {value: 'residual', label: 'Model residual'},
        {value: 'break-month', label: 'Selected break month'},
        {value: 'bic-gain', label: 'BIC improvement'},
        {value: 'fit-status', label: 'Fit status'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Map output',
      apply: 'param',
      default: 'cividis',
      help: 'Color ramp for continuous values. Fit status uses its fixed categorical colors.',
      options: [
        {value: 'cividis', label: 'Cividis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'grayscale', label: 'Grayscale'},
        {value: 'diverging', label: 'Diverging'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Map output',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Reveals more basemap under the coarse grid.'
    },
    {
      kind: 'slider',
      id: 'smoothWindow',
      label: 'SG window length',
      group: 'Savitzky–Golay · GPURasterTemporal',
      apply: 'compile',
      min: 7,
      max: 15,
      step: 2,
      default: 9,
      unit: 'months',
      help: 'Odd number of monthly slots in each local polynomial fit. This regular monthly record makes the filter’s time step meaningful; missing observations are excluded, and fitted values require sufficient support.'
    },
    {
      kind: 'slider',
      id: 'polynomialOrder',
      label: 'SG polynomial order',
      group: 'Savitzky–Golay · GPURasterTemporal',
      apply: 'compile',
      min: 1,
      max: 3,
      step: 1,
      default: 2,
      help: 'Degree of the local least-squares polynomial; it must be below the selected window length.'
    },
    {
      kind: 'select',
      id: 'edgeMode',
      label: 'SG edge handling',
      group: 'Savitzky–Golay · GPURasterTemporal',
      apply: 'compile',
      default: 'interp',
      help: 'At record edges, interp shifts a complete window inward; truncate keeps the centered window position and uses only in-range samples.',
      options: [
        {value: 'interp', label: 'Shift a full window inward'},
        {value: 'truncate', label: 'Truncate at edge'}
      ]
    },
    {
      kind: 'select',
      id: 'periodModel',
      label: 'Seasonality model',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      default: 'harmonic',
      help: 'Harmonic uses one annual sine/cosine pair; dummy fits a separate effect for each calendar-month phase. Both fit a linear trend plus at most one level shift.',
      options: [
        {value: 'harmonic', label: 'Harmonic annual cycle'},
        {value: 'dummy', label: '12 monthly phase effects'}
      ]
    },
    {
      kind: 'select',
      id: 'fitMethod',
      label: 'Fit method',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      default: 'joint',
      help: 'Joint solves the seasonal trend and one-break model together. Iterative alternates cyclic least-squares updates and reports convergence diagnostics.',
      options: [
        {value: 'joint', label: 'Joint fit'},
        {value: 'iterative', label: 'Iterative cyclic fit'}
      ]
    },
    {
      kind: 'select',
      id: 'missingPolicy',
      label: 'Missing-month policy',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      default: 'omit',
      help: 'Omit fits only observed months without compressing time; reject withholds models containing gaps; interpolate fills only interior training gaps and never future monitoring data.',
      options: [
        {value: 'omit', label: 'Omit missing months'},
        {value: 'reject', label: 'Reject incomplete series'},
        {value: 'interpolate', label: 'Interpolate interior gaps'}
      ]
    },
    {
      kind: 'slider',
      id: 'minimumSegmentLength',
      label: 'Minimum segment length',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      min: 6,
      max: 24,
      step: 3,
      default: 12,
      unit: 'months',
      help: 'Minimum elapsed slots and valid observations on both sides of a candidate jump.'
    },
    {
      kind: 'slider',
      id: 'minimumBICGain',
      label: 'Minimum BIC improvement',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      min: 0,
      max: 12,
      step: 1,
      default: 4,
      help: 'Minimum BIC(no break) minus BIC(one break) required before the model retains a jump. Raising it is more conservative.'
    },
    {
      kind: 'slider',
      id: 'maximumIterations',
      label: 'Maximum iterations',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      min: 4,
      max: 128,
      step: 4,
      default: 32,
      disabledWhen: state => state.fitMethod !== 'iterative',
      help: 'Upper limit for the iterative solver; cells that reach it receive the iteration-limit status.'
    },
    {
      kind: 'slider',
      id: 'tolerance',
      label: 'Convergence tolerance',
      group: 'Single-break model · GPURasterSeasonalBreak',
      apply: 'compile',
      min: 0.000001,
      max: 0.001,
      step: 0.000001,
      default: 0.00001,
      format: value => value.toExponential(0),
      disabledWhen: state => state.fitMethod !== 'iterative',
      help: 'Relative convergence threshold for iterative updates.'
    }
  ],
  readouts: [
    {id: 'source', label: 'Source and grid'},
    {
      id: 'annualEvidence',
      label: 'NOAA annual anomaly means',
      help: 'Annual means of NOAA’s supplied monthly anomaly across valid ocean cells; these are direct data summaries, separate from the fitted model.'
    },
    {id: 'sgStatus', label: 'Savitzky–Golay support/status'},
    {id: 'fitStatus', label: 'GPU model status'},
    {
      id: 'fitDiagnostics',
      label: 'Model diagnostics',
      help: 'Actual per-cell status and diagnostics written by GPURasterSeasonalBreak.'
    },
    {id: 'pixelSeries', label: 'Clicked cell · observed and fitted series', kind: 'chart'},
    {id: 'selectedCell', label: 'Clicked cell · model summary', layout: 'block'}
  ],
  legends: state =>
    state.layer === 'fit-status'
      ? [
          {
            kind: 'categories',
            title: 'Model status',
            entries: [
              {color: OCEAN_MODEL_STATUS_COLORS[0], label: 'Fit'},
              {color: OCEAN_MODEL_STATUS_COLORS[1], label: 'Insufficient support'},
              {color: OCEAN_MODEL_STATUS_COLORS[2], label: 'Numerical failure'},
              {color: OCEAN_MODEL_STATUS_COLORS[4], label: 'Rejected gap'},
              {color: OCEAN_MODEL_STATUS_COLORS[8], label: 'Iteration limit'}
            ],
            note: 'Only the offline single-break model status is shown.'
          }
        ]
      : (() => {
          const signed =
            (state.series === 'anomaly' &&
              ['observed', 'smoothed', 'trend'].includes(state.layer)) ||
            ['seasonality', 'residual'].includes(state.layer);
          const extent: readonly [number, number] =
            state.layer === 'bic-gain'
              ? [0, 20]
              : state.layer === 'break-month'
                ? [0, 71]
                : signed
                  ? state.layer === 'seasonality' || state.layer === 'residual'
                    ? [-2, 2]
                    : [-3, 3]
                  : [-3, 32];
          return [
            {
              kind: 'ramp' as const,
              title:
                state.layer === 'bic-gain'
                  ? 'BIC improvement'
                  : state.layer === 'break-month'
                    ? 'Break month index'
                    : state.layer === 'seasonality'
                      ? 'Annual seasonal component'
                      : state.layer === 'residual'
                        ? 'Model residual'
                        : state.series === 'sst'
                          ? 'Sea-surface temperature (°C)'
                          : 'Temperature anomaly (°C)',
              ramp: signed ? ('diverging' as const) : state.ramp,
              extent,
              ...(signed ? {midpoint: 0} : {}),
              unit:
                state.layer === 'bic-gain'
                  ? 'BIC units'
                  : state.layer === 'break-month'
                    ? 'monthly index'
                    : '°C'
            }
          ];
        })(),
  story: [
    {
      id: 'monthly-record',
      headline: 'The northeastern Pacific stayed unusually warm through several winters',
      textAlternative:
        'A monthly sea-surface temperature map shows the strong seasonal cycle and warm period during 2014–2016.',
      title: 'Start with observed monthly sea-surface temperature',
      body: 'NOAA ERSSTv5 reconstructs monthly sea-surface temperature from in-situ observations. The grid covers **26 × 21 cells** at **2 degrees**, from **2012 through 2017**. The map starts with reconstructed SST; switch to NOAA’s supplied anomaly relative to the 1971–2000 climatology to view departures from the reference period. NOAA documented the **2014 warm anomaly** as a basin-scale event with multiple possible mechanisms; this small subset is context, not attribution.\n\nThe initial view selects **July 2015** and a Pacific cell near **50°N, 146°W**. Click a cell to plot all 72 observed months with the smoother and fitted seasonal components.',
      evidence:
        '**{{annualEvidence}}**. These cos(latitude)-weighted annual means come from NOAA’s anomaly array and show multi-year positive departures.',
      caveat:
        'This is a coarse monthly reconstruction, not satellite pixels, raw buoy readings or a daily event record.',
      options: {series: 'sst', layer: 'observed', month: 42},
      controls: ['series', 'month'],
      readouts: ['source', 'annualEvidence']
    },
    {
      id: 'seasonal-cycle',
      headline: 'A large seasonal cycle can coexist with a positive anomaly',
      textAlternative:
        'The fitted annual component varies within each year while the observed anomaly remains elevated.',
      title: 'Separate recurring seasonality from the monthly departure',
      body: 'The **Annual seasonal component** below comes from `GPURasterSeasonalBreak`, which models a 12-month seasonal cycle with a linear trend and at most one level shift. The per-pixel **Savitzky–Golay smoother** uses the regular monthly slots to reveal shorter local variation.\n\nUse **Seasonality model** to compare a harmonic cycle with 12 month effects. The clicked-cell chart keeps missing land samples as gaps; it does not treat them as zero.',
      evidence:
        '**{{fitStatus}}**, **{{sgStatus}}** and **{{fitDiagnostics}}** report model support and actual per-cell status.',
      caveat:
        'A six-year record covers only a few annual cycles, so seasonal coefficients and long-term trend remain sensitive to the chosen model.',
      options: {layer: 'seasonality', series: 'anomaly'},
      controls: ['layer', 'periodModel'],
      readouts: ['pixelSeries', 'fitStatus', 'sgStatus']
    },
    {
      id: 'single-break',
      headline: 'A BIC-selected level shift is a compact description of the record',
      textAlternative:
        'The model map highlights cells whose one-break fit improves on a no-break seasonal trend.',
      title: 'Inspect the one-break model and its evidence threshold',
      body: '`GPURasterSeasonalBreak` compares a no-break seasonal trend with candidate single-jump fits and selects a boundary only when it improves BIC enough. **Minimum segment length** below requires enough elapsed months and observed samples on both sides; **Minimum BIC improvement** raises or lowers the evidence threshold.\n\nMap **BIC improvement** or **Selected break month** and inspect the pixel chart’s trend-plus-jump line. The output is an offline model fit, not a detected ocean event.',
      evidence:
        '**{{fitStatus}}** · **{{fitDiagnostics}}**. A break month is retained only where its BIC gain clears the chosen threshold.',
      caveat:
        'One level shift cannot represent multiple regimes, and the model does not test a causal mechanism or attribute warmth to any driver.',
      options: {
        layer: 'bic-gain',
        periodModel: 'harmonic',
        fitMethod: 'joint',
        missingPolicy: 'omit',
        minimumSegmentLength: 12,
        minimumBICGain: 4
      },
      controls: ['layer', 'minimumSegmentLength', 'minimumBICGain'],
      readouts: ['fitStatus', 'fitDiagnostics']
    },
    {
      id: 'limits',
      headline: 'Use the components to describe, not classify, this ocean record',
      textAlternative:
        'The residual and status maps show what the compact model leaves unexplained.',
      title: 'Check residuals, gaps and solver status',
      body: 'The **Model residual** is the observed value after removing the fitted trend, jump and seasonality. **Fit status** distinguishes a solved model from insufficient support, numerical failure, a rejected gap or an iteration limit. For iterative fitting, change **Fit method** and inspect the convergence diagnostics.\n\nThe SST layer helps reveal the seasonal amplitude; the supplied anomaly emphasizes departure from a climatology. Neither one supports daily marine heatwave classification at this grid and cadence.',
      evidence:
        '**{{fitStatus}}**. The clicked cell chart displays observed values, SG smoothing and the fitted components over the actual monthly dates.',
      caveat:
        'Annual anomaly is NOAA’s external climatological reference, not the residual of this fitted six-year model.',
      options: {layer: 'residual', series: 'anomaly'},
      controls: ['layer', 'fitMethod'],
      readouts: ['pixelSeries', 'selectedCell']
    }
  ],
  about: {
    what: '`GPURasterTemporal` applies a regular-time Savitzky–Golay smoother. `GPURasterSeasonalBreak` fits a seasonal trend with at most one BIC-selected level shift and writes fitted components, break index, BIC improvement, status and diagnostics.',
    why: 'The monthly 2014–2016 northeast Pacific warm anomaly is a useful case for separating recurring seasonality from multi-month departures. The observed monthly series and modeled components remain available together for inspection.',
    howToRead:
      'The grid cells span two degrees and are drawn in an Albers equal-area projection. Break month and BIC maps are direct contributor outputs; no selected break is transparent. Model status colors distinguish fit, insufficient support, numerical failure, rejected gaps and iteration limits. Annual means use cosine-of-latitude weights across valid ocean cells. [NOAA’s ERSSTv5 dataset](https://doi.org/10.7289/V5T72FNM) is a monthly in-situ reconstruction.'
  },
  create: async ctx => (await import('./ocean-seasonality.compute')).createOceanSeasonality(ctx)
});
