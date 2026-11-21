// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';

/** Raster-native catalog: these bindings consume time stacks, never feature collections. */
export const RASTER_TIMESERIES_BINDINGS = [
  {
    id: 'rolling-mean',
    title: 'Rolling mean',
    contributors: ['GPURasterRollingMean'],
    defaults: {windowLength: 3},
    note: 'Trailing complete windows; equal acquisition spacing required.'
  },
  {
    id: 'whittaker',
    title: 'Whittaker smoother',
    contributors: ['GPURasterWhittakerSmoother'],
    defaults: {smoothingParameter: 10, differenceOrder: 2, reweightPassCount: 3},
    note: 'Divided differences on the acquisition axis; missing samples have zero weight.'
  },
  {
    id: 'hants',
    title: 'HANTS',
    contributors: ['GPURasterHANTS'],
    defaults: {period: 1, harmonicCount: 1, fitErrorTolerance: 1},
    note: 'Period in years; both-sided outlier rejection.'
  },
  {
    id: 'harmonic-regression',
    title: 'Harmonic regression',
    contributors: ['GPURasterHarmonicRegression'],
    defaults: {period: 1, harmonicCount: 1},
    note: 'Intercept, trend per year, cosine/sine pairs; summary is status and valid count.'
  },
  {
    id: 'bfast',
    title: 'BFAST monitor',
    contributors: ['GPURasterBFASTMonitor'],
    defaults: {
      period: 1,
      harmonicCount: 1,
      historySliceCount: 12,
      bandwidth: 0.25,
      criticalValue: 2
    },
    note: 'Critical value is illustrative: supply the published value for your bandwidth/horizon. Summary: break slice, status, history count; 4294967295 means no break.'
  },
  {
    id: 'cold',
    title: 'COLD monitor',
    contributors: ['GPURasterCOLDMonitor'],
    defaults: {
      period: 1,
      minimumObservations: 12,
      consecutiveCount: 6,
      changeThreshold: 15,
      outlierThreshold: 100,
      minimumRMSE: 1,
      maximumSegmentCount: 4
    },
    note: 'Simplified OLS COLD, not pyxccd parity; thresholds depend on the selected band count. Summary: segment count, status bits, fit count. Segment words: start/end/break/coefficient count.'
  },
  {
    id: 'medoid',
    title: 'Medoid composite',
    contributors: ['GPURasterMedoidComposite'],
    defaults: {},
    note: 'Up to 64 observations; complete multi-band observations only.'
  },
  {
    id: 'geometric-median',
    title: 'Geometric median composite',
    contributors: ['GPURasterGeometricMedianComposite'],
    defaults: {iterationCount: 16, distanceEpsilon: 0.00001},
    note: 'Bounded Weiszfeld iterations; lastStep reports convergence distance.'
  },
  {
    id: 'best-pixel',
    title: 'Best available pixel (date score)',
    contributors: ['GPURasterBestAvailablePixel'],
    defaults: {targetDayOfYear: 180, dayOfYearSigma: 45, targetYear: 2020, maximumYearDistance: 10},
    note: 'Date/year scoring only: the host has no cloud-distance metadata. Nodata rejects the whole observation. Scores do not imply cloud QA.'
  },
  {
    id: 'phenology',
    title: 'Phenology metrics',
    contributors: ['GPURasterWhittakerSmoother', 'GPURasterPhenologyMetrics'],
    defaults: {smoothingParameter: 10, startFraction: 0.25, endFraction: 0.15, minimumAmplitude: 0},
    note: 'One season after Whittaker smoothing; metric order: start/end/peak time, length, peak/base/amplitude, left/right slope, integral. Times in centred years.'
  },
  {
    id: 'streaming',
    title: 'Streaming accumulator',
    contributors: ['GPURasterStreamingAccumulator', 'GPURasterClimateAnomaly'],
    defaults: {slotCount: 1},
    note: 'A fresh bounded chunk per run; mean and sample standard deviation. streamingStatus certifies observation counts only; decoded streamingCountStatus surfaces count loss and affected statistics become NaN. Slots: 1 overall or 12 UTC months.'
  },
  {
    id: 'climate-anomaly',
    title: 'Climate anomaly',
    contributors: ['GPURasterStreamingAccumulator', 'GPURasterClimateAnomaly'],
    defaults: {slotCount: 1, calibrationSliceCount: 12, standardized: 1},
    note: 'Calibrate on the first N observations; anomaly over the entire stack. Slots with fewer than two samples cannot supply a sample deviation.'
  },
  {
    id: 'standardized-calibration',
    title: 'Standardized index calibration',
    contributors: ['GPURasterStandardizedIndexCalibration'],
    defaults: {slotCount: 1, calibrationSliceCount: 12, distribution: 'gamma', location: 0},
    note: 'APP fit: shape/scale/zero probability per slot; zeroAtom defaults on for gamma and off for Fisk; an explicit boolean overrides it. Calibration is the first N samples. No rolling accumulation is implied.'
  },
  {
    id: 'standardized-index',
    title: 'Standardized index',
    contributors: ['GPURasterStandardizedIndexCalibration', 'GPURasterStandardizedIndex'],
    defaults: {slotCount: 1, calibrationSliceCount: 12, distribution: 'gamma', location: 0},
    note: 'Gamma/Fisk APP fit from the first N samples, evaluated over the full stack. zeroAtom is a boolean (gamma defaults on, Fisk off). Choose a meaningful calibration period and climate variable; this does not infer precipitation or water balance.'
  },
  {
    id: 'resample',
    title: 'Temporal resample',
    contributors: ['GPURasterTemporalResample'],
    defaults: {gridCount: 24, maximumGap: 1},
    note: 'Linear interpolation to an evenly spaced grid spanning the input dates. maximumGap in years; outside support remains missing.'
  },
  {
    id: 'lagged-correlation',
    title: 'Lagged correlation',
    contributors: ['GPURasterLaggedCorrelation'],
    defaults: {lags: [0, 1], referenceIndex: [], minimumPairCount: 3},
    note: 'Supply an external reference series with one finite value per date; lags in equal-spaced samples. No reference is invented.'
  },
  {
    id: 'snapshot-covariance',
    title: 'Snapshot covariance',
    contributors: ['GPURasterSnapshotCovariance'],
    defaults: {},
    note: 'Complete cases, equal cell weights; global temporal Gram matrix, not an area-weighted covariance. Published as a table.'
  },
  {
    id: 'eof',
    title: 'EOF projection',
    contributors: ['GPURasterSnapshotCovariance', 'GPURasterEOFProjection'],
    defaults: {modeCount: 2},
    note: 'Equal cell weights; reads the small temporal Gram matrix for the library CPU eigen step, then projects on the GPU. Missing dates exclude the whole cell.'
  },
  {
    id: 'climate-runs',
    title: 'Climate runs',
    contributors: ['GPURasterClimateRunAccumulator'],
    defaults: {threshold: 0, minimumSpellLength: 3},
    note: 'Above-threshold spells on daily acquisitions only; run fields: current/longest/exceedance/spell count/spell samples/missing. Degree days sum daily positive excess.'
  }
] as const;

export type RasterTimeSeriesOperation = (typeof RASTER_TIMESERIES_BINDINGS)[number]['id'];
export type RasterTimeSeriesSettings = Readonly<
  Record<string, number | string | boolean | readonly number[]> & {
    /** Explicit original-zero atom, defaulting to true for gamma and false for Fisk. */
    zeroAtom?: boolean;
  }
>;

export function getRasterTimeSeriesBinding(
  operation: RasterTimeSeriesOperation,
  settings?: RasterTimeSeriesSettings
) {
  validateRequirement(
    `raster-timeseries-${operation}`,
    settings === undefined || !('zeroInflated' in settings),
    'zeroInflated was removed; use zeroAtom'
  );
  const binding = RASTER_TIMESERIES_BINDINGS.find(candidate => candidate.id === operation);
  if (!binding) throw new Error(`Unknown raster time-series operation: ${operation}`);
  return binding;
}

/** Contributor inventory locations for the raster-native panel. */
export const RASTER_TIMESERIES_PANEL_OPERATORS: Readonly<Record<string, string>> =
  Object.fromEntries(
    RASTER_TIMESERIES_BINDINGS.flatMap(binding =>
      binding.contributors.map(name => [name, 'Raster tools · Raster time series'])
    )
  );

/** Output scope is known before execution, including the EOF diagnostic tables. */
export function isRasterTimeSeriesGlobalOutput(
  operation: RasterTimeSeriesOperation,
  outputName = ''
): boolean {
  return (
    (operation === 'snapshot-covariance' &&
      ['', 'covariance', 'keptCellCount'].includes(outputName)) ||
    (operation === 'eof' &&
      ['covariance', 'keptCellCount', 'explainedVarianceRatio'].includes(outputName))
  );
}
