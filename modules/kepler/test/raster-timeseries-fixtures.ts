// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {
  RasterTimeStack,
  RasterTimeSeriesOperation,
  RasterTimeSeriesSettings,
  RasterTimeSeriesResult
} from '../src/analysis/raster-timeseries';
import monitorReference from '../../analysis/test/oracles/fixtures/rts04-change-monitoring-numpy.json';

export type RasterTimeSeriesFixture = {
  name: string;
  operation: RasterTimeSeriesOperation;
  stack: RasterTimeStack;
  settings?: RasterTimeSeriesSettings;
  numerical: Record<string, readonly number[]>;
  integers?: Record<string, readonly number[]>;
  tolerance?: number;
  tolerances?: Record<string, number>;
};
const DAY_MILLISECONDS = 86400000;
const YEAR_DAYS = 365.2425;
function scalarStack(
  values: readonly number[],
  times = values.map((_, index) => index * DAY_MILLISECONDS)
): RasterTimeStack {
  return {
    cellCount: 1,
    sliceCount: values.length,
    bandCount: 1,
    values: Float32Array.from(values),
    times
  };
}

/** Hand-computed analytic cases, SciPy Fisk quantiles, and the existing independent NumPy monitor fixtures. */
export function makeRasterTimeSeriesFixtures(): RasterTimeSeriesFixture[] {
  const times = Array.from({length: 12}, (_, slice) =>
    Math.round((slice * YEAR_DAYS * DAY_MILLISECONDS) / 12)
  );
  const years = times.map(
    time => (time - (times[0] + times[11]) / 2) / (YEAR_DAYS * DAY_MILLISECONDS)
  );
  const harmonic = years.map(
    year => 10 + Math.cos(2 * Math.PI * year) + 0.5 * Math.sin(2 * Math.PI * year)
  );
  const trend = years.map((year, index) => harmonic[index] + 0.2 * year);
  const shifted = scalarStack([-5, -4, -3, -2, -1, 0]);
  // SciPy 1.x: Fisk APP shape/scale from mean=7.5, variance=35/12; norm.ppf(fisk.cdf(5..10)).
  const fit = [7.965391193415794, 7.310469314079422, 0];
  const indices = [
    -1.6821136202454432, -0.947424296045906, -0.2161607856234702, 0.44587469713376315,
    0.9933049093978062, 1.431225634542203
  ];
  const calibration = {calibrationSliceCount: 6, distribution: 'fisk', location: -10};
  const covariance = {
    cellCount: 2,
    sliceCount: 3,
    bandCount: 1,
    times: [0, DAY_MILLISECONDS, 2 * DAY_MILLISECONDS],
    values: new Float32Array([1, 2, 3, 2, 4, 6])
  };
  const matrix = [5, 0, -5, 0, 0, 0, -5, 0, 5];
  const bfast = monitorReference.bfastMonitor[0];
  const cold = monitorReference.cold[0];
  const coldStatistics = Array.from({length: cold.config.maximumSegmentCount}, (_, segment) =>
    Array.from({length: cold.config.bandCount}, (_, band) => [
      cold.simplified.segments[segment]?.rmse[band] ?? NaN,
      cold.simplified.segments[segment]?.changeMagnitude?.[band] ?? NaN
    ]).flat()
  ).flat();
  return [
    {
      name: 'rolling-mean',
      operation: 'rolling-mean',
      stack: scalarStack([1, 2, 3, 4]),
      numerical: {rollingMean: [NaN, NaN, 2, 3]}
    },
    {
      name: 'W5-17 window one short chunk keeps two-cell width-two pitch',
      operation: 'rolling-mean',
      stack: {
        cellCount: 2,
        sliceCount: 2,
        bandCount: 1,
        times: [0, 1],
        values: Float32Array.of(12, -999, 22, -999)
      },
      settings: {windowLength: 1, activeSliceCount: 1},
      numerical: {rollingMean: [12, NaN, 22, NaN]}
    },
    {
      name: 'B4 rolling ignores oversized inactive padding',
      operation: 'rolling-mean',
      stack: {
        cellCount: 2,
        sliceCount: 3,
        bandCount: 1,
        times: [0, 1, 2],
        values: Float32Array.of(2, 6, 1e22, 20, 60, 1e22)
      },
      settings: {windowLength: 1, activeSliceCount: 2},
      numerical: {rollingMean: [2, 6, NaN, 20, 60, NaN]}
    },
    {
      name: 'whittaker',
      operation: 'whittaker',
      stack: scalarStack([1, 2, 3, 4]),
      settings: {smoothingParameter: 0, reweightPassCount: 0},
      numerical: {smoothed: [1, 2, 3, 4]},
      integers: {smoothingStatus: [0]}
    },
    {
      name: 'hants',
      operation: 'hants',
      stack: scalarStack(harmonic, times),
      settings: {fitErrorTolerance: 0.001},
      numerical: {smoothed: Array.from(Float32Array.from(harmonic)), coefficients: [10, 1, 0.5]},
      integers: {status: [0]}
    },
    {
      name: 'harmonic-regression',
      operation: 'harmonic-regression',
      stack: scalarStack(trend, times),
      numerical: {fitted: Array.from(Float32Array.from(trend)), coefficients: [10, 0.2, 1, 0.5]},
      integers: {summary: [0, 12]}
    },
    {
      name: 'bfast',
      operation: 'bfast',
      stack: scalarStack(
        bfast.values.map(value => value ?? NaN),
        bfast.timesSeconds.map(time => time * 1000)
      ),
      settings: {
        historySliceCount: bfast.historyCount,
        harmonicCount: bfast.harmonicCount,
        bandwidth: bfast.bandwidth,
        criticalValue: bfast.criticalValue
      },
      numerical: {
        scores: [bfast.expected.magnitude, bfast.expected.maximumBoundaryRatio],
        coefficients: bfast.expected.coefficients
      },
      integers: {
        cellSummary: [
          bfast.expected.breakSlice ?? 0xffffffff,
          bfast.expected.status,
          bfast.expected.historyObservationCount
        ]
      },
      tolerance: 0.001,
      tolerances: {scores: 0.02}
    },
    {
      name: 'cold',
      operation: 'cold',
      stack: {
        cellCount: 1,
        sliceCount: cold.values.length,
        bandCount: cold.config.bandCount,
        times: cold.timesSeconds.map(time => time * 1000),
        values: Float32Array.from(cold.values.flat().map(value => value ?? NaN))
      },
      settings: {
        consecutiveCount: cold.config.consecutive,
        changeThreshold: cold.config.changeThreshold,
        outlierThreshold: cold.config.outlierThreshold,
        minimumRMSE: cold.config.minimumRMSE[0]
      },
      numerical: {segmentStatistics: coldStatistics},
      integers: {
        cellSummary: [
          cold.simplified.segments.length,
          cold.simplified.status,
          cold.simplified.fitCount
        ]
      },
      tolerance: 0.001
    },
    {
      name: 'medoid',
      operation: 'medoid',
      stack: scalarStack([1, 2, 3]),
      numerical: {composite: [2]},
      integers: {status: [0], sliceIndex: [1]}
    },
    {
      name: 'geometric-median',
      operation: 'geometric-median',
      stack: scalarStack([1, 2, 3]),
      numerical: {composite: [2]},
      integers: {status: [0]}
    },
    {
      name: 'best-pixel',
      operation: 'best-pixel',
      stack: scalarStack(
        [1, 2, 3],
        [Date.UTC(2020, 0, 1), Date.UTC(2020, 1, 1), Date.UTC(2020, 2, 1)]
      ),
      settings: {targetDayOfYear: 32},
      numerical: {composite: [2], bestScore: [2]},
      integers: {bestSlice: [1]}
    },
    {
      name: 'phenology',
      operation: 'phenology',
      stack: scalarStack([0, 0, 2, 4, 2, 0, 0]),
      settings: {smoothingParameter: 0},
      numerical: {
        metrics: [
          -1.5 / YEAR_DAYS,
          1.7 / YEAR_DAYS,
          0,
          3.2 / YEAR_DAYS,
          4,
          0,
          4,
          2 * YEAR_DAYS,
          -2 * YEAR_DAYS,
          7.66 / YEAR_DAYS
        ]
      },
      integers: {smoothingStatus: [0], status: [0], seasonCount: [1], flags: [0]},
      tolerance: 0.001
    },
    {
      name: 'streaming',
      operation: 'streaming',
      stack: scalarStack([1, 2, 3]),
      numerical: {mean: [2], standardDeviation: [1]},
      integers: {counts: [3], streamingStatus: [0]}
    },
    {
      name: 'climate-anomaly',
      operation: 'climate-anomaly',
      stack: scalarStack([1, 2, 3, 10, 11, NaN]),
      settings: {calibrationSliceCount: 3},
      numerical: {mean: [2], standardDeviation: [1], anomaly: [-1, 0, 1, 8, 9, NaN]},
      integers: {counts: [3], streamingStatus: [0]}
    },
    {
      name: 'shifted-fisk-calibration',
      operation: 'standardized-calibration',
      stack: shifted,
      settings: calibration,
      numerical: {fit},
      integers: {counts: [6, 0, 6]}
    },
    {
      name: 'shifted-fisk-index',
      operation: 'standardized-index',
      stack: shifted,
      settings: calibration,
      numerical: {fit, index: indices},
      integers: {counts: [6, 0, 6], status: [0, 0, 0, 0, 0, 0]}
    },
    {
      name: 'resample',
      operation: 'resample',
      stack: scalarStack([0, 10]),
      settings: {gridCount: 3},
      numerical: {resampled: [0, 5, 10]},
      integers: {status: [0, 0, 0]}
    },
    {
      name: 'lagged-correlation',
      operation: 'lagged-correlation',
      stack: scalarStack([1, 2, 3, 4]),
      settings: {referenceIndex: [1, 2, 3, 4]},
      numerical: {correlations: [1, 1]},
      integers: {status: [0, 0], pairCount: [4, 3]}
    },
    {
      name: 'snapshot-covariance',
      operation: 'snapshot-covariance',
      stack: covariance,
      numerical: {covariance: matrix},
      integers: {keptCellCount: [2]}
    },
    {
      name: 'eof',
      operation: 'eof',
      stack: covariance,
      settings: {modeCount: 1},
      numerical: {
        covariance: matrix,
        eofs: [-1 / Math.sqrt(5), -2 / Math.sqrt(5)],
        explainedVarianceRatio: [1]
      },
      integers: {status: [0, 0]}
    },
    {
      name: 'climate-runs-threshold-equality',
      operation: 'climate-runs',
      stack: scalarStack([0, 0, 0]),
      numerical: {degreeDays: [0]},
      integers: {runs: [0, 0, 0, 0, 0, 0]}
    },
    // The sample spread is nine float32 ULPs: allow input-scale rounding in Welford's mean updates.
    {
      name: 'tiny-climate-variance',
      operation: 'climate-anomaly',
      stack: scalarStack([1e-6, 1.000001e-6, 1.000002e-6]),
      settings: {calibrationSliceCount: 3},
      numerical: {standardDeviation: [1.0231815394945443e-12], anomaly: [-1, 0, 1]},
      integers: {counts: [3], streamingStatus: [0]},
      tolerances: {standardDeviation: 1e-13, anomaly: 0.06}
    }
  ];
}

/** Numerical arrays must match the oracle; finite status words cannot stand in for a usable result. */
export function expectRasterTimeSeriesFixture(
  result: RasterTimeSeriesResult,
  fixture: RasterTimeSeriesFixture
): void {
  for (const [name, expected] of Object.entries(fixture.numerical)) {
    expect(
      expected.some(Number.isFinite),
      `${fixture.name}.${name} oracle has numerical support`
    ).toBe(true);
    const actual = result.columns[name]?.values;
    expect(actual, `${fixture.name}.${name} exists`).toBeDefined();
    expect(actual.length).toBe(expected.length);
    expected.forEach((value, index) => {
      if (Number.isNaN(value))
        expect(Number.isNaN(actual[index]), `${name}[${index}] missing`).toBe(true);
      else {
        expect(Number.isFinite(actual[index]), `${fixture.name}.${name}[${index}] finite`).toBe(
          true
        );
        expect(Math.abs(actual[index] - value), `${fixture.name}.${name}[${index}]`).toBeLessThan(
          fixture.tolerances?.[name] ?? fixture.tolerance ?? 2e-5
        );
      }
    });
  }
  for (const [name, expected] of Object.entries(fixture.integers ?? {}))
    expect(Array.from(result.columns[name].values), `${fixture.name}.${name}`).toEqual(expected);
}
