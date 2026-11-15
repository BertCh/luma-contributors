// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  assignGPUAnalysisFieldRoles,
  assignGPUAnalysisRatePair,
  guessGPUAnalysisCoordinateModel,
  guessGPUAnalysisMeasurementLevel,
  guessGPUAnalysisRowSemantics,
  matchGPUAnalysisFieldRole,
  suggestGPUAnalysisRatePairs,
  tokenizeGPUAnalysisColumnName
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-data-roles';
import {
  GPU_ANALYSIS_MEASUREMENT_LEVELS,
  type GPUAnalysisColumnProfile,
  type GPUAnalysisColumnStatistics,
  type GPUAnalysisColumnType,
  type GPUAnalysisFieldRoleSpec,
  type GPUAnalysisMeasurementLevel,
  type GPUAnalysisDataProfile,
  type GPUAnalysisRowSemantics
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';

type StatisticsOptions = {
  rows?: number;
  nulls?: number;
  min?: number;
  max?: number;
  mean?: number;
  distinct?: number;
  estimate?: boolean;
  integral?: boolean;
  sample?: number;
};

function makeColumn(
  name: string,
  type: GPUAnalysisColumnType,
  options: StatisticsOptions = {}
): GPUAnalysisColumnProfile {
  const statistics: GPUAnalysisColumnStatistics = {
    name,
    type,
    rowCount: options.rows ?? 1000,
    nullCount: options.nulls ?? 0
  };
  if (options.min !== undefined) statistics.minimum = options.min;
  if (options.max !== undefined) statistics.maximum = options.max;
  if (options.mean !== undefined) statistics.mean = options.mean;
  if (options.distinct !== undefined) statistics.distinctCount = options.distinct;
  if (options.estimate) statistics.distinctCountIsEstimate = true;
  if (options.integral !== undefined) statistics.integral = options.integral;
  if (options.sample !== undefined) statistics.sampleSize = options.sample;
  return statistics;
}

describe('tokenizeGPUAnalysisColumnName', () => {
  const cases: readonly (readonly [string, readonly string[]])[] = [
    ['snake_case_name', ['snake', 'case', 'name']],
    ['kebab-case-name', ['kebab', 'case', 'name']],
    ['camelCaseName', ['camel', 'case', 'name']],
    ['PascalCaseName', ['pascal', 'case', 'name']],
    ['PULocationID', ['pu', 'location', 'id']],
    ['VendorID', ['vendor', 'id']],
    ['OBJECTID', ['objectid']],
    ['GEOID10', ['geoid', '10']],
    ['pop2020', ['pop', '2020']],
    ['cases_per_100k', ['cases', 'per', '100', 'k']],
    ['HTTPServerError', ['http', 'server', 'error']],
    ['  spaced  name ', ['spaced', 'name']],
    ['dot.separated.name', ['dot', 'separated', 'name']],
    ['__leading_and_trailing__', ['leading', 'and', 'trailing']],
    ['accident', ['accident']],
    ['created', ['created']],
    ['x', ['x']],
    ['', []],
    ['___', []],
    ['Über_größe', ['über', 'größe']],
    ['wind_dir_deg', ['wind', 'dir', 'deg']],
    ['a1b2', ['a', '1', 'b', '2']]
  ];
  for (const [name, expected] of cases) {
    it(`splits ${JSON.stringify(name)}`, () => {
      expect(tokenizeGPUAnalysisColumnName(name)).toEqual(expected);
    });
  }

  it('never lets a word match inside another word', () => {
    expect(tokenizeGPUAnalysisColumnName('accident')).not.toContain('id');
    expect(tokenizeGPUAnalysisColumnName('created')).not.toContain('rate');
    expect(tokenizeGPUAnalysisColumnName('rated')).not.toContain('rate');
    expect(tokenizeGPUAnalysisColumnName('paid')).not.toContain('id');
    expect(tokenizeGPUAnalysisColumnName('discount')).not.toContain('count');
    expect(tokenizeGPUAnalysisColumnName('grid')).not.toContain('id');
  });
});

describe('guessGPUAnalysisMeasurementLevel', () => {
  it('maps one probe name per level to that level', () => {
    // The word lists are private; probe them through names that must each map to one level.
    const probes: readonly (readonly [string, GPUAnalysisMeasurementLevel])[] = [
      ['fips', 'identifier'],
      ['density', 'rate'],
      ['heading', 'cyclic'],
      ['rank', 'ordinal'],
      ['cases', 'count'],
      ['revenue', 'amount'],
      ['county', 'nominal']
    ];
    for (const [name, level] of probes) {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn(name, 'integer', {min: 1, max: 5, distinct: 5, rows: 100})
      );
      expect(guess?.level, name).toBe(level);
    }
  });

  it('returns undefined for time columns and columns without evidence', () => {
    expect(
      guessGPUAnalysisMeasurementLevel(makeColumn('created', 'time', {min: 0, max: 5}))
    ).toBeUndefined();
    expect(
      guessGPUAnalysisMeasurementLevel(makeColumn('empty', 'number', {rows: 10, nulls: 10}))
    ).toBeUndefined();
    expect(
      guessGPUAnalysisMeasurementLevel(makeColumn('empty', 'text', {rows: 10, nulls: 10}))
    ).toBeUndefined();
    expect(
      guessGPUAnalysisMeasurementLevel(makeColumn('rows_zero', 'number', {rows: 0}))
    ).toBeUndefined();
    expect(guessGPUAnalysisMeasurementLevel(makeColumn('mystery', 'number'))).toBeUndefined();
  });

  it('ignores non-finite ranges', () => {
    const guess = guessGPUAnalysisMeasurementLevel(
      makeColumn('value', 'number', {min: Number.NaN, max: Number.POSITIVE_INFINITY})
    );
    expect(guess).toBeUndefined();
  });

  it('does not call a constant integer column an identifier', () => {
    const guess = guessGPUAnalysisMeasurementLevel(
      makeColumn('constant', 'integer', {min: 7, max: 7, distinct: 1})
    );
    expect(guess).toBeUndefined();
  });

  it('gives every guess a reason sentence and a valid level', () => {
    const guess = guessGPUAnalysisMeasurementLevel(
      makeColumn('population', 'integer', {min: 0, max: 9000})
    );
    expect(guess?.reason.length).toBeGreaterThan(20);
    expect(GPU_ANALYSIS_MEASUREMENT_LEVELS).toContain(guess?.level);
    expect(guess?.reason).toContain('population');
  });

  describe('identifiers', () => {
    it('reads all-distinct integers without gaps as a low-confidence identifier (values only)', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('row', 'integer', {rows: 500, min: 1, max: 500, distinct: 500})
      );
      expect(guess).toMatchObject({level: 'identifier', confidence: 'low'});
    });

    it('reads all-distinct integers with gaps as a low-confidence identifier', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('seq', 'integer', {rows: 500, min: 1, max: 90000, distinct: 500})
      );
      expect(guess).toMatchObject({level: 'identifier', confidence: 'low'});
    });

    it('lets a count name beat all-distinct values (population of 50 states)', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('population', 'integer', {rows: 50, min: 570000, max: 39000000, distinct: 50})
      );
      expect(guess).toMatchObject({level: 'count', confidence: 'high'});
    });

    it('trusts an estimated distinct count only near the row count', () => {
      const near = guessGPUAnalysisMeasurementLevel(
        makeColumn('seq', 'integer', {
          rows: 1000,
          min: 1,
          max: 99999,
          distinct: 990,
          estimate: true
        })
      );
      const far = guessGPUAnalysisMeasurementLevel(
        makeColumn('seq', 'integer', {
          rows: 1000,
          min: 1,
          max: 99999,
          distinct: 800,
          estimate: true
        })
      );
      expect(near?.level).toBe('identifier');
      expect(far?.level).toBe('count');
    });

    it('does not call a single row an identifier from distinctness alone', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('thing', 'integer', {rows: 1, min: 4, max: 4, distinct: 1})
      );
      expect(guess).toBeUndefined();
    });

    it('is low confidence for few rows', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('t', 'text', {rows: 5, distinct: 5})
      );
      expect(guess).toMatchObject({level: 'identifier', confidence: 'low'});
    });

    it('uses the sample size for the row count when statistics are sampled', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('seq', 'integer', {
          rows: 1_000_000,
          sample: 100,
          min: 1,
          max: 100,
          distinct: 100
        })
      );
      expect(guess).toMatchObject({level: 'identifier', confidence: 'low'});
    });
  });

  describe('counts and amounts', () => {
    it('reads total with fractions as an amount', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('total', 'number', {min: 0, max: 99.5, integral: false, distinct: 80})
      );
      expect(guess).toMatchObject({level: 'amount', confidence: 'low'});
    });

    it('keeps a count name with negative values as a low-confidence count', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('cases', 'integer', {min: -3, max: 40, distinct: 30})
      );
      expect(guess).toMatchObject({level: 'count', confidence: 'low'});
    });

    it('marks a weak count word on a number column of unknown integrality low', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('total', 'number', {min: 0, max: 100})
      );
      expect(guess).toMatchObject({level: 'count', confidence: 'low'});
    });

    it('marks an amount with negative values low', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('revenue', 'number', {min: -20, max: 900, integral: false})
      );
      expect(guess).toMatchObject({level: 'amount', confidence: 'low'});
    });
  });

  describe('cyclic range checks', () => {
    it('rejects hour with values beyond 24', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('hour', 'integer', {min: 0, max: 5000, distinct: 400})
      );
      expect(guess?.level).not.toBe('cyclic');
    });

    it('rejects a heading with the 511 sentinel as not cyclic (known limit)', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('heading', 'integer', {min: 0, max: 511, distinct: 360})
      );
      expect(guess?.level).not.toBe('cyclic');
    });

    it('is low confidence without a range', () => {
      const guess = guessGPUAnalysisMeasurementLevel(makeColumn('bearing', 'number'));
      expect(guess).toMatchObject({level: 'cyclic', confidence: 'low'});
    });

    it('accepts hours that run 1 to 24 and signed angles', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('hour', 'integer', {min: 1, max: 24, distinct: 24})
        )?.level
      ).toBe('cyclic');
      expect(
        guessGPUAnalysisMeasurementLevel(makeColumn('bearing', 'number', {min: -180, max: 180}))
          ?.level
      ).toBe('cyclic');
    });

    it('does not call hours (a duration) cyclic', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('hours', 'number', {min: 0, max: 12.5, integral: false})
      );
      expect(guess?.level).not.toBe('cyclic');
    });

    it('does not read aspect_ratio as an angle', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('aspect_ratio', 'number', {min: 0.2, max: 8, integral: false})
      );
      expect(guess?.level).toBe('rate');
    });

    it('reads day_of_week through its phrase', () => {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn('day_of_week', 'integer', {min: 0, max: 6, distinct: 7})
      );
      expect(guess?.level).toBe('cyclic');
    });
  });

  describe('value rules with no name evidence', () => {
    it('reads two integer values as nominal', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('flag', 'integer', {min: 0, max: 1, distinct: 2})
        )
      ).toMatchObject({level: 'nominal', confidence: 'low'});
    });

    it('reads few integer values as ordinal and many as a count', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('score5', 'integer', {min: 1, max: 5, distinct: 5})
        )
      ).toMatchObject({level: 'ordinal'});
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('thing', 'integer', {min: 0, max: 400, distinct: 11})
        )
      ).toMatchObject({level: 'count', confidence: 'low'});
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('thing', 'integer', {min: 0, max: 400, distinct: 10})
        )
      ).toMatchObject({level: 'ordinal'});
    });

    it('treats the boundaries of the rate ranges consistently', () => {
      const at = (maximum: number, minimum = 0) =>
        guessGPUAnalysisMeasurementLevel(
          makeColumn('v', 'number', {min: minimum, max: maximum, integral: false, distinct: 500})
        )?.level;
      expect(at(1)).toBe('rate');
      expect(at(1.0001)).toBe('rate');
      expect(at(100)).toBe('rate');
      expect(at(100.0001)).toBe('amount');
      expect(at(100, -0.0001)).toBeUndefined();
    });

    it('gives no level to negative fractions', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('v', 'number', {min: -3, max: 8, integral: false})
        )
      ).toBeUndefined();
    });

    it('needs a range for a fractional column', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(makeColumn('v', 'number', {integral: false}))
      ).toBeUndefined();
    });
  });

  describe('category columns', () => {
    it('reads boolean and category columns as nominal', () => {
      expect(guessGPUAnalysisMeasurementLevel(makeColumn('paid', 'boolean'))).toMatchObject({
        level: 'nominal',
        confidence: 'high'
      });
      expect(
        guessGPUAnalysisMeasurementLevel(makeColumn('county', 'category', {distinct: 40}))
      ).toMatchObject({level: 'nominal', confidence: 'high'});
    });

    it('reads a text column named by an identifier word as an identifier even when values repeat', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(
          makeColumn('county_fips', 'text', {rows: 5000, distinct: 60})
        )
      ).toMatchObject({level: 'identifier', confidence: 'high'});
    });

    it('is low confidence for repeating free text', () => {
      expect(
        guessGPUAnalysisMeasurementLevel(makeColumn('place', 'text', {rows: 5000, distinct: 300}))
      ).toMatchObject({level: 'nominal', confidence: 'low'});
    });
  });
});

// ---------------------------------------------------------------------------------------------
// Accuracy table
// ---------------------------------------------------------------------------------------------

type TableRow = {
  dataset: string;
  column: GPUAnalysisColumnProfile;
  /** Level a careful analyst assigns, decided before the rules were run. undefined: none. */
  expected: GPUAnalysisMeasurementLevel | undefined;
};

const NUMBER_ROWS = 100_000;

function row(
  dataset: string,
  name: string,
  type: GPUAnalysisColumnType,
  expected: GPUAnalysisMeasurementLevel | undefined,
  options: StatisticsOptions = {}
): TableRow {
  return {dataset, column: makeColumn(name, type, {rows: NUMBER_ROWS, ...options}), expected};
}

const ACCURACY_TABLE: readonly TableRow[] = [
  // US census tracts, 2,300 rows
  row('census tracts', 'GEOID', 'text', 'identifier', {rows: 2300, distinct: 2300}),
  row('census tracts', 'STATEFP', 'text', 'nominal', {rows: 2300, distinct: 1}),
  row('census tracts', 'COUNTYFP', 'text', 'nominal', {rows: 2300, distinct: 30}),
  row('census tracts', 'total_pop', 'integer', 'count', {
    rows: 2300,
    min: 0,
    max: 14000,
    distinct: 2100
  }),
  row('census tracts', 'median_household_income', 'integer', 'rate', {
    rows: 2300,
    min: 9000,
    max: 250000,
    distinct: 2100
  }),
  row('census tracts', 'pct_poverty', 'number', 'rate', {
    rows: 2300,
    min: 0,
    max: 100,
    integral: false,
    distinct: 1900
  }),
  row('census tracts', 'pop_density', 'number', 'rate', {
    rows: 2300,
    min: 0,
    max: 120000,
    integral: false,
    distinct: 2290
  }),
  row('census tracts', 'ALAND', 'integer', 'amount', {
    rows: 2300,
    min: 10000,
    max: 900000000,
    distinct: 2290
  }),
  row('census tracts', 'households', 'integer', 'count', {
    rows: 2300,
    min: 0,
    max: 5000,
    distinct: 1800
  }),
  row('census tracts', 'unemployment_rate', 'number', 'rate', {
    rows: 2300,
    min: 0,
    max: 40,
    integral: false,
    distinct: 1500
  }),
  row('census tracts', 'median_age', 'number', 'rate', {
    rows: 2300,
    min: 18,
    max: 70,
    integral: false,
    distinct: 400
  }),
  row('census tracts', 'NAMELSAD', 'text', 'identifier', {rows: 2300, distinct: 2300}),
  row('census tracts', 'INTPTLAT', 'number', undefined, {
    rows: 2300,
    min: 24.5,
    max: 49.2,
    integral: false,
    distinct: 2300
  }),
  // Chicago crime incidents
  row('crime incidents', 'case_number', 'text', 'identifier', {distinct: NUMBER_ROWS}),
  row('crime incidents', 'id', 'integer', 'identifier', {
    min: 1,
    max: NUMBER_ROWS,
    distinct: NUMBER_ROWS
  }),
  row('crime incidents', 'primary_type', 'category', 'nominal', {distinct: 35}),
  row('crime incidents', 'arrest', 'boolean', 'nominal'),
  row('crime incidents', 'beat', 'integer', 'nominal', {min: 111, max: 2535, distinct: 300}),
  row('crime incidents', 'district', 'integer', 'nominal', {min: 1, max: 31, distinct: 23}),
  row('crime incidents', 'ward', 'integer', 'nominal', {min: 1, max: 50, distinct: 50}),
  row('crime incidents', 'year', 'integer', 'ordinal', {min: 2001, max: 2024, distinct: 24}),
  row('crime incidents', 'latitude', 'number', undefined, {
    min: 41.6,
    max: 42.02,
    integral: false,
    distinct: 90000
  }),
  row('crime incidents', 'longitude', 'number', undefined, {
    min: -87.94,
    max: -87.52,
    integral: false,
    distinct: 90000
  }),
  row('crime incidents', 'x_coordinate', 'integer', undefined, {
    min: 1_090_000,
    max: 1_205_000,
    distinct: 80000
  }),
  row('crime incidents', 'date', 'time', undefined, {min: 978307200, max: 1735689600}),
  row('crime incidents', 'community_area', 'integer', 'nominal', {min: 1, max: 77, distinct: 77}),
  // NYC taxi trips
  row('taxi trips', 'VendorID', 'integer', 'identifier', {min: 1, max: 2, distinct: 2}),
  row('taxi trips', 'passenger_count', 'integer', 'count', {min: 0, max: 9, distinct: 8}),
  row('taxi trips', 'trip_distance', 'number', 'amount', {
    min: 0,
    max: 300,
    integral: false,
    distinct: 5000
  }),
  row('taxi trips', 'fare_amount', 'number', 'amount', {
    min: -50,
    max: 500,
    integral: false,
    distinct: 6000
  }),
  row('taxi trips', 'tip_amount', 'number', 'amount', {
    min: 0,
    max: 200,
    integral: false,
    distinct: 3000
  }),
  row('taxi trips', 'payment_type', 'integer', 'nominal', {min: 1, max: 6, distinct: 5}),
  row('taxi trips', 'PULocationID', 'integer', 'identifier', {min: 1, max: 265, distinct: 260}),
  row('taxi trips', 'tpep_pickup_datetime', 'time', undefined, {min: 1, max: 2}),
  row('taxi trips', 'total_amount', 'number', 'amount', {
    min: 0,
    max: 900,
    integral: false,
    distinct: 9000
  }),
  row('taxi trips', 'store_and_fwd_flag', 'category', 'nominal', {distinct: 2}),
  row('taxi trips', 'trip_duration_s', 'integer', 'amount', {min: 0, max: 86400, distinct: 5000}),
  row('taxi trips', 'avg_speed', 'number', 'rate', {
    min: 0,
    max: 120,
    integral: false,
    distinct: 4000
  }),
  // Weather stations (GHCN daily)
  row('weather stations', 'STATION', 'text', 'identifier', {rows: 300000, distinct: 1000}),
  row('weather stations', 'elevation', 'number', undefined, {
    min: -10,
    max: 4000,
    integral: false,
    distinct: 900
  }),
  row('weather stations', 'TMAX', 'number', undefined, {
    min: -40,
    max: 50,
    integral: false,
    distinct: 700
  }),
  row('weather stations', 'PRCP', 'number', 'amount', {
    min: 0,
    max: 300,
    integral: false,
    distinct: 2000
  }),
  row('weather stations', 'wind_dir', 'integer', 'cyclic', {min: 0, max: 359, distinct: 360}),
  row('weather stations', 'humidity', 'number', 'rate', {
    min: 5,
    max: 100,
    integral: false,
    distinct: 900
  }),
  row('weather stations', 'pressure_hpa', 'number', undefined, {
    min: 950,
    max: 1050,
    integral: false,
    distinct: 900
  }),
  row('weather stations', 'wind_speed_ms', 'number', 'rate', {
    min: 0,
    max: 40,
    integral: false,
    distinct: 300
  }),
  row('weather stations', 'hour', 'integer', 'cyclic', {min: 0, max: 23, distinct: 24}),
  row('weather stations', 'month', 'integer', 'cyclic', {min: 1, max: 12, distinct: 12}),
  row('weather stations', 'day_of_week', 'integer', 'cyclic', {min: 0, max: 6, distinct: 7}),
  row('weather stations', 'temperature_c', 'number', undefined, {
    min: -30,
    max: 45,
    integral: false,
    distinct: 900
  }),
  // AIS vessel positions
  row('AIS', 'mmsi', 'integer', 'identifier', {
    rows: 4_000_000,
    min: 200_000_000,
    max: 780_000_000,
    distinct: 18000
  }),
  row('AIS', 'sog', 'number', 'rate', {min: 0, max: 102.2, integral: false, distinct: 1000}),
  row('AIS', 'cog', 'number', 'cyclic', {min: 0, max: 359.9, integral: false, distinct: 3600}),
  row('AIS', 'heading', 'integer', 'cyclic', {min: 0, max: 511, distinct: 361}),
  row('AIS', 'vessel_type', 'integer', 'nominal', {min: 0, max: 99, distinct: 40}),
  row('AIS', 'length', 'number', 'amount', {min: 0, max: 400, integral: false, distinct: 350}),
  row('AIS', 'width', 'number', 'amount', {min: 0, max: 60, integral: false, distinct: 250}),
  row('AIS', 'nav_status', 'integer', 'nominal', {min: 0, max: 15, distinct: 12}),
  // USGS earthquakes
  row('earthquakes', 'mag', 'number', undefined, {
    min: -1,
    max: 9.1,
    integral: false,
    distinct: 900
  }),
  row('earthquakes', 'depth', 'number', 'amount', {
    min: -3.4,
    max: 700,
    integral: false,
    distinct: 9000
  }),
  row('earthquakes', 'nst', 'integer', 'count', {min: 0, max: 900, distinct: 400}),
  row('earthquakes', 'gap', 'number', undefined, {
    min: 0,
    max: 360,
    integral: false,
    distinct: 3000
  }),
  row('earthquakes', 'magType', 'category', 'nominal', {distinct: 12}),
  row('earthquakes', 'id', 'text', 'identifier', {distinct: NUMBER_ROWS}),
  row('earthquakes', 'place', 'text', 'nominal', {distinct: 40000}),
  row('earthquakes', 'rms', 'number', undefined, {min: 0, max: 5, integral: false, distinct: 3000}),
  // Land parcels
  row('parcels', 'PARCEL_ID', 'text', 'identifier', {distinct: NUMBER_ROWS}),
  row('parcels', 'APN', 'text', 'identifier', {distinct: NUMBER_ROWS}),
  row('parcels', 'LOT_AREA_SQFT', 'number', 'amount', {
    min: 500,
    max: 2_000_000,
    integral: false,
    distinct: 20000
  }),
  row('parcels', 'YEAR_BUILT', 'integer', 'ordinal', {min: 1850, max: 2024, distinct: 170}),
  row('parcels', 'ASSESSED_VALUE', 'integer', 'amount', {
    min: 10000,
    max: 90_000_000,
    distinct: 50000
  }),
  row('parcels', 'ZONING', 'category', 'nominal', {distinct: 40}),
  row('parcels', 'ZIP', 'integer', 'identifier', {min: 90001, max: 96162, distinct: 1500}),
  row('parcels', 'NUM_UNITS', 'integer', 'count', {min: 1, max: 500, distinct: 90}),
  // Adversarial names
  row('adversarial', 'discount', 'number', 'rate', {
    min: 0,
    max: 0.4,
    integral: false,
    distinct: 40
  }),
  row('adversarial', 'discount_money', 'number', 'amount', {
    min: 0,
    max: 250,
    integral: false,
    distinct: 900
  }),
  row('adversarial', 'paid', 'boolean', 'nominal'),
  row('adversarial', 'paid_total', 'number', 'amount', {
    min: 0,
    max: 9999,
    integral: false,
    distinct: 5000
  }),
  row('adversarial', 'grid_id', 'integer', 'identifier', {
    rows: 4000,
    min: 1,
    max: 4000,
    distinct: 4000
  }),
  row('adversarial', 'accident_severity', 'integer', 'ordinal', {min: 1, max: 3, distinct: 3}),
  row('adversarial', 'rated', 'integer', 'ordinal', {min: 1, max: 5, distinct: 5}),
  row('adversarial', 'county', 'integer', 'nominal', {min: 1, max: 840, distinct: 60}),
  row('adversarial', 'pct_change', 'number', 'rate', {
    min: -80,
    max: 400,
    integral: false,
    distinct: 9000
  }),
  row('adversarial', 'lat', 'number', undefined, {
    min: 10,
    max: 40,
    integral: false,
    distinct: 9000
  }),
  row('adversarial', 'lon', 'number', undefined, {
    min: 100,
    max: 120,
    integral: false,
    distinct: 9000
  }),
  row('adversarial', 'injuries', 'integer', 'count', {min: 0, max: 20, distinct: 15}),
  row('adversarial', 'aspect', 'number', 'cyclic', {
    min: 0,
    max: 360,
    integral: false,
    distinct: 3600
  }),
  row('adversarial', 'aspect_ratio', 'number', 'rate', {
    min: 0.1,
    max: 10,
    integral: false,
    distinct: 900
  }),
  row('adversarial', 'slope_deg', 'number', undefined, {
    min: 0,
    max: 90,
    integral: false,
    distinct: 900
  }),
  row('adversarial', 'ndvi', 'number', undefined, {
    min: -0.2,
    max: 0.9,
    integral: false,
    distinct: 900
  }),
  row('adversarial', 'cases', 'integer', 'count', {min: 0, max: 5000, distinct: 800}),
  row('adversarial', 'cases_per_100k', 'number', 'rate', {
    min: 0,
    max: 5000,
    integral: false,
    distinct: 800
  }),
  row('adversarial', 'rank', 'integer', 'ordinal', {rows: 3000, min: 1, max: 3000, distinct: 3000}),
  row('adversarial', 'row_index', 'integer', 'identifier', {
    rows: 3000,
    min: 0,
    max: 2999,
    distinct: 3000
  }),
  row('adversarial', 'OBJECTID', 'integer', 'identifier', {
    rows: 3000,
    min: 1,
    max: 3000,
    distinct: 3000
  }),
  row('adversarial', 'country', 'text', 'nominal', {distinct: 1}),
  row('adversarial', 'ratio_water', 'number', 'rate', {
    min: 0,
    max: 1,
    integral: false,
    distinct: 900
  }),
  row('adversarial', 'acres', 'number', 'amount', {
    min: 0.1,
    max: 9000,
    integral: false,
    distinct: 900
  }),
  row('adversarial', 'is_weekend', 'integer', 'nominal', {min: 0, max: 1, distinct: 2}),
  row('adversarial', 'created_by', 'text', 'nominal', {distinct: 60}),
  row('adversarial', 'pop_2020', 'integer', 'count', {min: 0, max: 9000, distinct: 3000}),
  row('adversarial', 'income_per_capita', 'number', 'rate', {
    min: 8000,
    max: 190000,
    integral: false,
    distinct: 9000
  }),
  // Columns an independent reviewer found wrong, 3,000 rows through the real profiler
  row('reviewer', 'age', 'integer', undefined, {rows: 3000, min: 0, max: 99, distinct: 90}),
  row('reviewer', 'bedrooms', 'integer', 'count', {rows: 3000, min: 0, max: 6, distinct: 7}),
  row('reviewer', 'SHAPE_Leng', 'number', 'amount', {
    rows: 3000,
    min: 0.001002,
    max: 0.04999,
    integral: false,
    distinct: 3000
  }),
  row('reviewer', 'catch_kg', 'number', 'amount', {
    rows: 3000,
    min: 0,
    max: 50,
    integral: false,
    distinct: 2000
  }),
  row('reviewer', 'MEDHHINC', 'integer', 'rate', {
    rows: 3000,
    min: 9000,
    max: 250000,
    distinct: 2500
  }),
  row('reviewer', 'magnitude', 'number', undefined, {
    rows: 3000,
    min: 2,
    max: 7.5,
    integral: false,
    distinct: 60
  }),
  row('reviewer', 'dbh_cm', 'number', 'amount', {
    rows: 3000,
    min: 1,
    max: 89.9,
    integral: false,
    distinct: 800
  }),
  row('reviewer', 'STATEFP', 'integer', 'identifier', {rows: 3000, min: 1, max: 56, distinct: 51}),
  row('reviewer', 'ZCTA5CE10', 'integer', 'identifier', {
    rows: 3000,
    min: 1001,
    max: 99950,
    distinct: 2800
  }),
  row('reviewer', 'mmsi', 'integer', 'identifier', {
    rows: 3000,
    min: 200000000,
    max: 780000000,
    distinct: 150
  }),
  row('reviewer', 'parcel_no', 'integer', 'identifier', {
    rows: 3000,
    min: 1000,
    max: 990000,
    distinct: 2900
  }),
  row('reviewer', 'apn', 'integer', 'identifier', {
    rows: 3000,
    min: 100000000,
    max: 999999999,
    distinct: 2950
  }),
  row('reviewer', 'order_number', 'integer', 'identifier', {
    rows: 3000,
    min: 10000,
    max: 99999,
    distinct: 2900
  }),
  row('reviewer', 'PUMA', 'integer', 'identifier', {
    rows: 3000,
    min: 100,
    max: 4000,
    distinct: 120
  }),
  row('reviewer', 'tract', 'integer', 'identifier', {
    rows: 3000,
    min: 100,
    max: 990000,
    distinct: 800
  }),
  row('reviewer', 'beat', 'integer', 'nominal', {rows: 3000, min: 111, max: 2535, distinct: 300}),
  row('reviewer', 'precinct', 'integer', 'nominal', {rows: 3000, min: 1, max: 123, distinct: 77}),
  row('reviewer', 'route', 'integer', 'nominal', {rows: 3000, min: 1, max: 900, distinct: 120}),
  row('reviewer', 'severity_code', 'integer', 'ordinal', {rows: 3000, min: 1, max: 3, distinct: 3}),
  row('reviewer', 'timestamp', 'integer', undefined, {
    rows: 3000,
    min: 1600000000000,
    max: 1700000000000,
    distinct: 3000
  }),
  row('reviewer', 'unix_time', 'integer', undefined, {
    rows: 3000,
    min: 1600000000,
    max: 1700000000,
    distinct: 3000
  }),
  row('reviewer', 'date_int', 'integer', undefined, {
    rows: 3000,
    min: 20100101,
    max: 20241231,
    distinct: 2500
  }),
  row('reviewer', 'value', 'integer', undefined, {rows: 3000, min: 0, max: 900, distinct: 500}),
  row('reviewer', 'income', 'number', 'amount', {
    rows: 3000,
    min: 8000,
    max: 190000,
    integral: false,
    distinct: 3000
  }),
  row('adversarial', 'population_rank', 'integer', 'ordinal', {
    min: 1,
    max: 300,
    distinct: 300,
    rows: 300
  })
];

/**
 * Rows the rules get wrong, with the level they give and why. They are listed so that a change to
 * the rules that fixes or breaks one shows up here. Each is a limit of reading names and ranges,
 * not a bug to fix without new evidence.
 */
const KNOWN_LIMITS: Readonly<
  Record<string, {actual: GPUAnalysisMeasurementLevel | undefined; why: string}>
> = {
  'reviewer/MEDHHINC': {
    actual: 'count',
    why: 'A Census abbreviation (median household income) glued into one token: names are split into words, not abbreviations, and med- is too close to medical to read as median.'
  },
  'weather stations/STATION': {
    actual: 'nominal',
    why: 'A station label that repeats over days is text with repeating values; only an identifier word or all-distinct values mark identifiers, and the word station is not on the list.'
  },
  'AIS/sog': {
    actual: 'amount',
    why: 'Speed over ground in knots with a 102.2 sentinel maximum; the abbreviation is not on the word list and the range passes 100.'
  },
  'AIS/heading': {
    actual: 'count',
    why: 'The value 511 means not available, so the range exceeds 360 and the angle check rejects a column that is cyclic.'
  },
  'earthquakes/gap': {
    actual: 'amount',
    why: 'An azimuthal gap in degrees is an angle with no level; non-negative fractions read as an amount.'
  },
  'earthquakes/rms': {
    actual: 'rate',
    why: 'A residual in seconds that happens to lie within 0 to 100; values alone cannot tell a small measure from a rate.'
  },
  'adversarial/slope_deg': {
    actual: 'rate',
    why: 'Slope in degrees within 0 to 90 reads as a percentage by range; the word slope is not on the list.'
  }
};

describe('guessGPUAnalysisMeasurementLevel accuracy table', () => {
  const results = ACCURACY_TABLE.map(entry => ({
    key: `${entry.dataset}/${entry.column.name}`,
    entry,
    actual: guessGPUAnalysisMeasurementLevel(entry.column)?.level
  }));
  const misses = results.filter(result => result.actual !== result.entry.expected);

  it('has at least sixty realistic columns across seven sources', () => {
    expect(ACCURACY_TABLE.length).toBeGreaterThanOrEqual(60);
    expect(new Set(ACCURACY_TABLE.map(entry => entry.dataset)).size).toBeGreaterThanOrEqual(7);
  });

  it('reports every miss as a documented known limit', () => {
    const report = misses.map(
      miss => `${miss.key}: expected ${String(miss.entry.expected)}, got ${String(miss.actual)}`
    );
    // eslint-disable-next-line no-console
    console.info(
      `measurement-level accuracy: ${results.length - misses.length}/${results.length}\n${report.join('\n')}`
    );
    const undocumented = misses.filter(miss => KNOWN_LIMITS[miss.key] === undefined);
    expect(undocumented.map(miss => miss.key)).toEqual([]);
    for (const miss of misses) {
      expect(KNOWN_LIMITS[miss.key]?.actual, miss.key).toBe(miss.actual);
    }
    const stale = Object.keys(KNOWN_LIMITS).filter(key => !misses.some(miss => miss.key === key));
    expect(stale).toEqual([]);
  });

  it('is at least 85 percent accurate', () => {
    expect((results.length - misses.length) / results.length).toBeGreaterThanOrEqual(0.85);
  });
});

describe('suggestGPUAnalysisRatePairs', () => {
  const cases = makeColumn('cases', 'integer', {min: 0, max: 900, mean: 40, distinct: 300});
  const population = makeColumn('population', 'integer', {
    min: 120,
    max: 90000,
    mean: 6000,
    distinct: 900
  });

  it('pairs cases with a larger population', () => {
    const pairs = suggestGPUAnalysisRatePairs([cases, population]);
    expect(pairs).toHaveLength(1);
    expect(pairs[0]).toMatchObject({numerator: 'cases', denominator: 'population'});
    expect(pairs[0].reason).toContain('cases per population');
  });

  it('keeps a valid event rate when the denominator is smaller', () => {
    const smaller = makeColumn('population', 'integer', {
      min: 0,
      max: 800,
      mean: 30,
      distinct: 200
    });
    expect(suggestGPUAnalysisRatePairs([cases, smaller])).toHaveLength(1);
  });

  it('uses maximum size as evidence rather than excluding repeated-event rates', () => {
    const lowMaximum = makeColumn('population', 'integer', {
      min: 0,
      max: 500,
      mean: 100,
      distinct: 200
    });
    expect(suggestGPUAnalysisRatePairs([cases, lowMaximum])).toHaveLength(1);
  });

  it('falls back to the maximum when no mean exists and skips columns without statistics', () => {
    const noMeanCases = makeColumn('deaths', 'integer', {min: 0, max: 50, distinct: 20});
    const noMeanPopulation = makeColumn('pop', 'integer', {min: 0, max: 5000, distinct: 400});
    expect(suggestGPUAnalysisRatePairs([noMeanCases, noMeanPopulation])).toHaveLength(1);
    const bare = makeColumn('pop', 'integer', {});
    expect(suggestGPUAnalysisRatePairs([noMeanCases, bare])).toEqual([]);
  });

  it('needs both columns to read as counts', () => {
    const rate = makeColumn('cases_per_100k', 'number', {
      min: 0,
      max: 900,
      mean: 40,
      integral: false
    });
    expect(suggestGPUAnalysisRatePairs([rate, population])).toEqual([]);
    const declaredAmount: GPUAnalysisColumnProfile = {...population, measurement: 'amount'};
    expect(suggestGPUAnalysisRatePairs([cases, declaredAmount])).toEqual([]);
    const declaredCount: GPUAnalysisColumnProfile = {
      ...makeColumn('population', 'number', {min: 120, max: 90000, mean: 6000}),
      measurement: 'count'
    };
    expect(suggestGPUAnalysisRatePairs([cases, declaredCount])).toHaveLength(1);
  });

  it('uses names, not substrings: a column called suspected_cases pairs, one called case_manager does not', () => {
    const suspected = makeColumn('suspected_cases', 'integer', {
      min: 0,
      max: 100,
      mean: 4,
      distinct: 90
    });
    expect(suggestGPUAnalysisRatePairs([suspected, population])).toHaveLength(1);
    const manager = makeColumn('case_manager', 'integer', {
      min: 0,
      max: 100,
      mean: 4,
      distinct: 90
    });
    expect(suggestGPUAnalysisRatePairs([manager, population])).toEqual([]);
  });

  it('is empty for empty input and never pairs a column with itself', () => {
    expect(suggestGPUAnalysisRatePairs([])).toEqual([]);
    expect(suggestGPUAnalysisRatePairs([population])).toEqual([]);
  });

  it('returns several pairs in column order', () => {
    const deaths = makeColumn('deaths', 'integer', {min: 0, max: 90, mean: 4, distinct: 60});
    const households = makeColumn('households', 'integer', {
      min: 10,
      max: 30000,
      mean: 2000,
      distinct: 600
    });
    const pairs = suggestGPUAnalysisRatePairs([cases, deaths, population, households]);
    expect(pairs.map(pair => `${pair.numerator}/${pair.denominator}`)).toEqual([
      'cases/population',
      'deaths/population',
      'cases/households',
      'deaths/households'
    ]);
    expect(pairs.map(pair => pair.score)).toEqual([7, 7, 6, 6]);
  });
});

describe('guessGPUAnalysisCoordinateModel', () => {
  it('reads an ordinary extent inside the degree box as geographic with high confidence', () => {
    expect(guessGPUAnalysisCoordinateModel([-122.5, 37.6, -122.3, 37.9])).toMatchObject({
      model: 'geographic',
      confidence: 'high'
    });
  });

  it('reads the whole world as geographic and the box edges as inside', () => {
    expect(guessGPUAnalysisCoordinateModel([-180, -90, 180, 90])).toMatchObject({
      model: 'geographic',
      confidence: 'high'
    });
  });

  it('reads anything outside the box as planar with high confidence', () => {
    expect(guessGPUAnalysisCoordinateModel([500000, 4100000, 600000, 4200000])).toMatchObject({
      model: 'planar',
      confidence: 'high'
    });
    expect(guessGPUAnalysisCoordinateModel([-180.0001, 0, 10, 10]).model).toBe('planar');
    expect(guessGPUAnalysisCoordinateModel([0, -90.0001, 10, 10]).model).toBe('planar');
    expect(guessGPUAnalysisCoordinateModel([-20037508, -20037508, 20037508, 20037508]).model).toBe(
      'planar'
    );
  });

  it('is low confidence for a tiny extent near the origin and for an extent anchored at 0, 0', () => {
    expect(guessGPUAnalysisCoordinateModel([-0.2, -0.1, 0.4, 0.3])).toMatchObject({
      model: 'geographic',
      confidence: 'low'
    });
    expect(guessGPUAnalysisCoordinateModel([0, 0, 120, 80])).toMatchObject({
      model: 'geographic',
      confidence: 'low'
    });
    expect(guessGPUAnalysisCoordinateModel([0, 0, 120, 80]).reason).toContain('confirm');
  });

  it('is not low confidence for a tiny extent far from the origin', () => {
    expect(guessGPUAnalysisCoordinateModel([10.001, 50.001, 10.002, 50.002]).confidence).toBe(
      'high'
    );
  });

  it('reads longitudes 0 to 360 as geographic with low confidence', () => {
    expect(guessGPUAnalysisCoordinateModel([140, -40, 290, 30])).toMatchObject({
      model: 'geographic',
      confidence: 'low'
    });
  });

  it('handles a degenerate single-point extent', () => {
    expect(guessGPUAnalysisCoordinateModel([5, 5, 5, 5])).toMatchObject({
      model: 'geographic',
      confidence: 'high'
    });
  });

  it('throws for non-finite extents', () => {
    expect(() => guessGPUAnalysisCoordinateModel([Number.NaN, 0, 1, 1])).toThrow(
      /guessGPUAnalysisCoordinateModel/
    );
    expect(() => guessGPUAnalysisCoordinateModel([0, 0, Number.POSITIVE_INFINITY, 1])).toThrow(
      /finite/
    );
  });
});

describe('matchGPUAnalysisFieldRole', () => {
  const rateRole: GPUAnalysisFieldRoleSpec = {
    role: 'value',
    label: 'Value',
    accepts: ['number', 'integer'],
    measurement: ['rate', 'amount'],
    required: true
  };
  const columns: readonly GPUAnalysisColumnProfile[] = [
    makeColumn('tract_id', 'integer', {rows: 500, min: 1, max: 500, distinct: 500}),
    makeColumn('cases', 'integer', {rows: 500, min: 0, max: 900, distinct: 300}),
    makeColumn('pct_poverty', 'number', {
      rows: 500,
      min: 0,
      max: 60,
      integral: false,
      distinct: 400
    }),
    makeColumn('county', 'category', {rows: 500, distinct: 30}),
    makeColumn('revenue', 'number', {rows: 500, min: 0, max: 9000, integral: false, distinct: 400}),
    makeColumn('mystery', 'number', {rows: 500, min: -5, max: 5, integral: false, distinct: 400}),
    makeColumn('created', 'time', {rows: 500, min: 1, max: 2})
  ];

  it('orders preferred before accepted, drops wrong types and hides identifiers', () => {
    const matches = matchGPUAnalysisFieldRole(rateRole, columns);
    expect(matches.map(match => `${match.name}:${match.fit}`)).toEqual([
      'pct_poverty:preferred',
      'revenue:preferred',
      'mystery:accepted',
      'cases:accepted'
    ]);
  });

  it('warns with the column level and what the method is meant for', () => {
    const matches = matchGPUAnalysisFieldRole(rateRole, columns);
    expect(matches.find(match => match.name === 'cases')?.warning).toBe(
      'cases is a count; this method is meant for rates or amounts'
    );
    expect(matches.find(match => match.name === 'mystery')?.warning).toBe(
      'mystery has no known measurement level; this method is meant for rates or amounts'
    );
    expect(matches.find(match => match.name === 'pct_poverty')?.warning).toBeUndefined();
  });

  it('notes when a level is only a low-confidence guess', () => {
    const matches = matchGPUAnalysisFieldRole({...rateRole, measurement: ['rate']}, [
      makeColumn('thing', 'integer', {min: 0, max: 400, distinct: 100})
    ]);
    expect(matches[0].warning).toBe(
      'thing is a count (a guess from the data); this method is meant for rates'
    );
  });

  it('prefers a declared measurement over a guess', () => {
    const declared: GPUAnalysisColumnProfile = {
      ...makeColumn('cases', 'integer', {min: 0, max: 900, distinct: 300}),
      measurement: 'rate',
      measurementSource: 'declared'
    };
    const matches = matchGPUAnalysisFieldRole({...rateRole, measurement: ['rate']}, [declared]);
    expect(matches).toEqual([
      {
        name: 'cases',
        fit: 'preferred',
        measurementSource: 'declared',
        measurementConfidence: 'high'
      }
    ]);
  });

  it('offers identifiers only when nothing else exists, with a warning', () => {
    const onlyIdentifiers = [
      columns[0],
      makeColumn('zip', 'integer', {min: 90001, max: 96000, distinct: 400})
    ];
    const matches = matchGPUAnalysisFieldRole(rateRole, onlyIdentifiers);
    expect(matches.map(match => match.name)).toEqual(['tract_id', 'zip']);
    expect(
      matches.every(match => match.fit === 'accepted' && match.warning?.includes('identifier'))
    ).toBe(true);
  });

  it('offers identifiers freely when the role lists the identifier level', () => {
    const matches = matchGPUAnalysisFieldRole({...rateRole, measurement: ['identifier']}, columns);
    expect(matches[0]).toMatchObject({name: 'tract_id', fit: 'preferred'});
  });

  it('accepts every column of an accepted type when the role lists no measurement', () => {
    const categoryRole: GPUAnalysisFieldRoleSpec = {
      role: 'category',
      label: 'Group',
      accepts: ['category', 'boolean'],
      required: false
    };
    const matches = matchGPUAnalysisFieldRole(categoryRole, [
      ...columns,
      makeColumn('flag', 'boolean')
    ]);
    expect(matches.map(match => match.name)).toEqual(['county', 'flag']);
    expect(matches.every(match => match.fit === 'preferred')).toBe(true);
  });

  it('does not hide an identifier-like category column from a category role', () => {
    const categoryRole: GPUAnalysisFieldRoleSpec = {
      role: 'category',
      label: 'Group',
      accepts: ['category'],
      required: false
    };
    const matches = matchGPUAnalysisFieldRole(categoryRole, [
      makeColumn('geoid', 'category', {distinct: 500, rows: 500})
    ]);
    expect(matches).toMatchObject([{name: 'geoid', fit: 'preferred'}]);
  });

  it('treats integer columns as numbers and integral number columns as integers', () => {
    const integerRole: GPUAnalysisFieldRoleSpec = {
      role: 'count',
      label: 'Count',
      accepts: ['integer'],
      required: true
    };
    const matches = matchGPUAnalysisFieldRole(integerRole, [
      makeColumn('a', 'number', {integral: true, min: 0, max: 50, distinct: 30}),
      makeColumn('b', 'number', {integral: false, min: 0, max: 50}),
      makeColumn('c', 'number', {min: 0, max: 50})
    ]);
    expect(matches.map(match => match.name)).toEqual(['a']);
  });

  it('honours excluded column names and empty inputs', () => {
    expect(
      matchGPUAnalysisFieldRole(rateRole, columns, {
        excludedColumnNames: ['pct_poverty', 'revenue']
      }).map(match => match.name)
    ).toEqual(['mystery', 'cases']);
    expect(matchGPUAnalysisFieldRole(rateRole, [])).toEqual([]);
    expect(
      matchGPUAnalysisFieldRole(rateRole, columns, {
        excludedColumnNames: columns.map(column => column.name)
      })
    ).toEqual([]);
  });

  it('ranks a column whose name contains the role id first within its group', () => {
    const populationRole: GPUAnalysisFieldRoleSpec = {
      role: 'population',
      label: 'Population',
      accepts: ['integer'],
      measurement: ['count'],
      required: true
    };
    const matches = matchGPUAnalysisFieldRole(populationRole, [
      makeColumn('cases', 'integer', {min: 0, max: 900, distinct: 300}),
      makeColumn('total_population', 'integer', {min: 0, max: 90000, distinct: 300})
    ]);
    expect(matches.map(match => match.name)).toEqual(['total_population', 'cases']);
  });

  it('orders candidates by the position of their type in role.accepts', () => {
    const timeRole: GPUAnalysisFieldRoleSpec = {
      role: 'time',
      label: 'Time',
      accepts: ['time', 'integer', 'number'],
      required: true
    };
    const mixed = [
      makeColumn('severity', 'integer', {min: 1, max: 3, distinct: 3}),
      makeColumn('weight', 'number', {min: 0.5, max: 3.5, integral: false}),
      makeColumn('timestamp', 'time', {min: 1, max: 2})
    ];
    expect(matchGPUAnalysisFieldRole(timeRole, mixed).map(match => match.name)).toEqual([
      'timestamp',
      'severity',
      'weight'
    ]);
    expect(assignGPUAnalysisFieldRoles([timeRole], mixed).assignment).toEqual({time: 'timestamp'});
    const reversed: GPUAnalysisFieldRoleSpec = {
      ...timeRole,
      accepts: ['number', 'integer', 'time']
    };
    expect(matchGPUAnalysisFieldRole(reversed, mixed).map(match => match.name)).toEqual([
      'weight',
      'severity',
      'timestamp'
    ]);
  });

  it('orders by accepts within the accepted tier as well', () => {
    const role: GPUAnalysisFieldRoleSpec = {
      role: 'value',
      label: 'Value',
      accepts: ['number', 'integer'],
      measurement: ['rate'],
      required: true
    };
    const matches = matchGPUAnalysisFieldRole(role, [
      makeColumn('cases', 'integer', {min: 0, max: 900, distinct: 300}),
      makeColumn('revenue', 'number', {min: 0, max: 9000, integral: false})
    ]);
    expect(matches.map(match => match.name)).toEqual(['revenue', 'cases']);
  });

  it('is deterministic', () => {
    expect(matchGPUAnalysisFieldRole(rateRole, columns)).toEqual(
      matchGPUAnalysisFieldRole(rateRole, columns)
    );
  });
});

describe('assignGPUAnalysisFieldRoles', () => {
  const casesRole: GPUAnalysisFieldRoleSpec = {
    role: 'cases',
    label: 'Cases',
    accepts: ['integer'],
    measurement: ['count'],
    required: true
  };
  const populationRole: GPUAnalysisFieldRoleSpec = {
    role: 'population',
    label: 'Population',
    accepts: ['integer'],
    measurement: ['count'],
    required: true
  };
  const columns: readonly GPUAnalysisColumnProfile[] = [
    makeColumn('id', 'integer', {rows: 500, min: 1, max: 500, distinct: 500}),
    makeColumn('pop', 'integer', {rows: 500, min: 10, max: 90000, distinct: 400}),
    makeColumn('cases', 'integer', {rows: 500, min: 0, max: 900, distinct: 300}),
    makeColumn('pct', 'number', {rows: 500, min: 0, max: 70, integral: false, distinct: 400})
  ];

  it('assigns distinct columns by name affinity and role order', () => {
    const result = assignGPUAnalysisFieldRoles([casesRole, populationRole], columns);
    expect(result.assignment).toEqual({cases: 'cases', population: 'pop'});
    expect(result.unfilledRoles).toEqual([]);
    expect(result.warnings).toEqual([]);
  });

  it('lets a scarce role take the only column it can use before a flexible role', () => {
    const rateRole: GPUAnalysisFieldRoleSpec = {
      role: 'rate',
      label: 'Rate',
      accepts: ['number'],
      measurement: ['rate'],
      required: true
    };
    const anyNumber: GPUAnalysisFieldRoleSpec = {
      role: 'value',
      label: 'Value',
      accepts: ['number', 'integer'],
      required: true
    };
    const result = assignGPUAnalysisFieldRoles(
      [anyNumber, rateRole],
      [
        makeColumn('pct', 'number', {rows: 500, min: 0, max: 70, integral: false, distinct: 400}),
        makeColumn('cases', 'integer', {rows: 500, min: 0, max: 900, distinct: 300})
      ]
    );
    expect(result.assignment).toEqual({value: 'cases', rate: 'pct'});
  });

  it('reports unfilled required roles and warnings for accepted fits', () => {
    const result = assignGPUAnalysisFieldRoles(
      [casesRole, populationRole, {role: 'time', label: 'Time', accepts: ['time'], required: true}],
      [
        makeColumn('cases', 'integer', {rows: 500, min: 0, max: 900, distinct: 300}),
        makeColumn('pct', 'number', {rows: 500, min: 0, max: 70, integral: false, distinct: 400})
      ]
    );
    expect(result.assignment).toEqual({cases: 'cases'});
    expect(result.unfilledRoles).toEqual(['population', 'time']);
  });

  it('warns when a required role falls back to an accepted column', () => {
    const rateRole: GPUAnalysisFieldRoleSpec = {
      role: 'value',
      label: 'Value',
      accepts: ['number', 'integer'],
      measurement: ['rate'],
      required: true
    };
    const result = assignGPUAnalysisFieldRoles(
      [rateRole],
      [makeColumn('cases', 'integer', {rows: 500, min: 0, max: 900, distinct: 300})]
    );
    expect(result.assignment).toEqual({value: 'cases'});
    expect(result.warnings).toEqual(['Value: cases is a count; this method is meant for rates']);
  });

  it('honours conditional requirement and leaves optional roles to preferred fits only', () => {
    const weights: GPUAnalysisFieldRoleSpec = {
      role: 'weights',
      label: 'Weights',
      accepts: ['number'],
      measurement: ['amount'],
      required: {whenParameter: 'weighted', equals: [true]}
    };
    const only = [
      makeColumn('pct', 'number', {rows: 500, min: 0, max: 70, integral: false, distinct: 400})
    ];
    expect(assignGPUAnalysisFieldRoles([weights], only).assignment).toEqual({});
    expect(assignGPUAnalysisFieldRoles([weights], only).unfilledRoles).toEqual([]);
    expect(assignGPUAnalysisFieldRoles([weights], only, {weighted: false}).unfilledRoles).toEqual(
      []
    );
    const required = assignGPUAnalysisFieldRoles([weights], only, {weighted: true});
    expect(required.assignment).toEqual({weights: 'pct'});
    expect(required.warnings).toHaveLength(1);
    const unmet = assignGPUAnalysisFieldRoles([weights], [], {weighted: true});
    expect(unmet.unfilledRoles).toEqual(['weights']);
  });

  it('fills an optional role from a preferred column', () => {
    const optional: GPUAnalysisFieldRoleSpec = {
      role: 'weight',
      label: 'Weight',
      accepts: ['number'],
      measurement: ['rate'],
      required: false
    };
    const result = assignGPUAnalysisFieldRoles(
      [optional],
      [makeColumn('pct', 'number', {min: 0, max: 70, integral: false, distinct: 400})]
    );
    expect(result.assignment).toEqual({weight: 'pct'});
  });

  it('assigns the minimum number of columns to a multiple role and none when too few exist', () => {
    const predictors: GPUAnalysisFieldRoleSpec = {
      role: 'predictors',
      label: 'Predictors',
      accepts: ['number', 'integer'],
      measurement: ['rate', 'amount', 'count'],
      required: true,
      multiple: {minimum: 2, maximum: 5}
    };
    const result = assignGPUAnalysisFieldRoles([predictors], columns);
    expect(result.assignment['predictors']).toEqual(['pct', 'pop']);
    const tooFew = assignGPUAnalysisFieldRoles([predictors], [columns[3]]);
    expect(tooFew.assignment).toEqual({});
    expect(tooFew.unfilledRoles).toEqual(['predictors']);
  });

  it('is deterministic and handles no roles and no columns', () => {
    expect(assignGPUAnalysisFieldRoles([casesRole, populationRole], columns)).toEqual(
      assignGPUAnalysisFieldRoles([casesRole, populationRole], columns)
    );
    expect(assignGPUAnalysisFieldRoles([], columns)).toEqual({
      assignment: {},
      unfilledRoles: [],
      unconfirmedRoles: [],
      warnings: []
    });
    expect(assignGPUAnalysisFieldRoles([casesRole], []).unfilledRoles).toEqual(['cases']);
  });
});

describe('measurement rules added after review', () => {
  const level = (name: string, type: GPUAnalysisColumnType, options: StatisticsOptions = {}) =>
    guessGPUAnalysisMeasurementLevel(makeColumn(name, type, {rows: 3000, ...options}));

  it('reads registry and administrative unit numbers as identifiers (rule: last word)', () => {
    for (const name of [
      'mmsi',
      'apn',
      'parcel_no',
      'ParcelNo',
      'PUMA',
      'ZCTA5CE10',
      'census_tract',
      'STATEFP',
      'COUNTYFP',
      'order_number',
      'invoice_no',
      'account_num'
    ]) {
      expect(level(name, 'integer', {min: 1, max: 999999, distinct: 800})?.level, name).toBe(
        'identifier'
      );
    }
  });

  it('does not read a registry word that is not the head of the name as an identifier', () => {
    expect(level('parcel_value', 'integer', {min: 1, max: 999999, distinct: 800})?.level).not.toBe(
      'identifier'
    );
    expect(level('tract_pop', 'integer', {min: 1, max: 9999, distinct: 800})?.level).toBe('count');
    expect(level('block_area', 'number', {min: 1, max: 9999, integral: false})?.level).toBe(
      'amount'
    );
  });

  it('keeps a plain number a count and a number after a non-registry word a count', () => {
    expect(level('number', 'integer', {min: 0, max: 99, distinct: 50})?.level).toBe('count');
    expect(level('number_of_trips', 'integer', {min: 0, max: 99, distinct: 50})?.level).toBe(
      'count'
    );
  });

  it('reads a code with an ordinal word as ordinal and a plain code as identifier', () => {
    expect(level('severity_code', 'integer', {min: 1, max: 3, distinct: 3})?.level).toBe('ordinal');
    expect(level('grade_code', 'integer', {min: 1, max: 12, distinct: 12})?.level).toBe('ordinal');
    expect(level('land_use_code', 'integer', {min: 100, max: 999, distinct: 40})?.level).toBe(
      'identifier'
    );
  });

  it('reads a unit suffix as an amount, not a rate, even for small values', () => {
    for (const name of [
      'dbh_cm',
      'catch_kg',
      'depth_km',
      'width_m',
      'SHAPE_Leng',
      'trunk_mm',
      'run_ft'
    ]) {
      const guess = level(name, 'number', {min: 0.1, max: 0.9, integral: false, distinct: 900});
      expect(guess?.level, name).toBe('amount');
    }
    expect(level('speed_km', 'number', {min: 0, max: 90, integral: false})?.level).toBe('rate');
  });

  it('reads a time stored as a number as no level, by name and by range', () => {
    expect(
      level('timestamp', 'integer', {
        min: 1_600_000_000_000,
        max: 1_700_000_000_000,
        distinct: 3000
      })
    ).toBeUndefined();
    expect(
      level('unix_time', 'integer', {min: 1_600_000_000, max: 1_700_000_000, distinct: 3000})
    ).toBeUndefined();
    expect(
      level('date_int', 'integer', {min: 20100101, max: 20241231, distinct: 2500})
    ).toBeUndefined();
    expect(level('sample_date', 'integer', {min: 1, max: 99999, distinct: 90})).toBeUndefined();
    // A duration is not a time: a time word without epoch values is read by the other rules.
    expect(level('travel_time', 'integer', {min: 0, max: 3600, distinct: 900})?.level).toBe(
      'count'
    );
  });

  it('reads age and magnitude as no level but median_age as a rate', () => {
    expect(level('age', 'integer', {min: 0, max: 99, distinct: 90})).toBeUndefined();
    expect(level('magnitude', 'number', {min: 2, max: 7, integral: false})).toBeUndefined();
    expect(level('median_age', 'number', {min: 18, max: 70, integral: false})?.level).toBe('rate');
    expect(level('age_group', 'integer', {min: 1, max: 9, distinct: 9})?.level).not.toBe('rate');
  });

  it('reads income as an amount with low confidence and revenue with high', () => {
    expect(level('income', 'number', {min: 8000, max: 190000, integral: false})).toMatchObject({
      level: 'amount',
      confidence: 'low'
    });
    expect(level('revenue', 'number', {min: 0, max: 190000, integral: false})).toMatchObject({
      level: 'amount',
      confidence: 'high'
    });
  });

  it('reads countable nouns as counts with high confidence', () => {
    for (const name of ['bedrooms', 'employees', 'floors', 'beds', 'enrollment']) {
      expect(level(name, 'integer', {min: 0, max: 8, distinct: 8}), name).toMatchObject({
        level: 'count',
        confidence: 'high'
      });
    }
  });

  it('leaves generic names without a level', () => {
    expect(level('value', 'integer', {min: 0, max: 900, distinct: 500})).toBeUndefined();
    expect(level('data', 'number', {min: 0, max: 1, integral: false})).toBeUndefined();
    expect(level('assessed_value', 'integer', {min: 0, max: 9e6, distinct: 500})?.level).toBe(
      'amount'
    );
  });

  it('gives every value-only rule low confidence', () => {
    const guesses = [
      level('thing', 'integer', {min: 1, max: 3000, distinct: 3000}),
      level('thing', 'integer', {min: 0, max: 1, distinct: 2}),
      level('thing', 'integer', {min: 0, max: 5, distinct: 6}),
      level('thing', 'integer', {min: 0, max: 900, distinct: 500}),
      level('thing', 'number', {min: 0, max: 1, integral: false}),
      level('thing', 'number', {min: 0, max: 90, integral: false}),
      level('thing', 'number', {min: 0, max: 9000, integral: false}),
      level('thing', 'text', {distinct: 3000}),
      level('thing', 'text', {distinct: 30})
    ];
    expect(guesses.every(guess => guess?.confidence === 'low')).toBe(true);
  });
});

describe('low-confidence measurement guesses in roles', () => {
  const rateRole: GPUAnalysisFieldRoleSpec = {
    role: 'value',
    label: 'Value',
    accepts: ['number', 'integer'],
    measurement: ['count'],
    required: true
  };
  const profileColumn = (
    overrides: Partial<GPUAnalysisColumnProfile>
  ): GPUAnalysisColumnProfile => ({
    ...makeColumn('mmsi', 'integer', {min: 2e8, max: 7.8e8, distinct: 150}),
    measurement: 'count',
    measurementSource: 'guess',
    ...overrides
  });

  it('puts a profiler-made low-confidence guess in the accepted tier with a guess warning', () => {
    const [match] = matchGPUAnalysisFieldRole(rateRole, [profileColumn({})]);
    expect(match).toMatchObject({
      name: 'mmsi',
      fit: 'accepted',
      measurementSource: 'guess',
      measurementConfidence: 'low'
    });
    expect(match.warning).toBe(
      'mmsi looks like a count (a guess from the data); confirm what it measures before using it'
    );
  });

  it('reads an absent measurementConfidence as low for a guess and certain for a declaration', () => {
    expect(matchGPUAnalysisFieldRole(rateRole, [profileColumn({})])[0].fit).toBe('accepted');
    const declared = profileColumn({measurementSource: 'declared'});
    expect(matchGPUAnalysisFieldRole(rateRole, [declared])[0]).toMatchObject({
      fit: 'preferred',
      measurementConfidence: 'high'
    });
    const unmarked = profileColumn({measurementSource: undefined});
    expect(matchGPUAnalysisFieldRole(rateRole, [unmarked])[0].fit).toBe('preferred');
  });

  it('keeps a high-confidence guess preferred and ranks it before a low-confidence one', () => {
    const matches = matchGPUAnalysisFieldRole(rateRole, [
      profileColumn({name: 'low_one'}),
      profileColumn({name: 'cases', measurementConfidence: 'high'})
    ]);
    expect(matches.map(match => `${match.name}:${match.fit}`)).toEqual([
      'cases:preferred',
      'low_one:accepted'
    ]);
  });

  it('keeps a low-confidence guess of a level the role does not list as an ordinary mismatch', () => {
    const [match] = matchGPUAnalysisFieldRole({...rateRole, measurement: ['rate']}, [
      profileColumn({})
    ]);
    expect(match.warning).toBe(
      'mmsi is a count (a guess from the data); this method is meant for rates'
    );
  });

  it('does not mark a role without measurement levels, and shows an identifier guess instead of hiding it', () => {
    const anyNumber: GPUAnalysisFieldRoleSpec = {
      role: 'value',
      label: 'Value',
      accepts: ['number', 'integer'],
      required: true
    };
    expect(matchGPUAnalysisFieldRole(anyNumber, [profileColumn({})])[0].fit).toBe('preferred');
    const matches = matchGPUAnalysisFieldRole(rateRole, [
      profileColumn({name: 'seq', measurement: 'identifier'}),
      profileColumn({name: 'real', measurementSource: 'declared'})
    ]);
    expect(matches.map(match => `${match.name}:${match.fit}`)).toEqual([
      'real:preferred',
      'seq:accepted'
    ]);
    expect(matches[1].warning).toContain('may be an identifier (a guess from the data)');
  });

  it('assign reports roles filled by a low-confidence guess and does not fill optional roles with one', () => {
    const optional: GPUAnalysisFieldRoleSpec = {
      ...rateRole,
      role: 'weight',
      label: 'Weight',
      required: false
    };
    const result = assignGPUAnalysisFieldRoles(
      [rateRole, optional],
      [profileColumn({}), profileColumn({name: 'other'})]
    );
    expect(result.assignment).toEqual({value: 'mmsi'});
    expect(result.unconfirmedRoles).toEqual(['value']);
    expect(result.warnings[0]).toContain('a guess from the data');
    const confirmed = assignGPUAnalysisFieldRoles(
      [rateRole],
      [profileColumn({measurementConfidence: 'high'})]
    );
    expect(confirmed.unconfirmedRoles).toEqual([]);
  });
});

describe('name affinity with compound names', () => {
  const role: GPUAnalysisFieldRoleSpec = {
    role: 'one-way',
    label: 'One-way flag',
    accepts: ['boolean', 'category'],
    required: false
  };
  const target = (name: string) =>
    matchGPUAnalysisFieldRole(role, [
      makeColumn('highway', 'category', {distinct: 9}),
      makeColumn('way_id', 'category', {distinct: 90}),
      makeColumn(name, 'category', {distinct: 2})
    ])[0].name;

  it('matches oneway, one_way, OneWay, ONEWAY and is_oneway ahead of highway and way_id', () => {
    for (const name of ['oneway', 'one_way', 'OneWay', 'ONEWAY', 'is_oneway', 'one-way']) {
      expect(target(name), name).toBe(name);
    }
  });

  it('does not match highway or way_id', () => {
    const matches = matchGPUAnalysisFieldRole(role, [
      makeColumn('way_id', 'category', {distinct: 90}),
      makeColumn('highway', 'category', {distinct: 9})
    ]);
    expect(matches.map(match => match.name)).toEqual(['way_id', 'highway']);
    // Neither is promoted: order is column order, with no affinity ahead of the other.
  });

  it('assign takes the one-way flag for the role', () => {
    const result = assignGPUAnalysisFieldRoles(
      [role],
      [
        makeColumn('highway', 'category', {distinct: 9}),
        makeColumn('oneway', 'category', {distinct: 2})
      ]
    );
    expect(result.assignment).toEqual({'one-way': 'oneway'});
  });
});

describe('rate pair ranking', () => {
  const column = (
    name: string,
    max: number,
    mean: number,
    extra: Partial<GPUAnalysisColumnProfile> = {}
  ) => makeColumn(name, 'integer', {min: 0, max, mean, distinct: 300, ...extra});

  it('puts the plausible orientation first when residents, incidents and households exist', () => {
    const columns = [
      column('residents', 90000, 6000),
      column('incidents', 900, 40),
      column('households', 30000, 2200)
    ];
    const pairs = suggestGPUAnalysisRatePairs(columns);
    expect(pairs[0]).toMatchObject({numerator: 'incidents', denominator: 'residents'});
    expect(pairs.map(pair => `${pair.numerator}/${pair.denominator}`)).toEqual([
      'incidents/residents',
      'incidents/households'
    ]);
    expect(pairs[0].score).toBeGreaterThan(pairs[1].score);
    expect(assignGPUAnalysisRatePair(columns)).toEqual(pairs[0]);
  });

  it('never pairs a population with a rate, an identifier or a low-confidence column', () => {
    const population = column('population', 90000, 6000);
    const rate = makeColumn('rate_per_100k', 'number', {
      min: 0,
      max: 900,
      mean: 40,
      integral: false
    });
    expect(suggestGPUAnalysisRatePairs([population, rate])).toEqual([]);
    const lowCases: GPUAnalysisColumnProfile = {
      ...column('cases', 900, 40),
      measurement: 'count',
      measurementSource: 'guess'
    };
    expect(suggestGPUAnalysisRatePairs([lowCases, population])).toEqual([]);
    const identifierCases: GPUAnalysisColumnProfile = {
      ...column('cases', 900, 40),
      measurement: 'identifier',
      measurementSource: 'declared'
    };
    expect(suggestGPUAnalysisRatePairs([identifierCases, population])).toEqual([]);
    expect(assignGPUAnalysisRatePair([population, rate])).toBeUndefined();
  });

  it('ranks a plainer denominator and a larger ratio higher, and breaks ties by column order', () => {
    const columns = [
      column('cases', 900, 40),
      column('total', 90000, 6000),
      column('population', 90000, 6000)
    ];
    const pairs = suggestGPUAnalysisRatePairs(columns);
    expect(pairs.map(pair => pair.denominator)).toEqual(['population', 'total']);
    const tied = suggestGPUAnalysisRatePairs([
      column('cases', 900, 40),
      column('pop', 90000, 6000),
      column('population', 90000, 6000)
    ]);
    expect(tied.map(pair => pair.denominator)).toEqual(['pop', 'population']);
  });

  it('rewards a shared year suffix and retains repeated-event rates above one', () => {
    const pairs = suggestGPUAnalysisRatePairs([
      column('cases_2020', 900, 40),
      column('cases_2021', 900, 40),
      column('pop_2021', 90000, 6000)
    ]);
    expect(pairs[0]).toMatchObject({numerator: 'cases_2021', denominator: 'pop_2021'});
    const equal = suggestGPUAnalysisRatePairs([
      column('cases', 900, 40),
      column('population', 900, 40)
    ]);
    expect(equal).toHaveLength(1);
    const smaller = suggestGPUAnalysisRatePairs([
      column('cases', 900, 40),
      column('population', 900, 39)
    ]);
    expect(smaller).toHaveLength(1);
    expect(smaller[0]).toMatchObject({numerator: 'cases', denominator: 'population', score: 6});
  });
});

// ---------------------------------------------------------------------------------------------
// Row semantics
// ---------------------------------------------------------------------------------------------

type SemanticsRow = {
  dataset: string;
  kind: GPUAnalysisDataProfile['kind'];
  columns: readonly GPUAnalysisColumnProfile[];
  /** Share of points at an already used location. */
  repeated?: number;
  /** What a careful analyst says the rows are, decided before the rules were run. */
  expected: GPUAnalysisRowSemantics | undefined;
};

function makeSemanticsProfile(entry: SemanticsRow): GPUAnalysisDataProfile {
  const rows = 3000;
  const profile: GPUAnalysisDataProfile = {
    kind: entry.kind,
    rowCount: rows,
    columns: entry.columns
  };
  if (entry.repeated === undefined) {
    return profile;
  }
  return {
    ...profile,
    spacing: {
      pointCount: rows,
      sampleSize: rows,
      sampled: false,
      coordinateModel: 'geographic',
      unit: 'metre',
      extent: [-100, 30, -99, 31],
      extentWidth: 1000,
      extentHeight: 1000,
      extentDiagonal: 1414,
      center: [-99.5, 30.5],
      standardDistance: 300,
      medianDistanceFromCenter: 280,
      nearestNeighbor: {
        minimum: 0,
        lowerQuartile: 1,
        median: 2,
        upperQuartile: 3,
        mean: 2,
        maximum: 9,
        upperFence: 6,
        aboveUpperFenceCount: 1
      },
      duplicateLocationCount: Math.round(rows * entry.repeated)
    }
  };
}

const R = 3000;
const idColumn = makeColumn('id', 'integer', {rows: R, min: 1, max: R, distinct: R});
const timeColumn = (name = 'date') => makeColumn(name, 'time', {rows: R, min: 1, max: 2});
const category = (name: string) => makeColumn(name, 'category', {rows: R, distinct: 12});
const numberColumn = (name: string, min = 0, max = 100) =>
  makeColumn(name, 'number', {rows: R, min, max, integral: false, distinct: 2000});
const integerColumn = (name: string, min = 0, max = 100) =>
  makeColumn(name, 'integer', {rows: R, min, max, distinct: 80});
const latitude = numberColumn('latitude', 30, 31);
const longitude = numberColumn('longitude', -100, -99);

const SEMANTICS_TABLE: readonly SemanticsRow[] = [
  {
    dataset: 'crime incidents',
    kind: 'points',
    columns: [
      idColumn,
      category('primary_type'),
      timeColumn(),
      makeColumn('arrest', 'boolean', {rows: R}),
      latitude,
      longitude
    ],
    expected: 'events'
  },
  {
    dataset: 'traffic crashes',
    kind: 'points',
    columns: [
      makeColumn('crash_id', 'integer', {rows: R, min: 1, max: R, distinct: R}),
      timeColumn('crash_date'),
      integerColumn('severity', 1, 3),
      integerColumn('injuries', 0, 9),
      category('weather')
    ],
    expected: 'events'
  },
  {
    dataset: '311 requests',
    kind: 'points',
    columns: [
      makeColumn('unique_key', 'text', {rows: R, distinct: R}),
      timeColumn('created'),
      category('complaint_type'),
      category('status')
    ],
    expected: 'events'
  },
  {
    dataset: 'geocoded incidents, repeated addresses',
    kind: 'points',
    columns: [idColumn, category('category')],
    repeated: 0.4,
    expected: 'events'
  },
  {
    dataset: 'bird sightings',
    kind: 'points',
    columns: [
      category('species'),
      integerColumn('individual_count', 1, 90),
      timeColumn('observed')
    ],
    expected: 'events'
  },
  {
    dataset: 'accident locations, ids only',
    kind: 'points',
    columns: [idColumn, latitude, longitude],
    expected: 'events'
  },
  {
    dataset: 'weather snapshot',
    kind: 'points',
    columns: [
      makeColumn('station_id', 'text', {rows: R, distinct: R}),
      numberColumn('temperature_c', -20, 40),
      numberColumn('humidity', 10, 100),
      numberColumn('pressure_hpa', 950, 1050)
    ],
    repeated: 0,
    expected: 'surface-samples'
  },
  {
    dataset: 'air quality sensors',
    kind: 'points',
    columns: [idColumn, numberColumn('pm25', 1, 90), numberColumn('no2', 1, 80)],
    repeated: 0.01,
    expected: 'surface-samples'
  },
  {
    dataset: 'rain gauges',
    kind: 'points',
    columns: [
      makeColumn('gauge_id', 'text', {rows: R, distinct: R}),
      numberColumn('rainfall_mm', 0, 300)
    ],
    repeated: 0,
    expected: 'surface-samples'
  },
  {
    dataset: 'soil samples',
    kind: 'points',
    columns: [
      idColumn,
      numberColumn('ph', 4, 9),
      numberColumn('organic_matter', 0, 12),
      numberColumn('depth_cm', 0, 100)
    ],
    repeated: 0.02,
    expected: 'surface-samples'
  },
  {
    dataset: 'elevation survey',
    kind: 'points',
    columns: [idColumn, numberColumn('elevation_m', -5, 3000)],
    repeated: 0,
    expected: 'surface-samples'
  },
  {
    dataset: 'groundwater wells',
    kind: 'points',
    columns: [
      makeColumn('well_id', 'text', {rows: R, distinct: R}),
      numberColumn('water_level_m', 0, 90),
      numberColumn('depth_m', 5, 300)
    ],
    repeated: 0,
    expected: 'surface-samples'
  },
  {
    dataset: 'noise measurements',
    kind: 'points',
    columns: [idColumn, numberColumn('noise_db', 30, 110)],
    repeated: 0.03,
    expected: 'surface-samples'
  },
  {
    dataset: 'shops',
    kind: 'points',
    columns: [
      idColumn,
      makeColumn('name', 'text', {rows: R, distinct: R}),
      numberColumn('revenue', 0, 9e6),
      integerColumn('employees', 1, 90),
      category('category')
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'hospitals',
    kind: 'points',
    columns: [
      idColumn,
      makeColumn('name', 'text', {rows: R, distinct: R}),
      integerColumn('beds', 5, 1200),
      integerColumn('capacity', 5, 1500)
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'schools',
    kind: 'points',
    columns: [
      makeColumn('name', 'text', {rows: R, distinct: R}),
      integerColumn('enrollment', 20, 3000),
      numberColumn('rating', 1, 5)
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'parcels as points',
    kind: 'points',
    columns: [
      makeColumn('apn', 'integer', {rows: R, min: 1e8, max: 9e8, distinct: R}),
      makeColumn('assessed_value', 'integer', {rows: R, min: 1e4, max: 9e7, distinct: 2900}),
      numberColumn('lot_area_sqft', 500, 2e6),
      category('zoning')
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'restaurants',
    kind: 'points',
    columns: [
      makeColumn('name', 'text', {rows: R, distinct: R}),
      integerColumn('price', 1, 4),
      category('cuisine')
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'bike share stations',
    kind: 'points',
    columns: [idColumn, integerColumn('capacity', 4, 80), integerColumn('bikes_available', 0, 60)],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'street trees',
    kind: 'points',
    columns: [
      makeColumn('tree_id', 'integer', {rows: R, min: 1, max: R, distinct: R}),
      category('species'),
      numberColumn('dbh_cm', 1, 90)
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'census tracts',
    kind: 'polygons',
    columns: [
      makeColumn('GEOID', 'text', {rows: R, distinct: R}),
      integerColumn('total_pop', 0, 9000)
    ],
    expected: 'objects'
  },
  {
    dataset: 'grid cells',
    kind: 'cells',
    columns: [integerColumn('count', 0, 90)],
    expected: 'objects'
  },
  {
    dataset: 'counties',
    kind: 'regions',
    columns: [makeColumn('name', 'text', {rows: R, distinct: R})],
    expected: 'objects'
  },
  {
    dataset: 'earthquakes',
    kind: 'points',
    columns: [
      makeColumn('id', 'text', {rows: R, distinct: R}),
      timeColumn('time'),
      numberColumn('mag', 1, 9),
      numberColumn('depth', 0, 700)
    ],
    repeated: 0,
    expected: 'events'
  },
  {
    dataset: 'wildfires',
    kind: 'points',
    columns: [
      idColumn,
      timeColumn('discovered'),
      category('cause'),
      numberColumn('acres_burned', 0, 9e5)
    ],
    repeated: 0,
    expected: 'events'
  },
  {
    dataset: 'vessel pings',
    kind: 'points',
    columns: [
      makeColumn('mmsi', 'integer', {rows: R, min: 2e8, max: 7.8e8, distinct: 150}),
      timeColumn('timestamp'),
      numberColumn('sog', 0, 25),
      numberColumn('cog', 0, 359)
    ],
    repeated: 0.01,
    expected: 'events'
  },
  {
    dataset: 'taxi pickups',
    kind: 'points',
    columns: [
      timeColumn('pickup_time'),
      integerColumn('passenger_count', 1, 6),
      numberColumn('fare_amount', 3, 200)
    ],
    repeated: 0.1,
    expected: 'events'
  },
  {
    dataset: 'weather stations over time',
    kind: 'points',
    columns: [
      makeColumn('station', 'text', {rows: R, distinct: 40}),
      timeColumn('date'),
      numberColumn('temperature_c', -20, 40)
    ],
    repeated: 0.98,
    expected: 'surface-samples'
  },
  {
    dataset: 'police stations',
    kind: 'points',
    columns: [
      idColumn,
      makeColumn('name', 'text', {rows: R, distinct: R}),
      integerColumn('precinct', 1, 77)
    ],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'points of interest',
    kind: 'points',
    columns: [makeColumn('name', 'text', {rows: R, distinct: R}), category('category')],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'cell towers',
    kind: 'points',
    columns: [idColumn, category('radio'), integerColumn('samples', 1, 900)],
    repeated: 0,
    expected: 'objects'
  },
  {
    dataset: 'lightning strikes',
    kind: 'points',
    columns: [timeColumn('time'), numberColumn('amplitude', 1, 200)],
    repeated: 0,
    expected: 'events'
  },
  {dataset: 'lines of roads', kind: 'lines', columns: [category('highway')], expected: undefined},
  {
    dataset: 'sample and object attributes together',
    kind: 'points',
    columns: [idColumn, numberColumn('temperature_c', -20, 40), integerColumn('employees', 1, 90)],
    repeated: 0,
    expected: undefined
  },
  {
    dataset: 'numeric attributes with no evidence',
    kind: 'points',
    columns: [idColumn, numberColumn('score', 0, 100)],
    repeated: 0,
    expected: undefined
  }
];

/** Misses of the row semantics guess against the hand labels: why each one happens. */
const SEMANTICS_LIMITS: Readonly<
  Record<string, {actual: GPUAnalysisRowSemantics | undefined; why: string}>
> = {
  earthquakes: {
    actual: 'surface-samples',
    why: 'Depth reads as a measured quantity and nothing in the columns says the rows are occurrences; only the user knows an earthquake is an event.'
  },
  wildfires: {
    actual: undefined,
    why: 'An amount (acres burned) with a time column is both an object attribute and an occurrence stamp, which conflict, so no guess is made.'
  },
  'vessel pings': {
    actual: undefined,
    why: 'A speed and a course with a time stamp and no categories carry no evidence either way.'
  },
  'taxi pickups': {
    actual: undefined,
    why: 'A fare amount with a time column conflicts, as for wildfires.'
  },
  'weather stations over time': {
    actual: undefined,
    why: 'A measured quantity at heavily repeated locations conflicts (repeated points read as events, the quantity as a surface), so no guess is made.'
  },
  'cell towers': {
    actual: undefined,
    why: 'A generic count column (samples) and a category give no sign of a thing with attributes.'
  },
  'lightning strikes': {
    actual: undefined,
    why: 'A generic value column (amplitude) with a time stamp and no categories carries no evidence either way.'
  }
};

describe('guessGPUAnalysisRowSemantics', () => {
  const results = SEMANTICS_TABLE.map(entry => ({
    entry,
    actual: guessGPUAnalysisRowSemantics(makeSemanticsProfile(entry))?.value
  }));
  const misses = results.filter(result => result.actual !== result.entry.expected);

  it('has at least thirty labelled datasets', () => {
    expect(SEMANTICS_TABLE.length).toBeGreaterThanOrEqual(30);
  });

  it('reports every miss as a documented limit', () => {
    const report = misses.map(
      miss =>
        `${miss.entry.dataset}: expected ${String(miss.entry.expected)}, got ${String(miss.actual)}`
    );
    // eslint-disable-next-line no-console
    console.info(
      `row-semantics accuracy: ${results.length - misses.length}/${results.length}\n${report.join('\n')}`
    );
    expect(
      misses
        .filter(miss => SEMANTICS_LIMITS[miss.entry.dataset] === undefined)
        .map(miss => miss.entry.dataset)
    ).toEqual([]);
    for (const miss of misses) {
      expect(SEMANTICS_LIMITS[miss.entry.dataset]?.actual, miss.entry.dataset).toBe(miss.actual);
    }
    expect(
      Object.keys(SEMANTICS_LIMITS).filter(key => !misses.some(miss => miss.entry.dataset === key))
    ).toEqual([]);
  });

  it('is always low confidence and explains itself as a question to confirm', () => {
    for (const {entry} of results) {
      const guess = guessGPUAnalysisRowSemantics(makeSemanticsProfile(entry));
      if (guess) {
        expect(guess.confidence).toBe('low');
        expect(guess.reason).toContain('confirm it');
      }
    }
  });

  it('uses the full point count for the share of repeated locations', () => {
    const profile = makeSemanticsProfile({
      dataset: 'x',
      kind: 'points',
      columns: [idColumn, category('category')],
      repeated: 0,
      expected: 'events'
    });
    const spacing = profile.spacing;
    expect(spacing).toBeDefined();
    if (spacing) {
      const measuredProfile: GPUAnalysisDataProfile = {
        ...profile,
        columns: [idColumn, numberColumn('temperature_c', -20, 40)],
        spacing: {
          ...spacing,
          sampled: true,
          sampleSize: 100,
          duplicateLocationCount: spacing.pointCount * 0.05
        }
      };
      // The full-data count is at the 5 percent limit, regardless of the spacing sample.
      expect(guessGPUAnalysisRowSemantics(measuredProfile)).toBeUndefined();
      const fewer = {
        ...measuredProfile,
        spacing: {
          ...spacing,
          sampled: true,
          sampleSize: 100,
          duplicateLocationCount: spacing.pointCount * 0.04
        }
      };
      expect(guessGPUAnalysisRowSemantics(fewer)?.value).toBe('surface-samples');
    }
  });
});

describe('reviewed measurement and role edge cases', () => {
  it('finds the complete cost/source/target matching from the review probe', () => {
    const columns = [
      makeColumn('from', 'integer', {rows: 3000, min: 10, max: 99, distinct: 90}),
      makeColumn('travel_minutes', 'number', {
        rows: 3000,
        min: 0.2,
        max: 42.8,
        distinct: 900,
        integral: false
      }),
      makeColumn('to', 'text', {rows: 3000, distinct: 90})
    ];
    const roles: GPUAnalysisFieldRoleSpec[] = [
      {role: 'cost', label: 'Travel cost', required: true, accepts: ['number']},
      {role: 'source', label: 'Source node', required: true, accepts: ['integer', 'text']},
      {role: 'target', label: 'Target node', required: true, accepts: ['integer', 'text']}
    ];
    const result = assignGPUAnalysisFieldRoles(roles, columns);
    expect(result.assignment).toEqual({cost: 'travel_minutes', source: 'from', target: 'to'});
    expect(result.unfilledRoles).toEqual([]);
    expect(result.warnings).toEqual([]);
    expect(assignGPUAnalysisFieldRoles(roles, columns)).toEqual(result);
  });

  it('matches multi-column required roles before optional roles without reserving incomplete roles', () => {
    const roles: GPUAnalysisFieldRoleSpec[] = [
      {
        role: 'values',
        label: 'Values',
        accepts: ['number'],
        required: true,
        multiple: {minimum: 2, maximum: 3}
      },
      {role: 'source', label: 'Source', accepts: ['integer'], required: true},
      {role: 'extra', label: 'Extra', accepts: ['number'], required: false}
    ];
    const columns = [
      makeColumn('from', 'integer'),
      makeColumn('travel_minutes', 'number'),
      makeColumn('distance', 'number')
    ];
    expect(assignGPUAnalysisFieldRoles(roles, columns)).toMatchObject({
      assignment: {values: ['distance', 'travel_minutes'], source: 'from'},
      unfilledRoles: []
    });
    const insufficient = assignGPUAnalysisFieldRoles(roles, [columns[0]]);
    expect(insufficient.assignment).toEqual({source: 'from'});
    expect(insufficient.unfilledRoles).toEqual(['values']);
  });

  it.each([
    'UnitPrice',
    'unit_price',
    'price_per_item',
    'price_per_kg',
    'unit_cost'
  ])('reads %s as an intensive unit price', name => {
    const column = makeColumn(name, 'number', {
      rows: 3000,
      min: 0.1,
      max: 150,
      distinct: 1000,
      integral: false
    });
    expect(guessGPUAnalysisMeasurementLevel(column)).toMatchObject({
      level: 'rate',
      confidence: 'high'
    });
    const amountRole: GPUAnalysisFieldRoleSpec = {
      role: 'value',
      label: 'Amount to total',
      accepts: ['number'],
      measurement: ['amount'],
      required: true
    };
    expect(matchGPUAnalysisFieldRole(amountRole, [column])[0]).toMatchObject({fit: 'accepted'});
    expect(assignGPUAnalysisFieldRoles([amountRole], [column]).warnings).toHaveLength(1);
  });

  it('keeps unqualified price ambiguous', () => {
    const column = makeColumn('price', 'number', {min: 1, max: 150, integral: false});
    expect(guessGPUAnalysisMeasurementLevel(column)).toMatchObject({
      level: 'amount',
      confidence: 'low'
    });
  });

  it('suggests visits per resident above one and ranks magnitude without fabricating larger denominators', () => {
    const visits = makeColumn('visits', 'integer', {
      rows: 20,
      min: 120,
      max: 220,
      mean: 160,
      distinct: 15
    });
    const residents = makeColumn('residents', 'integer', {
      rows: 20,
      min: 50,
      max: 100,
      mean: 80,
      distinct: 15
    });
    const suggestions = suggestGPUAnalysisRatePairs([visits, residents]);
    expect(suggestions).toHaveLength(1);
    expect(suggestions[0]).toMatchObject({numerator: 'visits', denominator: 'residents', score: 4});
    expect(suggestions[0].reason).not.toContain('is larger');
    const proportion = {...visits, name: 'share_visits', measurement: 'count' as const};
    expect(suggestGPUAnalysisRatePairs([proportion, residents])).toEqual([]);
  });

  it('keeps absent and empty spacing unknown in the surface-sample explanation', () => {
    const profile: GPUAnalysisDataProfile = {
      kind: 'points',
      rowCount: 3000,
      columns: [makeColumn('temperature', 'number', {rows: 3000, min: 5, max: 30})]
    };
    const guess = guessGPUAnalysisRowSemantics(profile);
    expect(guess).toMatchObject({value: 'surface-samples', confidence: 'low'});
    expect(guess?.reason).toContain('repeated locations have not been checked');
    expect(guess?.reason).not.toContain('few points share a location');
    const spaced = makeSemanticsProfile({
      dataset: 'empty spacing',
      kind: 'points',
      columns: profile.columns,
      repeated: 0,
      expected: 'surface-samples'
    });
    if (!spaced.spacing) throw new Error('reviewed spacing test: spaced.spacing must be present');
    const emptySpacing = guessGPUAnalysisRowSemantics({
      ...spaced,
      spacing: {...spaced.spacing, sampleSize: 0}
    });
    expect(emptySpacing?.reason).toContain('repeated locations have not been checked');
    expect(guessGPUAnalysisRowSemantics(spaced)?.reason).toContain('few points share a location');
  });

  it('never reads prototype properties as cyclic units', () => {
    for (const name of ['constructor', '__proto__', 'hasOwnProperty', 'toString']) {
      const guess = guessGPUAnalysisMeasurementLevel(
        makeColumn(name, 'number', {min: 0.1, max: 150, integral: false})
      );
      expect(guess?.level).not.toBe('cyclic');
      expect(guess?.reason).not.toContain('[native code]');
    }
  });

  it('does not fabricate Boolean values in empty or entirely null columns', () => {
    for (const rows of [0, 3000]) {
      expect(
        guessGPUAnalysisMeasurementLevel(makeColumn('flag', 'boolean', {rows, nulls: rows}))
      ).toBeUndefined();
    }
    expect(
      guessGPUAnalysisMeasurementLevel(makeColumn('flag', 'boolean', {rows: 3000, nulls: 2999}))
    ).toMatchObject({level: 'nominal', confidence: 'high'});
  });
});

it('keeps sampled and unsampled row guesses equal and caps repeated shares at 100 percent', () => {
  const profile = makeSemanticsProfile({
    dataset: 'sampled counts',
    kind: 'points',
    columns: [numberColumn('quantity', 100, 200)],
    repeated: 0,
    expected: undefined
  });
  const spacing = profile.spacing;
  if (!spacing) throw new Error('row semantics test: profile.spacing must be present');
  for (const duplicateLocationCount of [10, 300, 1000, 2000]) {
    const full = {
      ...profile,
      rowCount: 1000,
      spacing: {...spacing, pointCount: 1000, sampleSize: 1000, duplicateLocationCount}
    };
    const sampled = {...full, spacing: {...full.spacing, sampleSize: 5, sampled: true}};
    const guess = guessGPUAnalysisRowSemantics(sampled);
    expect(guess).toEqual(guessGPUAnalysisRowSemantics(full));
    if (duplicateLocationCount === 10) expect(guess).toBeUndefined();
    else expect(guess?.reason).toContain(`${Math.min(100, duplicateLocationCount / 10)} percent`);
  }
  expect(
    guessGPUAnalysisRowSemantics({
      ...profile,
      spacing: {...spacing, pointCount: 0, duplicateLocationCount: 10}
    })
  ).toBeUndefined();
});
