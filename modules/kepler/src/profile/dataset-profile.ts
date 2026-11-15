// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeGPUAnalysisSpacingStatistics,
  guessGPUAnalysisCoordinateModel,
  guessGPUAnalysisMeasurementLevel,
  guessGPUAnalysisRowSemantics,
  tokenizeGPUAnalysisColumnName,
  type GPUAnalysisColumnProfile,
  type GPUAnalysisColumnStatistics,
  type GPUAnalysisColumnType,
  type GPUAnalysisCoordinateModel,
  type GPUAnalysisRowSemantics,
  type GPUAnalysisSpacingStatistics,
  type GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogColumnType,
  CatalogCoordinateSystem,
  CatalogFeature,
  CatalogFeatureCollection,
  CatalogGeometry,
  CatalogOutputTable,
  CatalogPosition
} from '../analysis/catalog-geojson';
import {
  countFeatureVertices,
  getLineParts,
  getPointPositions,
  getPolygonParts
} from '../analysis/catalog-geojson';
import {getFeatureAnchor} from '../analysis/catalog-platform3-b-common';
import type {LocalFrame} from '../analysis/catalog-local-frame';
import type {KeplerDatasetProfile, KeplerDatasetProfileDeclarations} from './dataset-profile-types';

// Collects what the host knows about a dataset the moment it is loaded: its shape, coordinates,
// one statistics record per column, the guessed or declared meaning of each column and how the
// locations are spread. The profile is computed once per dataset revision (see
// `dataset-profile-cache.ts`) and feeds the recommender and the default rules. Everything here is
// a pure CPU function with a deterministic result; nothing is read from a clock or a random source.

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------

/** How a column is stored, in the four types the catalog and the recommender know. */
export type KeplerProfileStorageType = 'string' | 'float64' | 'int32' | 'boolean';

/**
 * A profiled column. `storage` records how the values are held, which a time column needs: the
 * analysis column type `time` does not say whether the values are ISO text or epoch numbers.
 */
export type KeplerDatasetProfileColumn = GPUAnalysisColumnProfile & {
  storage: KeplerProfileStorageType;
};

/** Geometry families of the rows of a dataset. */
export type KeplerGeometryFeatureCounts = {
  point: number;
  line: number;
  polygon: number;
  /** Rows with no geometry, or a geometry the catalog cannot read (for example a collection). */
  withoutGeometry: number;
};

/**
 * What the profile functions return: the contract type plus the storage of each column, the
 * geometry mix and plain-language notes (rejected coordinate pairs, sampling, mixed geometry).
 */
export type KeplerProfiledDataset = Omit<KeplerDatasetProfile, 'columns'> & {
  /**
   * Bounds of points and geometry edges, `[west, south, east, north]`.
   * For a planar model it is `[minimumX, minimumY, maximumX, maximumY]` in source coordinates. For
   * a geographic model the longitude range covers points and the literal longitude intervals of
   * geometry edges (RFC 7946 sections 3.1.1 and 3.1.9). For data across 180 degrees the first
   * value exceeds the third (a box from 178 to 182 is `[178, south, -178, north]`). Measure it with
   * `getGPUAnalysisExtentSizes(extent, model)`, never by subtracting. It is recomputed when a
   * declaration changes the coordinate model.
   */
  extent?: readonly [number, number, number, number];
  columns: readonly KeplerDatasetProfileColumn[];
  geometryFeatureCounts?: KeplerGeometryFeatureCounts;
  notes?: readonly string[];
};

export type KeplerDatasetProfileOptions = {
  datasetId: string;
  label: string;
  revision?: number;
  declarations?: KeplerDatasetProfileDeclarations;
  /**
   * Largest number of rows the column statistics and the neighbour distances are computed on, an
   * integer of at least 2. Absent: the column statistics are exact (every row) and the spacing
   * uses the analysis package default. Present and below the row count: both use a deterministic
   * sample and `sampleSize` is recorded: stride for columns, hashed valid-row indices for spacing.
   */
  maximumSampleSize?: number;
  /**
   * Coordinate system of the geometry, when the host knows it. A feature collection carries none
   * (the catalog takes it as a binding parameter), so the host passes it here. `EPSG:4326` is
   * geographic, `planar` is planar. A `coordinateModel` declaration overrides it.
   */
  coordinateSystem?: CatalogCoordinateSystem;
  /**
   * Declared type by column name, for columns whose type the values cannot settle (epoch numbers
   * that are times). A feature collection has no other way to declare a type; in a column table a
   * type given on the column itself wins.
   */
  columnTypes?: Readonly<Record<string, CatalogColumnType | GPUAnalysisColumnType>>;
};

/** One column of a table the host holds as columns. */
export type KeplerProfileColumnInput = {
  name: string;
  /**
   * Declared type: a catalog type (`string`, `float64`, `int32`, `boolean`) or an analysis column
   * type. Absent: the type is inferred from the values. See {@link profileKeplerColumnTable}.
   */
  type?: CatalogColumnType | GPUAnalysisColumnType;
  values: ArrayLike<unknown>;
};

/** A table as a list of columns of equal length. */
export type KeplerProfileColumnTable = {
  rowCount: number;
  columns: readonly KeplerProfileColumnInput[];
  /** Column holding GeoJSON geometry text. */
  geometryColumn?: string;
};

export type KeplerColumnTableProfileOptions = KeplerDatasetProfileOptions & {
  /** Numeric columns that hold the longitude and latitude (or x and y) of point rows. */
  pointColumns?: {longitude: string; latitude: string};
  /** Columns that hold GeoJSON geometry text. Defaults to the table's `geometryColumn`. */
  geometryColumns?: readonly string[];
};

/** The structural shape the recommender takes; defined here so this module needs no other. */
export type KeplerRecommendationSubject = {
  kind: 'point' | 'line' | 'polygon' | 'features' | 'table';
  rowCount: number;
  columns: readonly {name: string; type: 'string' | 'float64' | 'int32' | 'boolean'}[];
};

// ---------------------------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------------------------

/**
 * Distinct values are counted exactly with a set up to this many. A set of 10,000 numbers or short
 * strings costs well under a megabyte, and a column with more distinct values than this is not a
 * category whatever its exact count is.
 */
export const KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT = 10_000;

/**
 * Past the exact limit the distinct count is an estimate by linear counting: each value sets one
 * bit of a bitmap of this many bits (128 KiB), chosen by a 32-bit hash, and the estimate is
 * -m ln(z / m) with m the bit count and z the bits still clear. The standard error is under 0.1
 * percent up to about a million distinct values, and the estimate is clamped to the non-null
 * count. Linear counting is deterministic and one hash per value, which a sketch with several
 * registers would not be at this cost.
 */
const DISTINCT_BITMAP_BIT_COUNT = 1 << 20;

/** A text column with at most this many distinct values, and at most half as many as rows, is a category. */
export const KEPLER_PROFILE_MAXIMUM_CATEGORY_DISTINCT_COUNT = 100;

/** Rows scanned for the minimum sample size; below 2 a stride sample has no meaning. */
const MINIMUM_SAMPLE_SIZE = 2;

/** Latitude and longitude limits in degrees for accepting a column pair as coordinates. */
const LATITUDE_LIMIT_DEGREES = 90;
const LONGITUDE_LIMIT_DEGREES = 180;

const LATITUDE_WORDS: ReadonlySet<string> = new Set(['lat', 'latitude']);
const LONGITUDE_WORDS: ReadonlySet<string> = new Set(['lng', 'lon', 'long', 'longitude']);

/**
 * Column pairs that probably hold planar coordinates. They are never taken as coordinates (`x` and
 * `y` name too many things), but a note names them so that a host can offer them as point columns.
 */
const PLANAR_PAIR_WORDS: readonly (readonly [ReadonlySet<string>, ReadonlySet<string>])[] = [
  [new Set(['x']), new Set(['y'])],
  [new Set(['easting']), new Set(['northing'])]
];

const INT32_MINIMUM = -2147483648;
const INT32_MAXIMUM = 2147483647;

const SPATIAL_KIND_BY_FAMILY = {point: 'points', line: 'lines', polygon: 'polygons'} as const;
const CATALOG_GEOMETRY_BY_FAMILY = {point: 'point', line: 'line', polygon: 'polygon'} as const;

// ---------------------------------------------------------------------------------------------
// Distinct counting
// ---------------------------------------------------------------------------------------------

const hashScratchFloat = new Float64Array(1);
const hashScratchWords = new Uint32Array(hashScratchFloat.buffer);

/** Murmur3 32-bit finaliser: every input bit changes about half of the output bits. */
function mixHash(value: number): number {
  let hash = value >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return hash >>> 0;
}

function hashValue(value: number | string): number {
  if (typeof value === 'number') {
    // `value + 0` turns -0 into 0 so that both count as one value, like a Set does.
    hashScratchFloat[0] = value + 0;
    return mixHash(hashScratchWords[0] ^ mixHash(hashScratchWords[1] + 0x9e3779b9));
  }
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return mixHash(hash);
}

/** Counts distinct numbers or strings: exact with a set, then a linear-counting estimate. */
class DistinctCounter {
  private exactValues: Set<number | string> | null = new Set();
  private bitmap: Uint32Array | null = null;

  add(value: number | string): void {
    if (this.exactValues) {
      this.exactValues.add(value);
      if (this.exactValues.size > KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT) this.spill();
      return;
    }
    this.setBit(hashValue(value));
  }

  isExact(): boolean {
    return this.exactValues !== null;
  }

  /** Distinct values seen: exact, or the linear-counting estimate. */
  getCount(): number {
    if (this.exactValues) return this.exactValues.size;
    const bitmap = this.bitmap as Uint32Array;
    let setBitCount = 0;
    for (let index = 0; index < bitmap.length; index++) {
      let word = bitmap[index];
      while (word) {
        word &= word - 1;
        setBitCount++;
      }
    }
    const clearBitCount = DISTINCT_BITMAP_BIT_COUNT - setBitCount;
    // A full bitmap has no estimate; the clamp to the non-null count in the caller bounds it.
    if (clearBitCount === 0) return DISTINCT_BITMAP_BIT_COUNT * Math.log(DISTINCT_BITMAP_BIT_COUNT);
    return Math.round(
      -DISTINCT_BITMAP_BIT_COUNT * Math.log(clearBitCount / DISTINCT_BITMAP_BIT_COUNT)
    );
  }

  private spill(): void {
    this.bitmap = new Uint32Array(DISTINCT_BITMAP_BIT_COUNT >>> 5);
    for (const value of this.exactValues as Set<number | string>) this.setBit(hashValue(value));
    this.exactValues = null;
  }

  private setBit(hash: number): void {
    const bitIndex = hash & (DISTINCT_BITMAP_BIT_COUNT - 1);
    (this.bitmap as Uint32Array)[bitIndex >>> 5] |= 1 << (bitIndex & 31);
  }
}

// ---------------------------------------------------------------------------------------------
// Column statistics
// ---------------------------------------------------------------------------------------------

/** Returned by {@link parseIsoTimestamp} for text that has the shape of a timestamp but names an impossible time. */
const IMPOSSIBLE_TIMESTAMP = Number.NEGATIVE_INFINITY;

function readDigits(text: string, start: number, count: number): number {
  let value = 0;
  for (let index = start; index < start + count; index++) {
    const code = text.charCodeAt(index);
    if (code < 48 || code > 57) return -1;
    value = value * 10 + (code - 48);
  }
  return value;
}

function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's algorithm); valid for year 0 and up. */
function getDaysFromCivil(year: number, month: number, day: number): number {
  const shiftedYear = month <= 2 ? year - 1 : year;
  const era = Math.floor(shiftedYear / 400);
  const yearOfEra = shiftedYear - era * 400;
  const dayOfYear = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1;
  const dayOfEra =
    yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100) + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}

/**
 * Milliseconds since the epoch of an ISO 8601 date or date-time string (`YYYY-MM-DD`, optionally
 * `THH:MM`, `:SS`, `.fraction` and a `Z` or `+HH:MM` / `+HHMM` zone). NaN when the text does not
 * have that shape. {@link IMPOSSIBLE_TIMESTAMP} when it has the shape but not a real time: a day
 * beyond the month's length (leap years honoured), month outside 1 to 12, hour above 23 (so
 * `24:00:00` is rejected), minute or second above 59, a zone offset beyond 23:59. The engine's
 * `Date.parse` rolls such text over to another date, so the check is done here. A date-time
 * without a zone designator is read as UTC, because `Date.parse` would read it in the zone of the
 * machine and the same data would then profile differently on two machines.
 */
function parseIsoTimestamp(text: string): number {
  // Cheap rejection first: most text is not a timestamp, and this runs on every string.
  const length = text.length;
  if (length < 10 || text.charCodeAt(4) !== 45 || text.charCodeAt(7) !== 45) return Number.NaN;
  const year = readDigits(text, 0, 4);
  const month = readDigits(text, 5, 2);
  const day = readDigits(text, 8, 2);
  if (year < 0 || month < 0 || day < 0) return Number.NaN;
  let hour = 0;
  let minute = 0;
  let second = 0;
  let fractionMilliseconds = 0;
  let offsetMinutes = 0;
  let offsetHour = 0;
  let offsetMinute = 0;
  if (length > 10) {
    if (text.charCodeAt(10) !== 84 || text.charCodeAt(13) !== 58) return Number.NaN;
    hour = readDigits(text, 11, 2);
    minute = readDigits(text, 14, 2);
    if (hour < 0 || minute < 0) return Number.NaN;
    let position = 16;
    if (position < length && text.charCodeAt(position) === 58) {
      second = readDigits(text, position + 1, 2);
      if (second < 0) return Number.NaN;
      position += 3;
      if (position < length && text.charCodeAt(position) === 46) {
        position++;
        const fractionStart = position;
        let scale = 100;
        while (position < length) {
          const code = text.charCodeAt(position);
          if (code < 48 || code > 57) break;
          if (scale >= 1) fractionMilliseconds += (code - 48) * scale;
          scale /= 10;
          position++;
        }
        if (position === fractionStart) return Number.NaN;
        fractionMilliseconds = Math.floor(fractionMilliseconds);
      }
    }
    if (position < length) {
      const code = text.charCodeAt(position);
      if (code === 90) {
        if (position + 1 !== length) return Number.NaN;
      } else if (code === 43 || code === 45) {
        offsetHour = readDigits(text, position + 1, 2);
        const minuteStart = text.charCodeAt(position + 3) === 58 ? position + 4 : position + 3;
        offsetMinute = readDigits(text, minuteStart, 2);
        if (offsetHour < 0 || offsetMinute < 0 || minuteStart + 2 !== length) return Number.NaN;
        offsetMinutes = (offsetHour * 60 + offsetMinute) * (code === 45 ? -1 : 1);
      } else {
        return Number.NaN;
      }
    }
  }
  if (month < 1 || month > 12) return IMPOSSIBLE_TIMESTAMP;
  const daysInMonth = month === 2 && isLeapYear(year) ? 29 : DAYS_IN_MONTH[month - 1];
  if (day < 1 || day > daysInMonth) return IMPOSSIBLE_TIMESTAMP;
  if (hour > 23 || minute > 59 || second > 59 || offsetHour > 23 || offsetMinute > 59)
    return IMPOSSIBLE_TIMESTAMP;
  return (
    getDaysFromCivil(year, month, day) * 86_400_000 +
    ((hour * 60 + minute) * 60 + second) * 1000 +
    fractionMilliseconds -
    offsetMinutes * 60_000
  );
}

const powerScratchFloat = new Float64Array(1);
const powerScratchWords = new Uint32Array(powerScratchFloat.buffer);

/** Exact 2^exponent for any exponent; 0 below the smallest subnormal and Infinity above 1023. */
function getPowerOfTwo(exponent: number): number {
  if (exponent > 1023) return Number.POSITIVE_INFINITY;
  if (exponent < -1074) return 0;
  if (exponent >= -1022) {
    powerScratchWords[0] = 0;
    powerScratchWords[1] = (exponent + 1023) << 20;
  } else {
    // A subnormal power of two is one set bit of the 52 bit mantissa.
    const bit = exponent + 1074;
    powerScratchWords[0] = bit < 32 ? 1 << bit : 0;
    powerScratchWords[1] = bit >= 32 ? 1 << (bit - 32) : 0;
  }
  return powerScratchFloat[0];
}

/** Binary exponent e of a positive finite number, so that 2^e <= value < 2^(e + 1); -1022 for subnormals. */
function getBinaryExponent(value: number): number {
  powerScratchFloat[0] = value;
  const biased = (powerScratchWords[1] >>> 20) & 0x7ff;
  return biased === 0 ? -1022 : biased - 1023;
}

/** Integers up to 2^64 are the range of 64 bit identifiers and counters, which a double cannot always hold exactly. */
const UNSAFE_INTEGER_LIMIT = 2 ** 64;
const MAXIMUM_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

type ResolvedColumnType = {
  type: GPUAnalysisColumnType;
  /** True when a text column keeps every kind of value (a mixed column), false for strings only. */
  keepsAllValues: boolean;
};

/**
 * One pass accumulator of one column. It records every class of value it meets (finite numbers,
 * strings, booleans, dates) so the type can be settled at the end, and allocates nothing per row.
 */
class ColumnAccumulator {
  numberCount = 0;
  nonFiniteCount = 0;
  stringCount = 0;
  isoStringCount = 0;
  dateCount = 0;
  booleanCount = 0;
  private trueSeen = false;
  private falseSeen = false;
  /** Mean and sum of squared deviations of the values divided by 2^scaleExponent; see addNumber. */
  private mean = 0;
  private sumOfSquares = 0;
  private scaleExponent = 0;
  private inverseScale = 1;
  /** Values at or above this magnitude need a larger scale; starts at the smallest positive double. */
  private scaleLimit = Number.MIN_VALUE;
  /** True when an integer beyond 2^53 (a bigint, or a double up to 2^64) was seen: digits may be lost. */
  hasUnsafeIntegers = false;
  /** Strings with the shape of a timestamp that name an impossible time (`2021-02-29`). */
  impossibleTimestampCount = 0;
  private bigIntegerDistinct: DistinctCounter | null = null;
  private minimum = Number.POSITIVE_INFINITY;
  private maximum = Number.NEGATIVE_INFINITY;
  private allIntegral = true;
  private allSafe = true;
  private allInt32 = true;
  private timeMinimum = Number.POSITIVE_INFINITY;
  private timeMaximum = Number.NEGATIVE_INFINITY;
  private numberDistinct: DistinctCounter | null = null;
  private stringDistinct: DistinctCounter | null = null;
  private dateDistinct: DistinctCounter | null = null;
  private timestampStringDistinct: DistinctCounter | null = null;

  add(value: unknown): void {
    switch (typeof value) {
      case 'number':
        this.addNumber(value);
        return;
      case 'string':
        this.addString(value);
        return;
      case 'boolean':
        this.booleanCount++;
        if (value) this.trueSeen = true;
        else this.falseSeen = true;
        return;
      case 'bigint':
        if (!Number.isFinite(Number(value))) {
          this.nonFiniteCount++;
          return;
        }
        if (value > MAXIMUM_SAFE_BIGINT || value < -MAXIMUM_SAFE_BIGINT) {
          // Beyond 2^53 the double used for the statistics is rounded, so the values are counted
          // as distinct by their exact text; the column is typed number and its guess withheld.
          this.hasUnsafeIntegers = true;
          this.bigIntegerDistinct = this.bigIntegerDistinct ?? new DistinctCounter();
          this.bigIntegerDistinct.add(value.toString());
          this.addNumber(Number(value), false);
        } else {
          this.addNumber(Number(value));
        }
        return;
      case 'object':
        if (value instanceof Date) this.addDate(value.getTime());
        // Other objects and arrays have no scalar meaning and count as null.
        return;
      default:
        return;
    }
  }

  private addNumber(value: number, countDistinct = true): void {
    if (!Number.isFinite(value)) {
      this.nonFiniteCount++;
      return;
    }
    const magnitude = Math.abs(value);
    if (magnitude >= this.scaleLimit) this.growScale(magnitude);
    this.numberCount++;
    // Welford's recurrence (one pass, no catastrophic cancellation) on the value divided by a
    // power of two that is at most its largest magnitude so far. Dividing by a power of two is
    // exact, and it keeps the squared deviations near 1 whatever the magnitude: unscaled, values
    // around 1e160 overflow to an infinite spread and values around 1e-200 underflow to zero.
    const scaled = value * this.inverseScale;
    const delta = scaled - this.mean;
    this.mean += delta / this.numberCount;
    this.sumOfSquares += delta * (scaled - this.mean);
    if (value < this.minimum) this.minimum = value;
    if (value > this.maximum) this.maximum = value;
    if (this.allIntegral && !Number.isInteger(value)) this.allIntegral = false;
    if (magnitude > Number.MAX_SAFE_INTEGER) {
      this.allSafe = false;
      if (magnitude < UNSAFE_INTEGER_LIMIT && Number.isInteger(value))
        this.hasUnsafeIntegers = true;
    }
    if (this.allInt32 && (value > INT32_MAXIMUM || value < INT32_MINIMUM)) this.allInt32 = false;
    if (countDistinct) {
      this.numberDistinct = this.numberDistinct ?? new DistinctCounter();
      this.numberDistinct.add(value);
    }
  }

  /** Moves the scale up to the binary exponent of `magnitude`, rescaling the running mean and deviations. */
  private growScale(magnitude: number): void {
    const exponent = getBinaryExponent(magnitude);
    if (exponent > this.scaleExponent || this.scaleLimit === Number.MIN_VALUE) {
      if (this.numberCount > 0) {
        const ratio = getPowerOfTwo(this.scaleExponent - exponent);
        this.mean *= ratio;
        this.sumOfSquares *= ratio * ratio;
      }
      this.scaleExponent = exponent;
      this.inverseScale = getPowerOfTwo(-exponent);
    }
    this.scaleLimit = getPowerOfTwo(this.scaleExponent + 1);
  }

  private addString(value: string): void {
    this.stringCount++;
    const timestamp = parseIsoTimestamp(value);
    if (timestamp === IMPOSSIBLE_TIMESTAMP) {
      this.impossibleTimestampCount++;
    } else if (timestamp === timestamp) {
      this.isoStringCount++;
      this.timestampStringDistinct = this.timestampStringDistinct ?? new DistinctCounter();
      this.timestampStringDistinct.add(value);
      if (timestamp < this.timeMinimum) this.timeMinimum = timestamp;
      if (timestamp > this.timeMaximum) this.timeMaximum = timestamp;
    }
    this.stringDistinct = this.stringDistinct ?? new DistinctCounter();
    this.stringDistinct.add(value);
  }

  private addDate(milliseconds: number): void {
    if (Number.isNaN(milliseconds)) return;
    this.dateCount++;
    if (milliseconds < this.timeMinimum) this.timeMinimum = milliseconds;
    if (milliseconds > this.timeMaximum) this.timeMaximum = milliseconds;
    this.dateDistinct = this.dateDistinct ?? new DistinctCounter();
    this.dateDistinct.add(milliseconds);
  }

  /** True when every value is a whole number in the window of epoch seconds or of epoch milliseconds. */
  isEpochLike(): boolean {
    if (this.numberCount === 0 || this.numberCount !== this.getPresentCount() || !this.allIntegral)
      return false;
    return (
      (this.minimum >= EPOCH_SECONDS_WINDOW[0] && this.maximum <= EPOCH_SECONDS_WINDOW[1]) ||
      (this.minimum >= EPOCH_MILLISECONDS_WINDOW[0] && this.maximum <= EPOCH_MILLISECONDS_WINDOW[1])
    );
  }

  /** Values of any class that are not null: finite numbers, strings, booleans and valid dates. */
  private getPresentCount(): number {
    return this.numberCount + this.stringCount + this.booleanCount + this.dateCount;
  }

  private getDistinct(
    type: GPUAnalysisColumnType,
    keepsAllValues: boolean
  ): {count: number; estimate: boolean} | undefined {
    const counters: DistinctCounter[] = [];
    let booleanValues = 0;
    if (type === 'number' || type === 'integer') {
      if (this.numberDistinct) counters.push(this.numberDistinct);
      if (this.bigIntegerDistinct) counters.push(this.bigIntegerDistinct);
    } else if (type === 'boolean') {
      booleanValues = (this.trueSeen ? 1 : 0) + (this.falseSeen ? 1 : 0);
    } else if (type === 'time') {
      if (this.timestampStringDistinct) counters.push(this.timestampStringDistinct);
      if (this.bigIntegerDistinct) counters.push(this.bigIntegerDistinct);
      if (this.dateDistinct) counters.push(this.dateDistinct);
      if (this.numberDistinct) counters.push(this.numberDistinct);
    } else {
      if (this.stringDistinct) counters.push(this.stringDistinct);
      if (keepsAllValues) {
        if (this.bigIntegerDistinct) counters.push(this.bigIntegerDistinct);
        if (this.numberDistinct) counters.push(this.numberDistinct);
        if (this.dateDistinct) counters.push(this.dateDistinct);
        booleanValues = (this.trueSeen ? 1 : 0) + (this.falseSeen ? 1 : 0);
      }
    }
    let count = booleanValues;
    let estimate = false;
    for (const counter of counters) {
      count += counter.getCount();
      if (!counter.isExact()) estimate = true;
    }
    return count === 0 ? undefined : {count, estimate};
  }

  /**
   * Settles the column type. A declared type picks the class of value that counts; any other
   * class is a null. Without a declaration the type follows the values: only finite numbers
   * (`integer` when all are whole and within 2^53, else `number`), only booleans, only timestamps
   * (ISO strings and dates), or only strings (`category` or `text`). A column that mixes classes
   * is `text` and keeps every value, because no single numeric reading is faithful to it; its
   * numbers are not summarised. A column with no value at all is `text`, except one that holds
   * only non-finite numbers, which is `number`.
   */
  private resolveType(declared: KeplerProfileColumnInput['type']): ResolvedColumnType {
    const timestampCount = this.isoStringCount + this.dateCount;
    const present = this.getPresentCount();
    switch (declared) {
      case 'float64':
      case 'number':
        return {type: 'number', keepsAllValues: false};
      case 'int32':
      case 'integer':
        return {type: 'integer', keepsAllValues: false};
      case 'boolean':
        return {type: 'boolean', keepsAllValues: false};
      case 'time':
        return {type: 'time', keepsAllValues: false};
      case 'category':
      case 'text':
        return {type: declared, keepsAllValues: false};
      case 'string':
        return {type: this.inferStringType(), keepsAllValues: false};
      default:
        break;
    }
    if (present === 0) {
      return {type: this.nonFiniteCount > 0 ? 'number' : 'text', keepsAllValues: false};
    }
    if (this.numberCount === present) {
      return {
        type: this.allIntegral && this.allSafe ? 'integer' : 'number',
        keepsAllValues: false
      };
    }
    if (this.booleanCount === present) return {type: 'boolean', keepsAllValues: false};
    if (timestampCount === present && this.impossibleTimestampCount === 0)
      return {type: 'time', keepsAllValues: false};
    if (this.stringCount === present) {
      return {type: this.inferStringType(), keepsAllValues: false};
    }
    return {type: 'text', keepsAllValues: true};
  }

  private inferStringType(): GPUAnalysisColumnType {
    // Text that looks like a date but names an impossible one (`2021-02-29`) is data to look at,
    // not a time column: it is `text`, whatever its distinct count.
    if (this.impossibleTimestampCount > 0) return 'text';
    if (this.stringCount > 0 && this.isoStringCount === this.stringCount && this.dateCount === 0)
      return 'time';
    const distinct = this.stringDistinct ? this.stringDistinct.getCount() : 0;
    if (
      distinct <= KEPLER_PROFILE_MAXIMUM_CATEGORY_DISTINCT_COUNT &&
      distinct * 2 <= this.stringCount
    )
      return 'category';
    return 'text';
  }

  /**
   * The column record. `rowsSeen` is the number of rows fed to the profile (the sample size under
   * sampling), so a row with no value, or with a value of another class than the declared one,
   * counts as null.
   */
  finish(
    name: string,
    rowCount: number,
    rowsSeen: number,
    declared: KeplerProfileColumnInput['type']
  ): KeplerDatasetProfileColumn {
    const {type, keepsAllValues} = this.resolveType(declared);
    let validCount: number;
    if (type === 'number' || type === 'integer') validCount = this.numberCount;
    else if (type === 'boolean') validCount = this.booleanCount;
    // A column declared `time` also takes finite numbers: epoch seconds or milliseconds.
    else if (type === 'time') validCount = this.isoStringCount + this.dateCount + this.numberCount;
    else validCount = keepsAllValues ? this.getPresentCount() : this.stringCount;

    const column: KeplerDatasetProfileColumn = {
      name,
      type,
      rowCount,
      nullCount: rowsSeen - validCount,
      storage: 'string'
    };
    if (rowsSeen < rowCount) column.sampleSize = rowsSeen;

    if ((type === 'number' || type === 'integer') && this.numberCount > 0) {
      column.minimum = this.minimum;
      column.maximum = this.maximum;
      column.mean = this.mean * getPowerOfTwo(this.scaleExponent);
      // Sample standard deviation (n - 1): the values are read as a sample of what the dataset
      // describes, which is what bandwidth rules such as Scott's and Silverman's expect. One value
      // has no spread estimate, so none is reported.
      if (this.numberCount > 1)
        column.standardDeviation =
          Math.sqrt(this.sumOfSquares / (this.numberCount - 1)) * getPowerOfTwo(this.scaleExponent);
      column.integral = this.allIntegral;
    } else if (type === 'time' && validCount > 0) {
      column.minimum =
        this.numberCount > 0 ? Math.min(this.timeMinimum, this.minimum) : this.timeMinimum;
      column.maximum =
        this.numberCount > 0 ? Math.max(this.timeMaximum, this.maximum) : this.timeMaximum;
    }

    const distinct = validCount > 0 ? this.getDistinct(type, keepsAllValues) : undefined;
    if (distinct) {
      column.distinctCount = Math.min(distinct.count, validCount);
      if (distinct.estimate) column.distinctCountIsEstimate = true;
    }

    if (type === 'boolean') column.storage = 'boolean';
    else if (type === 'integer') column.storage = this.allInt32 ? 'int32' : 'float64';
    else if (type === 'number') column.storage = 'float64';
    else if (type === 'time') column.storage = this.stringCount > 0 ? 'string' : 'float64';
    return column;
  }
}

/** Words in a column name that suggest a time. */
const TIME_NAME_WORDS: ReadonlySet<string> = new Set([
  'time',
  'timestamp',
  'datetime',
  'date',
  'epoch',
  'ts',
  'created',
  'updated',
  'modified',
  'when'
]);

/**
 * Epoch seconds from 1971 to 2100 and epoch milliseconds for the same dates. The windows start in
 * 1971, not 1970, so that small whole numbers (counts, durations) are not taken for times.
 */
const EPOCH_SECONDS_WINDOW = [31_536_000, 4_102_444_800] as const;
const EPOCH_MILLISECONDS_WINDOW = [31_536_000_000, 4_102_444_800_000] as const;

/** Writes a column's record and, where something about it needs a host's attention, a note. */
function finishColumn(
  name: string,
  accumulator: ColumnAccumulator,
  rowCount: number,
  rowsSeen: number,
  declared: KeplerProfileColumnInput['type'],
  notes: string[]
): KeplerDatasetProfileColumn {
  const column = accumulator.finish(name, rowCount, rowsSeen, declared);
  if (accumulator.impossibleTimestampCount > 0)
    notes.push(
      `Column "${name}" has ${accumulator.impossibleTimestampCount} values shaped like dates that name impossible dates or times (such as 2021-02-29 or 24:00:00); they are not read as times.`
    );
  if (accumulator.hasUnsafeIntegers) {
    // The statistics are of rounded doubles: a guess built on them would look more certain than it is.
    notes.push(
      `Column "${name}" holds integers beyond 2^53, which a double cannot keep exactly; its statistics are approximate and its measurement is not guessed.`
    );
    return column;
  }
  if (
    declared === undefined &&
    (column.type === 'number' || column.type === 'integer') &&
    accumulator.isEpochLike() &&
    tokenizeGPUAnalysisColumnName(name).some(word => TIME_NAME_WORDS.has(word))
  )
    notes.push(
      `Column "${name}" holds whole numbers in the range of epoch seconds or milliseconds (1971 to 2100) and its name suggests a time; declare its type as time (options.columnTypes or the column's type) if it is.`
    );
  const guess = guessGPUAnalysisMeasurementLevel(column as GPUAnalysisColumnStatistics);
  if (!guess) return column;
  return {
    ...column,
    measurement: guess.level,
    measurementSource: 'guess',
    measurementConfidence: guess.confidence,
    measurementReason: guess.reason
  };
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

type GeometryFamily = 'point' | 'line' | 'polygon';

function getGeometryFamily(geometry: CatalogGeometry | undefined): GeometryFamily | undefined {
  switch (geometry?.type) {
    case 'Point':
    case 'MultiPoint':
      return 'point';
    case 'LineString':
    case 'MultiLineString':
      return 'line';
    case 'Polygon':
    case 'MultiPolygon':
      return 'polygon';
    default:
      return undefined;
  }
}

/** The catalog's planar vertex-mean helper requires an identity frame. */
const IDENTITY_FRAME: LocalFrame = {
  gpuFrame: null,
  shiftX: 0,
  shiftY: 0,
  forward: position => [position[0], position[1]],
  getNonFiniteCount: () => 0,
  inverse: (x, y) => [x, y]
};

/** A weighted position whose weight is kept as weight * 2^exponent until normalization. */
type WeightedAnchor = {weight: number; exponent: number; x: number; y: number};

/** Binary64's largest finite power of two; log2 rounds some finite values up to 1024. */
const MAXIMUM_FINITE_BINARY_EXPONENT = 1023;

/** Splits positive finite factors before multiplying, so an area need not fit in binary64. */
function scaleAnchorWeight(factors: readonly number[]): {weight: number; exponent: number} {
  let weight = 1;
  let exponent = 0;
  for (const factor of factors) {
    const factorExponent = Math.min(Math.floor(Math.log2(factor)), MAXIMUM_FINITE_BINARY_EXPONENT);
    weight *= factor / 2 ** factorExponent;
    exponent += factorExponent;
  }
  return {weight, exponent};
}

/** A translated coordinate scale even when the difference of two finite endpoints overflows. */
function getAnchorOffsetScale(value: number, origin: number): number {
  const difference = Math.abs(value - origin);
  return Number.isFinite(difference) ? difference : Math.max(Math.abs(value), Math.abs(origin));
}

/** Restores a translated coordinate, avoiding an overflowing displacement before cancellation. */
function restoreAnchorCoordinate(origin: number, offset: number, scale: number): number {
  const coordinate = origin + offset * scale;
  return Number.isFinite(coordinate) ? coordinate : (origin / scale + offset) * scale;
}

/** Subtract before scaling when possible, retaining small offsets from large origins. */
function normalizeAnchorOffset(value: number, origin: number, scale: number): number {
  const difference = value - origin;
  return Number.isFinite(difference) ? difference / scale : value / scale - origin / scale;
}

/**
 * Accumulates relative weights and translated, separately scaled coordinates. Neither an
 * unscaled area nor an area times a source coordinate is formed, keeping tiny and huge polygons.
 */
function combineWeightedAnchors(anchors: readonly WeightedAnchor[]): [number, number] | undefined {
  if (anchors.length === 0) return undefined;
  const originX = anchors[0].x;
  const originY = anchors[0].y;
  let maximumExponent = Number.NEGATIVE_INFINITY;
  let horizontalScale = 0;
  let verticalScale = 0;
  for (const anchor of anchors) {
    maximumExponent = Math.max(maximumExponent, anchor.exponent);
    horizontalScale = Math.max(horizontalScale, getAnchorOffsetScale(anchor.x, originX));
    verticalScale = Math.max(verticalScale, getAnchorOffsetScale(anchor.y, originY));
  }
  let totalWeight = 0;
  let horizontalMoment = 0;
  let verticalMoment = 0;
  for (const anchor of anchors) {
    const weight = anchor.weight * 2 ** (anchor.exponent - maximumExponent);
    totalWeight += weight;
    if (horizontalScale > 0)
      horizontalMoment += weight * normalizeAnchorOffset(anchor.x, originX, horizontalScale);
    if (verticalScale > 0)
      verticalMoment += weight * normalizeAnchorOffset(anchor.y, originY, verticalScale);
  }
  if (!(totalWeight > 0)) return undefined;
  return [
    restoreAnchorCoordinate(originX, horizontalMoment / totalWeight, horizontalScale),
    restoreAnchorCoordinate(originY, verticalMoment / totalWeight, verticalScale)
  ];
}

/** A midpoint without overflowing the sum or discarding a small, representable difference. */
function getAnchorMidpoint(first: number, second: number): number {
  const difference = second - first;
  return Number.isFinite(difference) ? first + difference / 2 : first / 2 + second / 2;
}

/** Length-weighted path centroid, unchanged by inserting vertices or cutting an edge into parts. */
function getPathAnchor(
  paths: readonly (readonly CatalogPosition[])[],
  close: boolean
): [number, number] | undefined {
  const edges: WeightedAnchor[] = [];
  const vertices: WeightedAnchor[] = [];
  const addEdge = (first: CatalogPosition, second: CatalogPosition): void => {
    let length = Math.hypot(second[0] - first[0], second[1] - first[1]);
    let exponent = 0;
    if (!Number.isFinite(length)) {
      length = Math.hypot(second[0] / 2 - first[0] / 2, second[1] / 2 - first[1] / 2);
      exponent = 1;
    }
    if (length > 0) {
      const scaled = scaleAnchorWeight([length]);
      edges.push({
        weight: scaled.weight,
        exponent: scaled.exponent + exponent,
        x: getAnchorMidpoint(first[0], second[0]),
        y: getAnchorMidpoint(first[1], second[1])
      });
    }
  };
  for (const path of paths) {
    let first: CatalogPosition | undefined;
    let previous: CatalogPosition | undefined;
    for (const position of path) {
      if (!Number.isFinite(position[0]) || !Number.isFinite(position[1])) continue;
      vertices.push({weight: 1, exponent: 0, x: position[0], y: position[1]});
      first ??= position;
      if (previous) addEdge(previous, position);
      previous = position;
    }
    if (close && previous && first) addEdge(previous, first);
  }
  return combineWeightedAnchors(edges) ?? combineWeightedAnchors(vertices);
}

/** Bounds rounding in coordinate normalization, cross products and accumulation for each ring vertex. */
const RING_AREA_ROUNDING_OPERATIONS_PER_VERTEX = 8;

/** Area and centroid of one ring, orientation-free, computed from the first vertex to keep precision. */
function measureRing(ring: readonly CatalogPosition[]): WeightedAnchor | undefined {
  const vertices = ring.filter(
    position => Number.isFinite(position[0]) && Number.isFinite(position[1])
  );
  if (vertices.length < 3) return undefined;
  const [originX, originY] = vertices[0];
  let horizontalScale = 0;
  let verticalScale = 0;
  for (const position of vertices) {
    horizontalScale = Math.max(horizontalScale, getAnchorOffsetScale(position[0], originX));
    verticalScale = Math.max(verticalScale, getAnchorOffsetScale(position[1], originY));
  }
  if (horizontalScale === 0 || verticalScale === 0)
    return {weight: 0, exponent: 0, x: originX, y: originY};
  let crossSum = 0;
  let absoluteProductSum = 0;
  let momentX = 0;
  let momentY = 0;
  for (let index = 0; index < vertices.length; index++) {
    const next = vertices[(index + 1) % vertices.length];
    const currentX = normalizeAnchorOffset(vertices[index][0], originX, horizontalScale);
    const currentY = normalizeAnchorOffset(vertices[index][1], originY, verticalScale);
    const nextX = normalizeAnchorOffset(next[0], originX, horizontalScale);
    const nextY = normalizeAnchorOffset(next[1], originY, verticalScale);
    const forwardProduct = currentX * nextY;
    const backwardProduct = nextX * currentY;
    const cross = forwardProduct - backwardProduct;
    crossSum += cross;
    absoluteProductSum += Math.abs(forwardProduct) + Math.abs(backwardProduct);
    momentX += (currentX + nextX) * cross;
    momentY += (currentY + nextY) * cross;
  }
  // The error depends on this ring's products, not its aspect ratio or distance to other parts.
  // Separate axis scales preserve thin, accurately computable rings without squared-size tests.
  const roundingSteps = RING_AREA_ROUNDING_OPERATIONS_PER_VERTEX * vertices.length;
  const relativeErrorBound =
    (roundingSteps * Number.EPSILON) / (1 - roundingSteps * Number.EPSILON);
  if (Math.abs(crossSum) <= relativeErrorBound * absoluteProductSum)
    return {weight: 0, exponent: 0, x: originX, y: originY};
  return {
    ...scaleAnchorWeight([Math.abs(crossSum) / 2, horizontalScale, verticalScale]),
    x: restoreAnchorCoordinate(originX, momentX / (3 * crossSum), horizontalScale),
    y: restoreAnchorCoordinate(originY, momentY / (3 * crossSum), verticalScale)
  };
}

/**
 * Representative position of a polygon or multipolygon in source coordinates: the area-weighted
 * centroid over all parts. Each part counts by the absolute area of its shell minus the absolute
 * areas of its holes, so ring orientation does not matter (a clockwise part and an anticlockwise
 * part add up instead of cancelling) and a hole pulls the centroid away from itself. A geometry
 * with no numerically resolved ring area uses the centroid of its shell edges (length-weighted),
 * or the mean of the vertices when all of them coincide. Each ring has its own rounding error bound, so
 * thin polygons and widely separated parts keep their real area. Undefined without a finite vertex.
 */
function getPolygonAnchor(geometry: CatalogGeometry): [number, number] | undefined {
  const anchors: WeightedAnchor[] = [];
  const shells: (readonly CatalogPosition[])[] = [];
  for (const polygon of getPolygonParts(geometry)) {
    if (polygon.length > 0) shells.push(polygon[0]);
    for (let ringIndex = 0; ringIndex < polygon.length; ringIndex++) {
      const measured = measureRing(polygon[ringIndex]);
      if (measured && measured.weight > 0) {
        anchors.push({...measured, weight: (ringIndex === 0 ? 1 : -1) * measured.weight});
      }
    }
  }
  return combineWeightedAnchors(anchors) ?? getPathAnchor(shells, true);
}

type Extent = [number, number, number, number];

/** Longitude buckets of the covering arc: a gap between occupied buckets is exact, one inside a bucket (under 0.1 degree) is not seen. */
const ARC_BUCKETS_PER_DEGREE = 10;
const ARC_BUCKET_COUNT = 360 * ARC_BUCKETS_PER_DEGREE;

/**
 * Collects the extent of positions in one pass and constant memory (two buckets arrays of 3,600
 * numbers): the plain minimum and maximum, and the geographic covering extent.
 *
 * The covering extent follows the convention of the spacing statistics of the analysis package
 * (`computeGPUAnalysisSpacingStatistics`, whose geographic extent helper is not exported; the
 * rule is restated here and the two are tested to agree through `getGPUAnalysisExtentSizes`):
 * `[west, south, east, north]` where the longitude range is the smallest arc that covers every
 * longitude and every geometry edge, 360 degrees minus the largest uncovered gap modulo 360. When
 * that arc is narrower than the plain range, west and east are the ends of the arc in
 * [-180, 180] and west is GREATER than east if the arc crosses the antimeridian (a box from 178 to
 * 182 is `[178, south, -178, north]`); otherwise it equals the plain extent. The arc is found
 * from occupied longitude intervals in each 0.1 degree bucket of the circle, so every gap of
 * 0.1 degree or more is exact; data with no such gap covers the whole circle and gets the plain
 * extent. Longitudes written 0 to 360 are normalised modulo 360 for the arc computation only;
 * the stored coordinates and the plain extent keep the source values. A dataset at 150 to 250 is
 * one arc of 100 degrees and keeps the source range `[150, south, 250, north]` (east above 180,
 * which the sizes measure correctly); a dataset at 350 and 10 becomes `[-10, south, 10, north]`.
 */
class ExtentAccumulator {
  private count = 0;
  private minimumX = Number.POSITIVE_INFINITY;
  private minimumY = Number.POSITIVE_INFINITY;
  private maximumX = Number.NEGATIVE_INFINITY;
  private maximumY = Number.NEGATIVE_INFINITY;
  private bucketMinimum = new Float64Array(ARC_BUCKET_COUNT).fill(Number.POSITIVE_INFINITY);
  private bucketMaximum = new Float64Array(ARC_BUCKET_COUNT).fill(Number.NEGATIVE_INFINITY);

  add(x: number, y: number): void {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    this.count++;
    if (x < this.minimumX) this.minimumX = x;
    if (x > this.maximumX) this.maximumX = x;
    if (y < this.minimumY) this.minimumY = y;
    if (y > this.maximumY) this.maximumY = y;
    const wrapped = ((x % 360) + 360) % 360;
    const bucket = Math.min(Math.floor(wrapped * ARC_BUCKETS_PER_DEGREE), ARC_BUCKET_COUNT - 1);
    if (wrapped < this.bucketMinimum[bucket]) this.bucketMinimum[bucket] = wrapped;
    if (wrapped > this.bucketMaximum[bucket]) this.bucketMaximum[bucket] = wrapped;
  }

  /**
   * GeoJSON edges interpolate the coordinates as written (RFC 7946 section 3.1.1). Crossing
   * geometries should be cut at 180 into separate parts (section 3.1.9); a broad uncut edge
   * therefore covers its interior longitudes, rather than silently choosing its complement.
   */
  addPath(positions: readonly CatalogPosition[], close: boolean): void {
    let previous: CatalogPosition | undefined;
    for (const position of positions) {
      this.add(position[0], position[1]);
      if (previous) this.addEdge(previous, position);
      previous = position;
    }
    if (close && previous && positions.length > 0) this.addEdge(previous, positions[0]);
  }

  private addEdge(first: CatalogPosition, second: CatalogPosition): void {
    if (![first[0], first[1], second[0], second[1]].every(Number.isFinite)) return;
    const minimum = Math.min(first[0], second[0]);
    const width = Math.abs(second[0] - first[0]);
    if (width >= 360) {
      this.addLongitudeInterval(0, 360);
      return;
    }
    const start = ((minimum % 360) + 360) % 360;
    const end = start + width;
    this.addLongitudeInterval(start, Math.min(end, 360));
    if (end > 360) this.addLongitudeInterval(0, end - 360);
  }

  private addLongitudeInterval(start: number, end: number): void {
    const firstBucket = Math.min(Math.floor(start * ARC_BUCKETS_PER_DEGREE), ARC_BUCKET_COUNT - 1);
    const lastBucket = Math.min(Math.floor(end * ARC_BUCKETS_PER_DEGREE), ARC_BUCKET_COUNT - 1);
    for (let bucket = firstBucket; bucket <= lastBucket; bucket++) {
      this.bucketMinimum[bucket] = Math.min(
        this.bucketMinimum[bucket],
        Math.max(start, bucket / ARC_BUCKETS_PER_DEGREE)
      );
      this.bucketMaximum[bucket] = Math.max(
        this.bucketMaximum[bucket],
        Math.min(end, (bucket + 1) / ARC_BUCKETS_PER_DEGREE)
      );
    }
  }

  /** Minimum and maximum of every coordinate, as found; undefined without a position. */
  getPlain(): Extent | undefined {
    return this.count > 0
      ? [this.minimumX, this.minimumY, this.maximumX, this.maximumY]
      : undefined;
  }

  /** The extent for geographic coordinates, see the class comment; undefined without a position. */
  getCovering(): Extent | undefined {
    const plain = this.getPlain();
    if (!plain) return undefined;
    let firstMinimum = Number.NaN;
    let previousMaximum = Number.NaN;
    let westValue = 0;
    let eastValue = 0;
    let largestGap = -1;
    for (let bucket = 0; bucket < ARC_BUCKET_COUNT; bucket++) {
      const minimum = this.bucketMinimum[bucket];
      if (minimum === Number.POSITIVE_INFINITY) continue;
      if (Number.isNaN(firstMinimum)) {
        firstMinimum = minimum;
      } else if (minimum - previousMaximum > largestGap) {
        largestGap = minimum - previousMaximum;
        westValue = minimum;
        eastValue = previousMaximum;
      }
      previousMaximum = this.bucketMaximum[bucket];
    }
    // The gap across 0/360 runs from the largest longitude round to the smallest.
    if (firstMinimum + 360 - previousMaximum > largestGap) {
      largestGap = firstMinimum + 360 - previousMaximum;
      westValue = firstMinimum;
      eastValue = previousMaximum;
    }
    const arcWidth = 360 - largestGap;
    if (!(arcWidth < plain[2] - plain[0] - 1e-9)) return plain;
    const toSigned = (degrees: number): number => (degrees > 180 ? degrees - 360 : degrees);
    return [toSigned(westValue), plain[1], toSigned(eastValue), plain[3]];
  }
}

/**
 * Puts complete parts in the feature's covering longitude arc. A part moves as a whole, so a
 * broad uncut edge keeps its literal interior; only the relative frames of separate parts change.
 */
function getGeographicGeometryAnchor(geometry: CatalogGeometry): [number, number] | undefined {
  const polygons = getPolygonParts(geometry);
  const lines = getLineParts(geometry);
  let west: number | undefined;
  if (polygons.length > 1 || lines.length > 1) {
    const extent = new ExtentAccumulator();
    for (const polygon of polygons) for (const ring of polygon) extent.addPath(ring, true);
    for (const line of lines) extent.addPath(line, false);
    west = extent.getCovering()?.[0];
  }
  const alignPart = (paths: readonly (readonly CatalogPosition[])[]): CatalogPosition[][] => {
    let minimumLongitude = Number.POSITIVE_INFINITY;
    for (const path of paths)
      for (const position of path)
        if (Number.isFinite(position[0]) && Number.isFinite(position[1]))
          minimumLongitude = Math.min(minimumLongitude, position[0]);
    const shift =
      west !== undefined && Number.isFinite(minimumLongitude)
        ? Math.ceil((west - minimumLongitude) / 360) * 360
        : 0;
    return paths.map(path => path.map(position => [position[0] + shift, position[1]]));
  };
  const anchor =
    polygons.length > 0
      ? getPolygonAnchor({type: 'MultiPolygon', coordinates: polygons.map(alignPart)})
      : getPathAnchor(
          lines.map(line => alignPart([line])[0]),
          false
        );
  if (!anchor) return undefined;
  // Avoid adding 180 before modulo: that would round tiny longitudes down to zero.
  const longitude = anchor[0] % 360;
  return [
    longitude > 180 ? longitude - 360 : longitude < -180 ? longitude + 360 : longitude,
    anchor[1]
  ];
}

type GeometryAnalysis = {
  counts: KeplerGeometryFeatureCounts;
  dominant?: GeometryFamily;
  vertexCount: number;
  /** Plain minimum and maximum of every vertex. */
  extent?: Extent;
  /** The geographic covering extent of every vertex; see {@link ExtentAccumulator}. */
  coveringExtent?: Extent;
  /** x0, y0, x1, y1, ...: the points, or one anchor per feature of the dominant family. */
  anchors: Float64Array;
  /** Anchors computed in a common longitude frame per feature, for geographic spacing. */
  geographicAnchors: Float64Array;
};

/**
 * Reads kind counts, vertex count, extent and the locations used for spacing from one geometry
 * per row.
 *
 * Mixed geometry: the dataset's kind is the family with the most rows (ties go to polygons, then
 * lines, then points, the family an area analysis can use). The catalog readers ignore the rows
 * of other families, so that is the data an analysis would see. The counts are kept and a note says
 * what was ignored. Rows without a readable geometry are counted, never silently dropped.
 *
 * Spacing locations: for points, every position (a multi-point row gives several); for lines, one
 * anchor per row (a length-weighted centroid for geographic coordinates, the catalog's vertex
 * mean for planar coordinates); for polygons,
 * one per row from {@link getPolygonAnchor}. The profile's polygon anchors can differ from the
 * catalog's `getFeatureAnchor` for polygons with holes or with parts of mixed orientation, and for
 * small polygons: the catalog's ignores holes, lets parts of opposite orientation cancel and falls
 * back to the vertex mean below an absolute area of 1e-12.
 */
function analyzeGeometry(geometries: ArrayLike<CatalogGeometry | undefined>): GeometryAnalysis {
  const counts: KeplerGeometryFeatureCounts = {point: 0, line: 0, polygon: 0, withoutGeometry: 0};
  let vertexCount = 0;
  const extentAccumulator = new ExtentAccumulator();
  const widen = (position: readonly [number, number]): void => {
    extentAccumulator.add(position[0], position[1]);
  };

  for (let index = 0; index < geometries.length; index++) {
    const geometry = geometries[index] ?? null;
    const family = getGeometryFamily(geometry);
    if (!family) {
      counts.withoutGeometry++;
      continue;
    }
    counts[family]++;
    vertexCount += countFeatureVertices({geometry});
    if (family === 'point') for (const position of getPointPositions(geometry)) widen(position);
    else if (family === 'line')
      for (const part of getLineParts(geometry)) extentAccumulator.addPath(part, false);
    else
      for (const polygon of getPolygonParts(geometry))
        for (const ring of polygon) extentAccumulator.addPath(ring, true);
  }

  let dominant: GeometryFamily | undefined;
  let dominantCount = 0;
  for (const family of ['polygon', 'line', 'point'] as const) {
    if (counts[family] > dominantCount) {
      dominant = family;
      dominantCount = counts[family];
    }
  }

  const anchors: number[] = [];
  const geographicAnchors: number[] = [];
  if (dominant) {
    for (let index = 0; index < geometries.length; index++) {
      const geometry = geometries[index] ?? null;
      if (getGeometryFamily(geometry) !== dominant) continue;
      if (dominant === 'point') {
        for (const position of getPointPositions(geometry)) {
          anchors.push(position[0], position[1]);
          geographicAnchors.push(position[0], position[1]);
        }
      } else {
        const anchor =
          dominant === 'polygon'
            ? getPolygonAnchor(geometry as CatalogGeometry)
            : getFeatureAnchor({geometry} as CatalogFeature, IDENTITY_FRAME);
        if (anchor) anchors.push(anchor[0], anchor[1]);
        const geographicAnchor = getGeographicGeometryAnchor(geometry as CatalogGeometry);
        if (geographicAnchor) geographicAnchors.push(geographicAnchor[0], geographicAnchor[1]);
      }
    }
  }

  return {
    counts,
    dominant,
    vertexCount,
    extent: extentAccumulator.getPlain(),
    coveringExtent: extentAccumulator.getCovering(),
    anchors: Float64Array.from(anchors),
    geographicAnchors: Float64Array.from(geographicAnchors)
  };
}

function describeGeometryMix(counts: KeplerGeometryFeatureCounts): string | undefined {
  const present = (['polygon', 'line', 'point'] as const).filter(family => counts[family] > 0);
  if (present.length < 2) return undefined;
  const dominant = present.reduce((best, family) =>
    counts[family] > counts[best] ? family : best
  );
  const ignored = present
    .filter(family => family !== dominant)
    .map(family => `${counts[family]} ${family} ${counts[family] === 1 ? 'row' : 'rows'}`);
  return `Mixed geometry: ${counts[dominant]} ${dominant} rows are the dominant kind and ${ignored.join(' and ')} are of other kinds. The catalog rejects an input that is not all one geometry kind, so the dataset must be split or filtered to one kind before the catalog can read it.`;
}

// ---------------------------------------------------------------------------------------------
// Shared assembly
// ---------------------------------------------------------------------------------------------

function toCoordinateModel(system: CatalogCoordinateSystem): GPUAnalysisCoordinateModel {
  return system === 'planar' ? 'planar' : 'geographic';
}

function validateOptions(functionName: string, options: KeplerDatasetProfileOptions): void {
  if (typeof options.datasetId !== 'string' || options.datasetId.length === 0)
    throw new Error(`${functionName} options.datasetId must be a non-empty string`);
  if (typeof options.label !== 'string')
    throw new Error(`${functionName} options.label must be a string`);
  if (options.revision !== undefined && !Number.isInteger(options.revision))
    throw new Error(`${functionName} options.revision must be an integer; got ${options.revision}`);
  if (
    options.maximumSampleSize !== undefined &&
    (!Number.isInteger(options.maximumSampleSize) ||
      options.maximumSampleSize < MINIMUM_SAMPLE_SIZE)
  )
    throw new RangeError(
      `${functionName} options.maximumSampleSize must be an integer of at least ${MINIMUM_SAMPLE_SIZE}; got ${options.maximumSampleSize}`
    );
  if (
    options.coordinateSystem !== undefined &&
    options.coordinateSystem !== 'EPSG:4326' &&
    options.coordinateSystem !== 'planar'
  )
    throw new Error(
      `${functionName} options.coordinateSystem must be 'EPSG:4326' or 'planar'; got ${String(options.coordinateSystem)}`
    );
}

/** Row indices of a deterministic stride sample, or undefined when every row is used. */
function getSampleRowCount(rowCount: number, maximumSampleSize: number | undefined): number {
  return maximumSampleSize !== undefined && rowCount > maximumSampleSize
    ? maximumSampleSize
    : rowCount;
}

function getSampleRowIndex(rowCount: number, sampleRowCount: number, sampleIndex: number): number {
  return sampleRowCount === rowCount
    ? sampleIndex
    : Math.floor((sampleIndex * rowCount) / sampleRowCount);
}

type SpatialFacts = {
  kind: GPUSpatialAnalysisDataKind;
  catalogGeometry?: 'point' | 'line' | 'polygon';
  vertexCount?: number;
  /** Plain minimum and maximum, in source coordinates. */
  extent?: Extent;
  /** The extent for a geographic model: smallest covering longitude arc (see {@link ExtentAccumulator}). */
  coveringExtent?: Extent;
  /** x, y pairs for the spacing statistics. */
  locations: Float64Array;
  /** Geometry anchors in a common longitude frame; point tables use locations for both models. */
  geographicLocations?: Float64Array;
  geometryFeatureCounts?: KeplerGeometryFeatureCounts;
  /** Why the coordinates are what they are, when something besides the extent decided it. */
  coordinateReason?: string;
  pointColumns?: {longitude: string; latitude: string};
  notes: string[];
};

/**
 * Full-data summaries for both coordinate models are computed while the source locations are
 * available. Each spacing computation uses the analysis package's deterministic hashed sample
 * for neighbours and all valid locations for counts, extent and centre. Retaining summaries
 * instead of locations bounds memory and avoids treating a retained subset as a whole dataset.
 * These facts are private to the profile object and do not survive JSON or a host's object spread.
 */
type ProfileInternals = {
  spacingByModel: Record<GPUAnalysisCoordinateModel, GPUAnalysisSpacingStatistics | undefined>;
  modelNotes: Record<GPUAnalysisCoordinateModel, string[]>;
  /** Plain and covering extents of every vertex, for either coordinate model. */
  plainExtent?: Extent;
  coveringExtent?: Extent;
  undeclared?: KeplerProfiledDataset;
};

const PROFILE_INTERNALS = new WeakMap<object, ProfileInternals>();

/** The profile as it was before declarations were applied through `options.declarations`, or the profile itself. */
export function getKeplerDatasetProfileUndeclared<ProfileType extends KeplerDatasetProfile>(
  profile: ProfileType
): ProfileType {
  return (PROFILE_INTERNALS.get(profile)?.undeclared as ProfileType | undefined) ?? profile;
}

/** Marks the notes that describe the coordinate model, so that a change of model can replace them. */
const MODEL_NOTE_PREFIX = 'Coordinate model: ';

/**
 * Notes for a model that does not fit the locations. Under `geographic`, a location with latitude
 * outside [-90, 90] is left out of the spacing statistics (no great-circle distance exists for it)
 * and a longitude outside [-180, 180] is kept; both are counted. A declared geographic model with an
 * extent outside degrees means the values look projected.
 */
function describeModelNotes(
  locations: Float64Array,
  extent: readonly [number, number, number, number] | undefined,
  model: GPUAnalysisCoordinateModel
): string[] {
  if (model !== 'geographic' || !extent) return [];
  if (
    extent[1] >= -LATITUDE_LIMIT_DEGREES &&
    extent[3] <= LATITUDE_LIMIT_DEGREES &&
    extent[0] >= -LONGITUDE_LIMIT_DEGREES &&
    extent[2] <= LONGITUDE_LIMIT_DEGREES
  )
    return [];
  let latitudeOutside = 0;
  let longitudeOutside = 0;
  for (let index = 0; index < locations.length; index += 2) {
    if (Math.abs(locations[index]) > LONGITUDE_LIMIT_DEGREES) longitudeOutside++;
    if (Math.abs(locations[index + 1]) > LATITUDE_LIMIT_DEGREES) latitudeOutside++;
  }
  const total = locations.length / 2;
  const notes: string[] = [];
  if (latitudeOutside > 0)
    notes.push(
      `${MODEL_NOTE_PREFIX}${latitudeOutside} of ${total} locations have a latitude outside -${LATITUDE_LIMIT_DEGREES} to ${LATITUDE_LIMIT_DEGREES} and are left out of the spacing statistics.`
    );
  if (longitudeOutside > 0)
    notes.push(
      `${MODEL_NOTE_PREFIX}${longitudeOutside} of ${total} locations have a longitude outside -${LONGITUDE_LIMIT_DEGREES} to ${LONGITUDE_LIMIT_DEGREES}; the values look projected, not geographic.`
    );
  return notes;
}

/** True for longitudes written 0 to 360 (Pacific-centred data): none below 0, some above 180. */
function hasWrappedLongitudes(extent: Extent): boolean {
  return (
    extent[0] >= 0 &&
    extent[2] > LONGITUDE_LIMIT_DEGREES &&
    extent[2] <= 360 &&
    extent[1] >= -LATITUDE_LIMIT_DEGREES &&
    extent[3] <= LATITUDE_LIMIT_DEGREES
  );
}

function getWrappedLongitudeSentence(extent: Extent): string | undefined {
  return hasWrappedLongitudes(extent)
    ? 'The longitudes look like 0 to 360 values (Pacific-centred), not -180 to 180.'
    : undefined;
}

/**
 * Notes for a guessed geographic model that needs confirming. Longitudes written 0 to 360 are
 * read as geographic, but the catalog expects -180 to 180 and its grid sizing breaks on them. A
 * low-confidence guess for a small extent at the origin may be local metres or site coordinates.
 */
function describeCoordinateGuessNotes(extent: Extent, confidence: 'high' | 'low'): string[] {
  const guess = guessGPUAnalysisCoordinateModel(extent);
  if (guess.model !== 'geographic') return [];
  if (hasWrappedLongitudes(extent))
    return [
      `${MODEL_NOTE_PREFIX}the longitudes run from ${extent[0]} to ${extent[2]}; the catalog expects -180 to 180, so values above 180 need 360 subtracted before analysis (the stored coordinates are not changed).`
    ];
  if (confidence === 'low')
    return [
      `${MODEL_NOTE_PREFIX}small extents near the origin may be local metres or site coordinates, not degrees; confirm the coordinate system before any distance-based analysis.`
    ];
  return [];
}

/**
 * Sets `rowSemantics` from `guessGPUAnalysisRowSemantics` (source `guess`, always low confidence
 * and a reason), or removes a guessed value the evidence no longer supports. A declared value is
 * never touched. The guess reads the column levels and the spacing, so it is made again whenever a
 * declaration changes those.
 */
function setGuessedRowSemantics(profile: KeplerProfiledDataset): void {
  if (profile.rowSemantics?.source === 'declared') return;
  const guess = guessGPUAnalysisRowSemantics(profile);
  if (guess)
    profile.rowSemantics = {
      value: guess.value,
      source: 'guess',
      reason: guess.reason,
      confidence: guess.confidence
    };
  else delete profile.rowSemantics;
}

function assembleProfile(
  options: KeplerDatasetProfileOptions,
  rowCount: number,
  columns: KeplerDatasetProfileColumn[],
  geometryColumns: readonly string[],
  spatial: SpatialFacts
): KeplerProfiledDataset {
  const hostModel = options.coordinateSystem
    ? toCoordinateModel(options.coordinateSystem)
    : undefined;
  const profile: KeplerProfiledDataset = {
    datasetId: options.datasetId,
    label: options.label,
    kind: spatial.kind,
    rowCount,
    columns,
    geometryColumns
  };
  if (options.revision !== undefined) profile.revision = options.revision;
  if (spatial.catalogGeometry) profile.catalogGeometry = spatial.catalogGeometry;
  if (spatial.pointColumns) profile.pointColumns = spatial.pointColumns;
  if (spatial.vertexCount !== undefined) profile.vertexCount = spatial.vertexCount;
  if (spatial.geometryFeatureCounts) {
    profile.geometryFeatureCounts = spatial.geometryFeatureCounts;
    const {point, line, polygon} = spatial.geometryFeatureCounts;
    if ([point, line, polygon].filter(count => count > 0).length > 1)
      profile.mixedGeometry = {point, line, polygon};
  }
  const notes = [...spatial.notes];

  const spacingByModel = {
    geographic: computeGPUAnalysisSpacingStatistics(
      spatial.geographicLocations ?? spatial.locations,
      {
        coordinateModel: 'geographic',
        maximumSampleSize: options.maximumSampleSize
      }
    ),
    planar: computeGPUAnalysisSpacingStatistics(spatial.locations, {
      coordinateModel: 'planar',
      maximumSampleSize: options.maximumSampleSize
    })
  };
  const modelNotes = {
    geographic: describeModelNotes(spatial.locations, spatial.extent, 'geographic'),
    planar: describeModelNotes(spatial.locations, spatial.extent, 'planar')
  };
  if (spatial.extent) {
    const guess = guessGPUAnalysisCoordinateModel(spatial.extent);
    const model = hostModel ?? guess.model;
    // The extent follows the model: smallest covering longitude arc for geographic data,
    // minimum and maximum for planar data.
    profile.extent = model === 'geographic' ? spatial.coveringExtent : spatial.extent;
    profile.coordinates = hostModel
      ? {
          model,
          source: 'declared',
          reason: `Declared coordinate system ${options.coordinateSystem}.`
        }
      : {
          model,
          source: 'guess',
          confidence: guess.confidence,
          reason: [
            spatial.coordinateReason,
            guess.reason,
            getWrappedLongitudeSentence(spatial.extent)
          ]
            .filter(Boolean)
            .join(' ')
        };
    if (!hostModel) notes.push(...describeCoordinateGuessNotes(spatial.extent, guess.confidence));
    const spacing = spacingByModel[model];
    // `contiguity` stays absent: neighbour counts under queen contiguity are a GPU operation
    // (the spatial-weights contributors), not something a profile computes on the CPU at load.
    if (spacing) profile.spacing = spacing;
    if (hostModel) notes.push(...modelNotes[model]);
  }
  if (notes.length > 0) profile.notes = notes;
  setGuessedRowSemantics(profile);
  PROFILE_INTERNALS.set(profile, {
    spacingByModel,
    modelNotes,
    plainExtent: spatial.extent,
    coveringExtent: spatial.coveringExtent
  });
  if (!options.declarations) return profile;
  const declared = applyKeplerDatasetProfileDeclarations(profile, options.declarations);
  const internals = PROFILE_INTERNALS.get(declared);
  if (internals) internals.undeclared = profile;
  return declared;
}

// ---------------------------------------------------------------------------------------------
// Feature collections
// ---------------------------------------------------------------------------------------------

/**
 * Profiles a GeoJSON feature collection in the catalog's input form.
 *
 * Rows are features, so a feature without geometry is a row (it counts in `rowCount` and in the
 * columns) but not a location. Columns are the property names in order of first appearance; a
 * feature without a property gives a null for it. The column type is inferred from the values:
 * a column of finite numbers is `integer` or `number`, of booleans `boolean`, of ISO 8601 text
 * `time`, of other text `category` (at most 100 distinct values, at most half as many as rows) or
 * `text`. A column that mixes numbers and text (or booleans) is `text` and counts every value,
 * because nothing numeric can be said faithfully about it; a number written as a string stays
 * text. Null, undefined, NaN and the infinities count as null. Objects and arrays count as null.
 *
 * The coordinate system is the `coordinateSystem` option (a feature collection carries none),
 * else a guess from the extent of every vertex. The kind comes from the geometry; see
 * `analyzeGeometry` for mixed geometry. Features with no geometry anywhere give kind `attributes`.
 */
export function profileKeplerFeatureCollection(
  collection: CatalogFeatureCollection,
  options: KeplerDatasetProfileOptions
): KeplerProfiledDataset {
  const functionName = 'profileKeplerFeatureCollection';
  validateOptions(functionName, options);
  if (!collection || !Array.isArray(collection.features))
    throw new Error(`${functionName} collection must be a FeatureCollection with a features array`);

  const features = collection.features;
  const rowCount = features.length;
  const sampleRowCount = getSampleRowCount(rowCount, options.maximumSampleSize);
  const accumulators = new Map<string, ColumnAccumulator>();
  for (let sampleIndex = 0; sampleIndex < sampleRowCount; sampleIndex++) {
    const properties =
      features[getSampleRowIndex(rowCount, sampleRowCount, sampleIndex)]?.properties;
    if (!properties) continue;
    for (const key in properties) {
      let accumulator = accumulators.get(key);
      if (!accumulator) {
        accumulator = new ColumnAccumulator();
        accumulators.set(key, accumulator);
      }
      accumulator.add(properties[key]);
    }
  }
  const columns: KeplerDatasetProfileColumn[] = [];
  const columnNotes: string[] = [];
  for (const [name, accumulator] of accumulators)
    columns.push(
      finishColumn(
        name,
        accumulator,
        rowCount,
        sampleRowCount,
        options.columnTypes?.[name],
        columnNotes
      )
    );

  const geometries: (CatalogGeometry | undefined)[] = new Array(rowCount);
  for (let index = 0; index < rowCount; index++) geometries[index] = features[index]?.geometry;
  const notes: string[] = [];
  const spatial = buildGeometryFacts(geometries, notes);
  notes.push(...columnNotes);
  if (sampleRowCount < rowCount)
    notes.push(
      `Column statistics are computed on a stride sample of ${sampleRowCount} of ${rowCount} rows.`
    );
  return assembleProfile(options, rowCount, columns, [], spatial);
}

function buildGeometryFacts(
  geometries: ArrayLike<CatalogGeometry | undefined>,
  notes: string[]
): SpatialFacts {
  const geometry = analyzeGeometry(geometries);
  const mixNote = describeGeometryMix(geometry.counts);
  if (mixNote) notes.push(mixNote);
  if (!geometry.dominant) {
    return {
      kind: 'attributes',
      locations: new Float64Array(0),
      geometryFeatureCounts: geometry.counts,
      notes
    };
  }
  return {
    kind: SPATIAL_KIND_BY_FAMILY[geometry.dominant],
    catalogGeometry: CATALOG_GEOMETRY_BY_FAMILY[geometry.dominant],
    vertexCount: geometry.vertexCount,
    extent: geometry.extent,
    coveringExtent: geometry.coveringExtent,
    locations: geometry.anchors,
    geographicLocations: geometry.geographicAnchors,
    geometryFeatureCounts: geometry.counts,
    notes
  };
}

// ---------------------------------------------------------------------------------------------
// Column tables
// ---------------------------------------------------------------------------------------------

function isCatalogOutputTable(
  table: CatalogOutputTable | KeplerProfileColumnTable
): table is CatalogOutputTable {
  return !Array.isArray(table.columns);
}

/** Makes the column list of a {@link CatalogOutputTable}. */
function readCatalogOutputTable(table: CatalogOutputTable): KeplerProfileColumnTable {
  return {
    rowCount: table.rowCount,
    geometryColumn: table.geometryColumn,
    columns: Object.keys(table.columnTypes).map(name => ({
      name,
      type: table.columnTypes[name],
      values: table.columns[name] ?? []
    }))
  };
}

/** What a column pair must satisfy to be taken as coordinates. */
type CoordinateRange = {finiteCount: number; outOfRangeCount: number};

function measureCoordinateRange(values: ArrayLike<unknown>, limit: number): CoordinateRange {
  let finiteCount = 0;
  let outOfRangeCount = 0;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    finiteCount++;
    if (value < -limit || value > limit) outOfRangeCount++;
  }
  return {finiteCount, outOfRangeCount};
}

type CoordinateCandidate = {latitude: string; longitude: string; plain: boolean};

/**
 * Finds latitude and longitude columns by name the way kepler.gl does, with the same word lists
 * (`lat`, `latitude`; `lng`, `lon`, `long`, `longitude`) and the same pairing of a prefix or
 * suffix (`pickup_lat` goes with `pickup_lng`, `lat_start` with `lon_start`). Words are matched
 * whole after splitting on separators, case and digits, so `latency` is not a latitude. A plain
 * pair (`lat`, `lng`) comes first, then pairs by the position of their latitude column.
 */
function findNamePairs(
  names: readonly string[],
  firstWords: ReadonlySet<string>,
  secondWords: ReadonlySet<string>
): {first: string; second: string; plain: boolean}[] {
  const firsts = new Map<string, string>();
  const seconds = new Map<string, string>();
  const order: string[] = [];
  for (const name of names) {
    const tokens = tokenizeGPUAnalysisColumnName(name);
    const firstIndex = tokens.findIndex(token => firstWords.has(token));
    const secondIndex = tokens.findIndex(token => secondWords.has(token));
    // A name holding both words (`lat_lon`) or neither is no single coordinate.
    if (firstIndex < 0 === secondIndex < 0) continue;
    const index = firstIndex >= 0 ? firstIndex : secondIndex;
    const key = `${tokens.slice(0, index).join('_')}|${tokens.slice(index + 1).join('_')}`;
    const target = firstIndex >= 0 ? firsts : seconds;
    if (!target.has(key)) target.set(key, name);
    if (firstIndex >= 0 && !order.includes(key)) order.push(key);
  }
  const pairs: {first: string; second: string; plain: boolean}[] = [];
  for (const key of order) {
    const second = seconds.get(key);
    if (second !== undefined)
      pairs.push({first: firsts.get(key) as string, second, plain: key === '|'});
  }
  return pairs.sort((first, second) => Number(second.plain) - Number(first.plain));
}

function findCoordinateCandidates(names: readonly string[]): CoordinateCandidate[] {
  return findNamePairs(names, LATITUDE_WORDS, LONGITUDE_WORDS).map(pair => ({
    latitude: pair.first,
    longitude: pair.second,
    plain: pair.plain
  }));
}

function describeRejection(
  candidate: CoordinateCandidate,
  latitudeRange: CoordinateRange | undefined,
  longitudeRange: CoordinateRange | undefined
): string | undefined {
  const label = `${candidate.longitude}/${candidate.latitude}`;
  if (!latitudeRange || !longitudeRange) return `${label} is not numeric`;
  if (latitudeRange.finiteCount === 0 || longitudeRange.finiteCount === 0)
    return `${label} has no finite value`;
  if (latitudeRange.outOfRangeCount > 0)
    return `${candidate.latitude} has ${latitudeRange.outOfRangeCount} values outside -${LATITUDE_LIMIT_DEGREES} to ${LATITUDE_LIMIT_DEGREES}`;
  if (longitudeRange.outOfRangeCount > 0)
    return `${candidate.longitude} has ${longitudeRange.outOfRangeCount} values outside -${LONGITUDE_LIMIT_DEGREES} to ${LONGITUDE_LIMIT_DEGREES}`;
  return undefined;
}

function gatherColumnPoints(
  longitudes: ArrayLike<unknown>,
  latitudes: ArrayLike<unknown>
): {locations: Float64Array; extent?: Extent; coveringExtent?: Extent} {
  const locations: number[] = [];
  const extentAccumulator = new ExtentAccumulator();
  for (let index = 0; index < longitudes.length; index++) {
    const x = longitudes[index];
    const y = latitudes[index];
    if (
      typeof x !== 'number' ||
      typeof y !== 'number' ||
      !Number.isFinite(x) ||
      !Number.isFinite(y)
    )
      continue;
    locations.push(x, y);
    extentAccumulator.add(x, y);
  }
  return {
    locations: Float64Array.from(locations),
    extent: extentAccumulator.getPlain(),
    coveringExtent: extentAccumulator.getCovering()
  };
}

/**
 * Profiles a dataset held as columns, including the catalog's {@link CatalogOutputTable}.
 *
 * Declared types of a catalog table map as follows: `float64` to `number`, `int32` to `integer`,
 * `boolean` to `boolean`, and `string` to `time` when every non-null value is an ISO 8601
 * date or date-time, else `category` or `text` by the rule of {@link profileKeplerFeatureCollection}.
 * Analysis types (`number`, `integer`, `category`, `time`, `text`, `boolean`) are taken as given. A
 * value of another class than the declared one counts as null (a number in a `string` column),
 * and the statistics describe the declared class only. Column types absent: inferred from values
 * exactly as for features.
 *
 * Geometry: columns named in `geometryColumns` (default: the table's `geometryColumn`) hold
 * GeoJSON text. Only the first is analysed; every one is listed in `geometryColumns` and left out
 * of the column statistics. Unparseable text counts as a row without geometry. Passing both
 * geometry columns and `pointColumns` is an error.
 *
 * Points: with `pointColumns` the rows are points at those columns. With neither given, a
 * latitude and longitude pair is looked for by name (see `findCoordinateCandidates`) and taken only
 * when every finite value of the latitude lies within [-90, 90] and of the longitude within
 * [-180, 180], which kepler.gl does not check. A rejected pair is explained in `notes`; extra
 * valid pairs are named in `coordinates.reason`. The range check reads every row even when the
 * statistics are sampled. Spacing uses rows where both coordinates are finite.
 */
export function profileKeplerColumnTable(
  table: CatalogOutputTable | KeplerProfileColumnTable,
  options: KeplerColumnTableProfileOptions
): KeplerProfiledDataset {
  const functionName = 'profileKeplerColumnTable';
  validateOptions(functionName, options);
  const input = isCatalogOutputTable(table) ? readCatalogOutputTable(table) : table;
  const rowCount = input.rowCount;
  if (!Number.isInteger(rowCount) || rowCount < 0)
    throw new RangeError(
      `${functionName} table.rowCount must be a non-negative integer; got ${rowCount}`
    );
  const names = new Set<string>();
  for (const column of input.columns) {
    if (typeof column.name !== 'string' || column.name.length === 0)
      throw new Error(`${functionName} every column needs a non-empty name`);
    if (names.has(column.name))
      throw new Error(`${functionName} column "${column.name}" appears twice`);
    names.add(column.name);
    if (column.values.length !== rowCount)
      throw new RangeError(
        `${functionName} column "${column.name}" has ${column.values.length} values; table.rowCount is ${rowCount}`
      );
  }
  const geometryColumns =
    options.geometryColumns ?? (input.geometryColumn ? [input.geometryColumn] : []);
  for (const name of geometryColumns)
    if (!names.has(name))
      throw new Error(`${functionName} geometry column "${name}" is not a column of the table`);
  if (geometryColumns.length > 0 && options.pointColumns)
    throw new Error(
      `${functionName} give either options.pointColumns or geometry columns, not both`
    );
  if (options.pointColumns) {
    for (const name of [options.pointColumns.longitude, options.pointColumns.latitude])
      if (!names.has(name))
        throw new Error(`${functionName} point column "${name}" is not a column of the table`);
  }

  const sampleRowCount = getSampleRowCount(rowCount, options.maximumSampleSize);
  const columns: KeplerDatasetProfileColumn[] = [];
  const columnNotes: string[] = [];
  const valuesByName = new Map<string, ArrayLike<unknown>>();
  for (const column of input.columns) {
    valuesByName.set(column.name, column.values);
    if (geometryColumns.includes(column.name)) continue;
    const accumulator = new ColumnAccumulator();
    for (let sampleIndex = 0; sampleIndex < sampleRowCount; sampleIndex++)
      accumulator.add(column.values[getSampleRowIndex(rowCount, sampleRowCount, sampleIndex)]);
    columns.push(
      finishColumn(
        column.name,
        accumulator,
        rowCount,
        sampleRowCount,
        column.type ?? options.columnTypes?.[column.name],
        columnNotes
      )
    );
  }

  const notes: string[] = [...columnNotes];
  let spatial: SpatialFacts = {kind: 'attributes', locations: new Float64Array(0), notes};
  if (geometryColumns.length > 0) {
    const texts = valuesByName.get(geometryColumns[0]) as ArrayLike<unknown>;
    const geometries: (CatalogGeometry | undefined)[] = new Array(rowCount);
    for (let index = 0; index < rowCount; index++) {
      const text = texts[index];
      try {
        geometries[index] =
          typeof text === 'string' ? (JSON.parse(text) as CatalogGeometry) : undefined;
      } catch {
        geometries[index] = undefined;
      }
    }
    spatial = buildGeometryFacts(geometries, notes);
    if (geometryColumns.length > 1)
      notes.push(`Only the first geometry column "${geometryColumns[0]}" is analysed.`);
  } else {
    const chosen = chooseCoordinateColumns(options.pointColumns, columns, valuesByName, notes);
    notes.push(...describePlanarPairs(columns, chosen?.pointColumns));
    if (chosen) {
      const points = gatherColumnPoints(
        valuesByName.get(chosen.pointColumns.longitude) as ArrayLike<unknown>,
        valuesByName.get(chosen.pointColumns.latitude) as ArrayLike<unknown>
      );
      if (!points.extent)
        notes.push(
          `Point columns ${chosen.pointColumns.longitude}/${chosen.pointColumns.latitude} never both have a finite value in the same row, so the dataset has no extent and no spacing statistics.`
        );
      spatial = {
        kind: 'points',
        catalogGeometry: 'point',
        extent: points.extent,
        coveringExtent: points.coveringExtent,
        locations: points.locations,
        pointColumns: chosen.pointColumns,
        coordinateReason: chosen.reason,
        notes
      };
    }
  }
  if (sampleRowCount < rowCount)
    notes.push(
      `Column statistics are computed on a stride sample of ${sampleRowCount} of ${rowCount} rows.`
    );
  return assembleProfile(options, rowCount, columns, geometryColumns, spatial);
}

/** Notes for numeric x/y and easting/northing pairs that were not taken as the point columns. */
function describePlanarPairs(
  columns: readonly KeplerDatasetProfileColumn[],
  chosen: {longitude: string; latitude: string} | undefined
): string[] {
  const numericNames = columns
    .filter(column => column.type === 'number' || column.type === 'integer')
    .map(column => column.name);
  const notes: string[] = [];
  for (const [firstWords, secondWords] of PLANAR_PAIR_WORDS) {
    for (const pair of findNamePairs(numericNames, firstWords, secondWords)) {
      if (chosen && (pair.first === chosen.longitude || pair.second === chosen.latitude)) continue;
      notes.push(
        `Columns ${pair.first}/${pair.second} look like planar point columns and were not taken as coordinates; give them as options.pointColumns (with coordinateSystem 'planar') to use them.`
      );
    }
  }
  return notes;
}

function chooseCoordinateColumns(
  given: {longitude: string; latitude: string} | undefined,
  columns: readonly KeplerDatasetProfileColumn[],
  valuesByName: ReadonlyMap<string, ArrayLike<unknown>>,
  notes: string[]
): {pointColumns: {longitude: string; latitude: string}; reason: string} | undefined {
  if (given) {
    return {
      pointColumns: {longitude: given.longitude, latitude: given.latitude},
      reason: `Point columns ${given.longitude}/${given.latitude} were given by the host.`
    };
  }
  const numericNames = new Set(
    columns
      .filter(column => column.type === 'number' || column.type === 'integer')
      .map(column => column.name)
  );
  const candidates = findCoordinateCandidates(columns.map(column => column.name));
  const rejected: string[] = [];
  let chosen: CoordinateCandidate | undefined;
  const alternatives: string[] = [];
  for (const candidate of candidates) {
    const isNumeric = numericNames.has(candidate.latitude) && numericNames.has(candidate.longitude);
    const latitudeRange = isNumeric
      ? measureCoordinateRange(
          valuesByName.get(candidate.latitude) as ArrayLike<unknown>,
          LATITUDE_LIMIT_DEGREES
        )
      : undefined;
    const longitudeRange = isNumeric
      ? measureCoordinateRange(
          valuesByName.get(candidate.longitude) as ArrayLike<unknown>,
          LONGITUDE_LIMIT_DEGREES
        )
      : undefined;
    const rejection = describeRejection(candidate, latitudeRange, longitudeRange);
    if (rejection)
      rejected.push(
        `${candidate.longitude}/${candidate.latitude} rejected as coordinates: ${rejection}.`
      );
    else if (!chosen) chosen = candidate;
    else alternatives.push(`${candidate.longitude}/${candidate.latitude}`);
  }
  notes.push(...rejected);
  if (!chosen) return undefined;
  const reason =
    `Detected the coordinate columns ${chosen.longitude}/${chosen.latitude} by name; their values lie within valid degrees.` +
    (alternatives.length > 0
      ? ` Other valid pairs were found and not used: ${alternatives.join(', ')}.`
      : '');
  return {pointColumns: {longitude: chosen.longitude, latitude: chosen.latitude}, reason};
}

// ---------------------------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------------------------

/**
 * Returns a copy of the profile with the declarations applied; the profile is not changed and no
 * column statistic is recomputed. Each declaration replaces the matching guess and marks it
 * `declared`; an entry the declarations leave out keeps what the profile already has (so a
 * second call adds to the first; removing a correction means applying to the profile before it,
 * see {@link getKeplerDatasetProfileUndeclared}, which the cache does).
 *
 * Measurement names that are not a column of the profile are ignored: a saved correction can
 * outlive the column. A coordinate model is only applied to a profile that has coordinates.
 *
 * A declared coordinate model that differs from the profile's changes the unit of every distance,
 * so its spacing is replaced by the summary computed for that model from the full input when
 * the dataset was profiled. Counts and full-data statistics are preserved, and neighbour searches
 * retain their hashed sample's size and provenance. No retained subset is presented as exact.
 * A profile without private summaries (for example after JSON serialization) drops spacing and
 * asks the host to profile the dataset again. The extent uses geometry edges for geographic data
 * and minimum/maximum source coordinates for planar data; without private facts it is kept with
 * a warning when a wrapped extent cannot be interpreted as planar.
 */
export function applyKeplerDatasetProfileDeclarations<ProfileType extends KeplerDatasetProfile>(
  profile: ProfileType,
  declarations: KeplerDatasetProfileDeclarations
): ProfileType {
  const next: ProfileType = {...profile};
  const internals = PROFILE_INTERNALS.get(profile);
  if (declarations.measurement) {
    const measurement = declarations.measurement;
    next.columns = profile.columns.map(column => {
      if (!Object.prototype.hasOwnProperty.call(measurement, column.name)) return column;
      const declared = {
        ...column,
        measurement: measurement[column.name],
        measurementSource: 'declared' as const
      };
      delete declared.measurementReason;
      delete declared.measurementConfidence;
      return declared;
    });
  }
  if (declarations.rowSemantics)
    // A declared value carries no confidence: it is a fact the user stated.
    next.rowSemantics = {
      value: declarations.rowSemantics,
      source: 'declared',
      reason: 'Declared by the user.'
    };
  if (declarations.coordinateModel && (profile.coordinates || profile.extent)) {
    const model = declarations.coordinateModel;
    if (profile.coordinates && profile.coordinates.model !== model) {
      delete next.spacing;
      const existingNotes = (profile as unknown as {notes?: readonly string[]}).notes ?? [];
      const notes = existingNotes.filter(note => !note.startsWith(MODEL_NOTE_PREFIX));
      if (internals) {
        const extent = model === 'geographic' ? internals.coveringExtent : internals.plainExtent;
        if (extent) next.extent = extent;
        const spacing = internals.spacingByModel[model];
        if (spacing) next.spacing = spacing;
        notes.push(...internals.modelNotes[model]);
      } else {
        const extent = profile.extent;
        if (extent && extent[0] > extent[2] && model === 'planar')
          notes.push(
            `${MODEL_NOTE_PREFIX}the extent crosses the antimeridian (west greater than east) and was not recomputed for the declared planar model, which this profile holds no locations for; profile the dataset again with the declaration.`
          );
      }
      if (!internals && profile.spacing) {
        notes.push(
          `${MODEL_NOTE_PREFIX}spacing was dropped because the coordinate model was declared ${model} and this profile holds no locations to recompute it from; profile the dataset again with the declaration.`
        );
      }
      const mutableNext = next as unknown as {notes?: readonly string[]};
      if (notes.length > 0) mutableNext.notes = notes;
      else delete mutableNext.notes;
    }
    next.coordinates = {
      model,
      source: 'declared',
      reason: 'Declared by the user.'
    };
  }
  // The guess of row semantics reads the column levels and the spacing, which these declarations
  // may have changed; a declared value is left alone.
  if (!declarations.rowSemantics && (declarations.measurement || declarations.coordinateModel))
    setGuessedRowSemantics(next as unknown as KeplerProfiledDataset);
  if (internals) PROFILE_INTERNALS.set(next, {...internals});
  return next;
}

// ---------------------------------------------------------------------------------------------
// Description for a dataset card
// ---------------------------------------------------------------------------------------------

const KIND_LABELS: Readonly<Partial<Record<GPUSpatialAnalysisDataKind, string>>> = {
  points: 'Points',
  lines: 'Lines',
  polygons: 'Polygons',
  attributes: 'Table'
};

const NUMBER_FORMAT = new Intl.NumberFormat('en-US');

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${NUMBER_FORMAT.format(count)} ${count === 1 ? singular : plural}`;
}

/**
 * The one-line summary and the measurement chips of a dataset card, for example
 * `Points · 48,210 rows · longitude/latitude · 3 numeric columns · 1 time column`. Coordinate
 * columns are named, not counted as numeric. Parts with a zero count are left out. A chip exists
 * for every column that has a guessed or declared measurement level and says which it is. A
 * coordinate model that was guessed with low confidence is shown as a guess in the headline
 * (`geographic coordinates (guessed, low confidence)`) so a host can ask; guessed row semantics
 * are shown as `rows probably events (guessed, low confidence)` and a declared value as
 * `rows are events`; a confident guess or a
 * declared model is shown plainly. `coordinateModel` repeats the fact for a host that builds its
 * own wording.
 */
export function describeKeplerDatasetProfile(profile: KeplerDatasetProfile): {
  headline: string;
  chips: readonly {column: string; label: string; guessed: boolean}[];
  coordinateModel?: {
    model: GPUAnalysisCoordinateModel;
    guessed: boolean;
    confidence?: 'high' | 'low';
  };
  rowSemantics?: {
    value: GPUAnalysisRowSemantics;
    guessed: boolean;
    confidence?: 'high' | 'low';
  };
} {
  const parts: string[] = [];
  const isMixed = profile.mixedGeometry !== undefined;
  const kindLabel = KIND_LABELS[profile.kind] ?? profile.kind;
  parts.push(isMixed ? `${kindLabel} (mixed geometry)` : kindLabel);
  parts.push(pluralize(profile.rowCount, 'row'));
  const isLowConfidenceGuess =
    profile.coordinates?.source === 'guess' && profile.coordinates.confidence === 'low';
  const guessSuffix = isLowConfidenceGuess ? ' (guessed, low confidence)' : '';
  if (profile.pointColumns)
    parts.push(
      `${profile.pointColumns.longitude}/${profile.pointColumns.latitude}${isLowConfidenceGuess ? ` (${profile.coordinates?.model} guessed, low confidence)` : ''}`
    );
  else if (profile.coordinates)
    parts.push(`${profile.coordinates.model} coordinates${guessSuffix}`);

  if (profile.rowSemantics)
    parts.push(
      profile.rowSemantics.source === 'guess'
        ? `rows probably ${profile.rowSemantics.value} (guessed${profile.rowSemantics.confidence ? `, ${profile.rowSemantics.confidence} confidence` : ''})`
        : `rows are ${profile.rowSemantics.value}`
    );

  const coordinateNames = new Set(
    profile.pointColumns ? [profile.pointColumns.longitude, profile.pointColumns.latitude] : []
  );
  let numericCount = 0;
  let timeCount = 0;
  let categoryCount = 0;
  for (const column of profile.columns) {
    if (coordinateNames.has(column.name)) continue;
    if (column.type === 'number' || column.type === 'integer') numericCount++;
    else if (column.type === 'time') timeCount++;
    else if (column.type === 'category') categoryCount++;
  }
  if (numericCount > 0) parts.push(pluralize(numericCount, 'numeric column'));
  if (timeCount > 0) parts.push(pluralize(timeCount, 'time column'));
  if (categoryCount > 0) parts.push(pluralize(categoryCount, 'category column'));

  const chips: {column: string; label: string; guessed: boolean}[] = [];
  for (const column of profile.columns) {
    if (!column.measurement) continue;
    chips.push({
      column: column.name,
      label: `${column.name}: ${column.measurement}`,
      guessed: column.measurementSource !== 'declared'
    });
  }
  return {
    headline: parts.join(' · '),
    chips,
    ...(profile.rowSemantics
      ? {
          rowSemantics: {
            value: profile.rowSemantics.value,
            guessed: profile.rowSemantics.source === 'guess',
            confidence: profile.rowSemantics.confidence
          }
        }
      : {}),
    ...(profile.coordinates
      ? {
          coordinateModel: {
            model: profile.coordinates.model,
            guessed: profile.coordinates.source === 'guess',
            confidence: profile.coordinates.confidence
          }
        }
      : {})
  };
}

// ---------------------------------------------------------------------------------------------
// Recommendation subject
// ---------------------------------------------------------------------------------------------

/**
 * The structural shape a recommender takes: the geometry kind (`features` for a dataset that mixes
 * geometry families or has a kind the catalog does not read, `table` for rows without geometry),
 * the row count and the columns with the type each is stored as. Mapping: `number` to `float64`;
 * `integer` to `int32`, or `float64` when a value does not fit 32 bits; `boolean` to `boolean`;
 * `category` and `text` to `string`; `time` to the type its values are stored as, `string` for
 * ISO text and `float64` for epoch numbers or dates. A profile made by this module records the
 * storage; for any other profile a time column maps to `string`, because the catalog is fed
 * GeoJSON, where timestamps are text.
 */
export function toKeplerRecommendationSubject(
  profile: KeplerDatasetProfile
): KeplerRecommendationSubject {
  const isMixed = profile.mixedGeometry !== undefined;
  let kind: KeplerRecommendationSubject['kind'];
  if (isMixed) kind = 'features';
  else if (profile.catalogGeometry) kind = profile.catalogGeometry;
  else
    kind =
      profile.kind === 'attributes' ? 'table' : profile.kind === 'points' ? 'point' : 'features';
  return {
    kind,
    rowCount: profile.rowCount,
    columns: profile.columns.map(column => ({name: column.name, type: getStorageType(column)}))
  };
}

function getStorageType(column: GPUAnalysisColumnProfile): KeplerProfileStorageType {
  const storage = (column as Partial<KeplerDatasetProfileColumn>).storage;
  if (storage) return storage;
  switch (column.type) {
    case 'number':
      return 'float64';
    case 'integer':
      return 'int32';
    case 'boolean':
      return 'boolean';
    default:
      return 'string';
  }
}
