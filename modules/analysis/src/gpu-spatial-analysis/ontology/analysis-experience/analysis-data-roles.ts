// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisColumnProfile,
  GPUAnalysisColumnStatistics,
  GPUAnalysisCoordinateModel,
  GPUAnalysisDataProfile,
  GPUAnalysisFieldRoleSpec,
  GPUAnalysisMeasurementLevel,
  GPUAnalysisRowSemantics
} from './analysis-experience-types';

// Pure functions that guess what a column measures, match columns to the field roles an operation
// declares, and assign columns to the roles of one input. Every guess carries a confidence and a
// reason so a host can show it and let the user correct it: values can never prove that a column
// is a count and not a rate, only the name and the range can suggest it.

/** A guessed measurement level with the sentence that explains it. */
export type GPUAnalysisMeasurementGuess = {
  level: GPUAnalysisMeasurementLevel;
  /** One sentence for the chip that shows the guess. */
  reason: string;
  /** `high`: the name or the values say it plainly. `low`: it is the likeliest reading only. */
  confidence: 'high' | 'low';
};

/** Two columns that probably form a rate: events over the population exposed to them. */
export type GPUAnalysisRatePairSuggestion = {
  numerator: string;
  denominator: string;
  reason: string;
  /** Plausibility, higher is better; pairs are returned best first. */
  score: number;
};

/** A guessed coordinate model with the sentence that explains it. */
export type GPUAnalysisCoordinateGuess = {
  model: GPUAnalysisCoordinateModel;
  reason: string;
  confidence: 'high' | 'low';
};

/** One column offered for one field role. */
export type GPUAnalysisFieldRoleMatch = {
  name: string;
  /** `preferred`: the type fits and the measurement level is one the role is meant for. */
  fit: 'preferred' | 'accepted';
  /** Present when the column fits only by type, for example `cases is a count; ...`. */
  warning?: string;
  /** Where the column's level came from; absent when no level is known. */
  measurementSource?: 'declared' | 'guess';
  /**
   * How sure the level is. A `guess` with `low` confidence is never `preferred`: the user must
   * confirm it first.
   */
  measurementConfidence?: 'high' | 'low';
};

/** The result of assigning columns to every role of one input. */
export type GPUAnalysisFieldRoleAssignment = {
  /** Column name per role id; an array for a role that takes several columns. */
  assignment: Readonly<Record<string, string | readonly string[]>>;
  /** Ids of required roles that no remaining column could fill. */
  unfilledRoles: readonly string[];
  /**
   * Ids of roles that list measurement levels and were filled by a column whose level is only a
   * low-confidence guess. A caller should ask the user to confirm these before running.
   */
  unconfirmedRoles: readonly string[];
  warnings: readonly string[];
};

// ---------------------------------------------------------------------------------------------
// Name tokens
// ---------------------------------------------------------------------------------------------

/**
 * Splits a column name into lower-case words on snake, kebab, camel and Pascal case, on acronym
 * boundaries (`PULocationID` gives `pu`, `location`, `id`) and on digit boundaries. Matching on
 * whole words keeps `accident` from matching `id` and `created` from matching `rate`.
 */
export function tokenizeGPUAnalysisColumnName(name: string): readonly string[] {
  const tokens: string[] = [];
  for (const piece of name.split(/[^\p{L}\p{N}]+/u)) {
    if (piece.length === 0) {
      continue;
    }
    const words = piece
      .replace(/(\p{Ll})(\p{Lu})/gu, '$1 $2')
      .replace(/(\p{Lu}+)(\p{Lu}\p{Ll})/gu, '$1 $2')
      .replace(/(\p{L})(\p{N})/gu, '$1 $2')
      .replace(/(\p{N})(\p{L})/gu, '$1 $2')
      .split(' ');
    for (const word of words) {
      if (word.length > 0) {
        tokens.push(word.toLowerCase());
      }
    }
  }
  return tokens;
}

type NameCategory =
  | 'identifier'
  | 'intensive'
  | 'cyclic'
  | 'ordinal'
  | 'count'
  | 'amount'
  | 'nominal'
  | 'no-level';

/**
 * Words that settle a numeric column's level, by category. A word appears in one category only
 * (a later list would silently replace an earlier one). Each list says why the words belong there.
 */
const NAME_WORDS: Readonly<Record<NameCategory, readonly string[]>> = {
  // Labels for rows or places: they carry no magnitude, so averaging or summing them is wrong.
  identifier: [
    'id',
    'ids',
    'fid',
    'oid',
    'gid',
    'pid',
    'uid',
    'uuid',
    'guid',
    'objectid',
    'geoid',
    'fips',
    'zip',
    'zipcode',
    'postal',
    'postcode',
    'code',
    'key'
  ],
  // Ratios, shares and densities: intensive, so they must not be added over areas.
  intensive: [
    'rate',
    'rates',
    'ratio',
    'ratios',
    'share',
    'shares',
    'pct',
    'percent',
    'percentage',
    'perc',
    'proportion',
    'fraction',
    'density',
    'speed',
    'velocity',
    'index',
    'per',
    'mean',
    'avg',
    'average',
    'median'
  ],
  // Angles and clock positions that wrap around.
  cyclic: [
    'bearing',
    'heading',
    'azimuth',
    'aspect',
    'direction',
    'dir',
    'wdir',
    'cog',
    'hour',
    'hod',
    'month',
    'dow',
    'weekday'
  ],
  // Ordered classes and calendar years: order matters, differences do not add up.
  ordinal: [
    'rank',
    'ranking',
    'level',
    'class',
    'grade',
    'tier',
    'rating',
    'stars',
    'severity',
    'quarter',
    'ordinal',
    'year',
    'yr',
    'decade'
  ],
  // Totals of things that exist in whole units.
  count: [
    'count',
    'counts',
    'cnt',
    'n',
    'num',
    'number',
    'total',
    'cases',
    'deaths',
    'births',
    'infections',
    'population',
    'pop',
    'residents',
    'households',
    'hh',
    'units',
    'events',
    'incidents',
    'crimes',
    'crashes',
    'arrests',
    'trips',
    'visits',
    'passengers',
    'persons',
    'people',
    'staff',
    'employees',
    'students',
    'enrollment',
    'ridership',
    'boardings',
    'vehicles',
    'bedrooms',
    'bathrooms',
    'rooms',
    'beds',
    'floors',
    'stories',
    'lanes',
    'seats',
    'injuries',
    'fatalities',
    'casualties',
    'victims'
  ],
  // Quantities that add up but are not whole units: sizes, money, time spans.
  amount: [
    'area',
    'length',
    'width',
    'perimeter',
    'len',
    'leng',
    'height',
    'depth',
    'aland',
    'awater',
    'distance',
    'duration',
    'weight',
    'mass',
    'volume',
    'acres',
    'hectares',
    'sqft',
    'sqm',
    'revenue',
    'income',
    'sales',
    'cost',
    'price',
    'fare',
    'tip',
    'tolls',
    'amount',
    'budget',
    'spend',
    'spending',
    'tax',
    'assessed',
    'rainfall',
    'precipitation',
    'precip'
  ],
  // Words for groupings. Used for numeric columns that hold category codes.
  nominal: [
    'county',
    'state',
    'region',
    'province',
    'country',
    'district',
    'borough',
    'ward',
    'zone',
    'neighborhood',
    'beat',
    'precinct',
    'route',
    'type',
    'kind',
    'category',
    'status',
    'group',
    'cluster',
    'label'
  ],
  // Coordinates and interval-scale physical quantities. Not counts, not amounts, not rates.
  'no-level': [
    'lat',
    'latitude',
    'lon',
    'lng',
    'longitude',
    'coord',
    'coords',
    'coordinate',
    'coordinates',
    'easting',
    'northing',
    'elevation',
    'elev',
    'altitude',
    'temp',
    'temperature',
    'pressure'
  ]
};

const WORD_CATEGORY: ReadonlyMap<string, NameCategory> = new Map(
  (Object.keys(NAME_WORDS) as NameCategory[]).flatMap(category =>
    NAME_WORDS[category].map((word): [string, NameCategory] => [word, category])
  )
);

/**
 * Coordinate names glued to a prefix (`INTPTLAT`, `centroidlon`). Only long tokens are tested so
 * that words such as `plat` or `salon` are not read as coordinates.
 */
const COORDINATE_SUFFIX_MINIMUM_LENGTH = 7;

function isCoordinateWord(word: string): boolean {
  if (WORD_CATEGORY.get(word) === 'no-level') {
    return true;
  }
  return (
    word.length >= COORDINATE_SUFFIX_MINIMUM_LENGTH &&
    (word.endsWith('lat') || word.endsWith('lon') || word.endsWith('lng'))
  );
}

/** Words before `area` that make it a numbered region, not a size (`community_area`). */
const REGION_QUALIFIER_WORDS: readonly string[] = [
  'community',
  'service',
  'planning',
  'study',
  'catchment',
  'market',
  'census',
  'statistical'
];

/**
 * Words that say a number is a point in time stored as a number (`unix_time`, `date_int`). A
 * time has no level: summing or averaging it means nothing. A name with one of these words
 * and values in an epoch range, or a name that ends in an unambiguous one, is time-like.
 */
const TIME_WORDS: readonly string[] = [
  'date',
  'time',
  'timestamp',
  'datetime',
  'epoch',
  'ts',
  'unix',
  'created',
  'updated',
  'modified'
];

/** Time words that stay time-like even without epoch values (`time` alone may be a duration). */
const UNAMBIGUOUS_TIME_WORDS: readonly string[] = ['date', 'timestamp', 'datetime', 'epoch', 'ts'];

/** Epoch seconds from 1973 to 2286, epoch milliseconds likewise, and yyyymmdd dates. */
const TIME_VALUE_BANDS: readonly (readonly [number, number])[] = [
  [1e8, 1e10],
  [1e11, 1e13],
  [19000101, 21001231]
];

function hasTimeLikeValues(facts: NumericFacts): boolean {
  const {minimum, maximum} = facts;
  if (minimum === undefined || maximum === undefined) {
    return false;
  }
  return TIME_VALUE_BANDS.some(([low, high]) => minimum >= low && maximum <= high);
}

/**
 * Registries and administrative units whose numbers are codes (parcel, tract, MMSI), so they are
 * identifiers when they end a name, or when nothing else but short suffixes such as `ce` is in it.
 */
const REGISTRY_IDENTIFIER_WORDS: readonly string[] = [
  'apn',
  'mmsi',
  'imo',
  'icao',
  'vin',
  'isbn',
  'parcel',
  'puma',
  'zcta',
  'tract',
  'tractce',
  'block',
  'blockgroup',
  'uprn'
];

/** Words that make a trailing `number`, `num` or `no` a registry number (`order_number`). */
const REGISTRY_NUMBER_WORDS: readonly string[] = [
  'order',
  'invoice',
  'account',
  'serial',
  'ticket',
  'case',
  'license',
  'permit',
  'phone',
  'policy',
  'item',
  'customer',
  'flight',
  'lot',
  'claim'
];

/** Length at which a token ending in `fp` is read as a FIPS code (`STATEFP`, `COUNTYFP`). */
const FIPS_SUFFIX_MINIMUM_LENGTH = 5;

/** Longest suffix token (`ce`, `10`) allowed next to registry words in an all-registry name. */
const REGISTRY_SUFFIX_MAXIMUM_LENGTH = 2;

function isRegistryIdentifierName(tokens: readonly string[], last: string): boolean {
  if (REGISTRY_IDENTIFIER_WORDS.includes(last)) {
    return true;
  }
  const lastIndex = tokens.lastIndexOf(last);
  const previous = tokens[lastIndex - 1];
  if (
    ['no', 'num', 'number'].includes(last) &&
    previous !== undefined &&
    (REGISTRY_NUMBER_WORDS.includes(previous) || REGISTRY_IDENTIFIER_WORDS.includes(previous))
  ) {
    return true;
  }
  if (tokens.some(word => word.length >= FIPS_SUFFIX_MINIMUM_LENGTH && word.endsWith('fp'))) {
    return true;
  }
  const words = tokens.filter(word => !/^\p{N}+$/u.test(word));
  return (
    words.some(word => REGISTRY_IDENTIFIER_WORDS.includes(word)) &&
    words.every(
      word =>
        REGISTRY_IDENTIFIER_WORDS.includes(word) || word.length <= REGISTRY_SUFFIX_MAXIMUM_LENGTH
    )
  );
}

/**
 * Last words of measured quantities that are neither additive nor ratios: an age or a magnitude
 * (log scale) has no level, as with temperature. `median_age` is still a rate, by the ratio rule.
 */
const MEASURED_QUANTITY_LAST_WORDS: readonly string[] = ['age', 'magnitude'];

/**
 * Unit suffixes. A column ending in a unit holds a measured size or mass, which adds up (a total
 * length of roads, a catch in kilograms), and is not a share even when its values are small.
 */
const UNIT_SUFFIX_WORDS: readonly string[] = [
  'cm',
  'mm',
  'km',
  'm',
  'ft',
  'mi',
  'kg',
  'lb',
  'lbs',
  'ha',
  'm2',
  'km2',
  'sqkm',
  'sqmi',
  'tonnes',
  'tons',
  'litres',
  'liters',
  'gal'
];

/** Amount words whose columns may hold per-person values that do not add up. */
const AMBIGUOUS_AMOUNT_WORDS: readonly string[] = ['income', 'price'];

/** Names that say nothing about the content; the values are not read for them either. */
const GENERIC_NAME_WORDS: readonly string[] = [
  'value',
  'values',
  'val',
  'data',
  'field',
  'var',
  'variable',
  'attribute',
  'attr',
  'measure',
  'metric'
];

/** Single letters `x`, `y` and `z` name coordinates only as the last word of a name. */
const COORDINATE_LETTERS: readonly string[] = ['x', 'y', 'z'];

/** Words that make a name an intensive quantity wherever they stand (`median_income`). */
const INTENSIVE_ANYWHERE: readonly string[] = [
  'per',
  'mean',
  'avg',
  'average',
  'median',
  'pct',
  'percent',
  'percentage',
  'ratio',
  'proportion',
  'density',
  'share',
  'rate'
];

/** Count words too generic to be trusted without integer values (`total`, `n`). */
const WEAK_COUNT_WORDS: readonly string[] = ['n', 'num', 'number', 'total'];

/** Whole-name phrases that hold a clock or calendar position although their last word does not. */
const CYCLIC_PHRASES: readonly string[] = ['dayofweek', 'hourofday', 'monthofyear', 'dayofyear'];

type CyclicUnit = 'angle' | 'hour' | 'month' | 'weekday';

const CYCLIC_UNIT_OF_WORD: Readonly<Record<string, CyclicUnit>> = {
  bearing: 'angle',
  heading: 'angle',
  azimuth: 'angle',
  aspect: 'angle',
  direction: 'angle',
  dir: 'angle',
  wdir: 'angle',
  cog: 'angle',
  hour: 'hour',
  hod: 'hour',
  month: 'month',
  dow: 'weekday',
  weekday: 'weekday'
};

/** Largest magnitude an angle in degrees may have and still read as a bearing. */
const ANGLE_LIMIT_DEGREES = 360;
/** Hours of day run 0 to 23, or 1 to 24 in some systems. */
const HOUR_LIMIT = 24;
/** Months run 0 to 11 or 1 to 12. */
const MONTH_LIMIT = 12;
/** Days of week run 0 to 6 or 1 to 7. */
const WEEKDAY_LIMIT = 7;

// ---------------------------------------------------------------------------------------------
// Measurement level guess
// ---------------------------------------------------------------------------------------------

/** Distinct values at or below which an integer column with no name evidence reads as ordinal. */
const ORDINAL_DISTINCT_LIMIT = 10;
/** Share of rows an estimated distinct count must reach to count as unique. */
const ESTIMATED_UNIQUE_SHARE = 0.95;
/** Upper end of a percentage; values in (1, 100] with fractions read as percentages. */
const PERCENT_MAXIMUM = 100;

type NumericFacts = {
  nonNullCount: number;
  minimum: number | undefined;
  maximum: number | undefined;
  integral: boolean;
  /** True only when the statistics say the column has non-whole values. */
  fractional: boolean;
  distinctCount: number | undefined;
  distinctIsEstimate: boolean;
};

function readNumericFacts(statistics: GPUAnalysisColumnStatistics): NumericFacts {
  const rowsBehindStatistics = statistics.sampleSize ?? statistics.rowCount;
  return {
    nonNullCount: Math.max(rowsBehindStatistics - statistics.nullCount, 0),
    minimum: Number.isFinite(statistics.minimum) ? statistics.minimum : undefined,
    maximum: Number.isFinite(statistics.maximum) ? statistics.maximum : undefined,
    integral: statistics.type === 'integer' || statistics.integral === true,
    fractional: statistics.type === 'number' && statistics.integral === false,
    distinctCount: statistics.distinctCount,
    distinctIsEstimate: statistics.distinctCountIsEstimate === true
  };
}

function lastWord(tokens: readonly string[]): string | undefined {
  for (let index = tokens.length - 1; index >= 0; index--) {
    if (!/^\p{N}+$/u.test(tokens[index])) {
      return tokens[index];
    }
  }
  return undefined;
}

function isUniqueColumn(facts: NumericFacts): boolean {
  if (facts.distinctCount === undefined || facts.nonNullCount < 2) {
    return false;
  }
  return facts.distinctIsEstimate
    ? facts.distinctCount >= facts.nonNullCount * ESTIMATED_UNIQUE_SHARE
    : facts.distinctCount === facts.nonNullCount;
}

function describeRange(facts: NumericFacts): string {
  if (facts.minimum === undefined || facts.maximum === undefined) {
    return 'values';
  }
  return `values from ${facts.minimum} to ${facts.maximum}`;
}

function checkCyclicRange(unit: CyclicUnit, facts: NumericFacts): 'fits' | 'unknown' | 'violates' {
  if (facts.minimum === undefined || facts.maximum === undefined) {
    return 'unknown';
  }
  switch (unit) {
    case 'angle':
      return facts.minimum >= -ANGLE_LIMIT_DEGREES && facts.maximum <= ANGLE_LIMIT_DEGREES
        ? 'fits'
        : 'violates';
    case 'hour':
      return facts.minimum >= 0 && facts.maximum <= HOUR_LIMIT ? 'fits' : 'violates';
    case 'month':
      return facts.minimum >= 0 && facts.maximum <= MONTH_LIMIT ? 'fits' : 'violates';
    case 'weekday':
      return facts.minimum >= 0 && facts.maximum <= WEEKDAY_LIMIT ? 'fits' : 'violates';
  }
}

function guessCyclicLevel(
  name: string,
  tokens: readonly string[],
  facts: NumericFacts
): GPUAnalysisMeasurementGuess | undefined {
  const last = lastWord(tokens);
  const joined = tokens.join('');
  let unit: CyclicUnit | undefined =
    last !== undefined && Object.hasOwn(CYCLIC_UNIT_OF_WORD, last)
      ? CYCLIC_UNIT_OF_WORD[last]
      : undefined;
  if (unit === undefined) {
    const phrase = CYCLIC_PHRASES.find(candidate => joined.includes(candidate));
    if (phrase === 'dayofweek') {
      unit = 'weekday';
    } else if (phrase === 'hourofday') {
      unit = 'hour';
    } else if (phrase === 'monthofyear') {
      unit = 'month';
    } else if (phrase === 'dayofyear') {
      return undefined;
    }
  }
  if (unit === undefined) {
    return undefined;
  }
  const fit = checkCyclicRange(unit, facts);
  if (fit === 'violates') {
    return undefined;
  }
  return fit === 'fits'
    ? {
        level: 'cyclic',
        reason: `${name} is named like a ${unit} and holds ${describeRange(facts)}, which wrap around, so 0 and the top of the range are neighbours.`,
        confidence: 'high'
      }
    : {
        level: 'cyclic',
        reason: `${name} is named like a ${unit}, which wraps around; no range was computed to confirm it.`,
        confidence: 'low'
      };
}

function guessCountLevel(
  name: string,
  word: string,
  facts: NumericFacts
): GPUAnalysisMeasurementGuess {
  const weak = WEAK_COUNT_WORDS.includes(word);
  const isNegative = facts.minimum !== undefined && facts.minimum < 0;
  if (facts.fractional && weak && !isNegative) {
    return {
      level: 'amount',
      reason: `${name} is named "${word}" but holds fractional values, so it reads as an amount that adds up, not a count of whole things.`,
      confidence: 'low'
    };
  }
  if (isNegative || facts.fractional) {
    return {
      level: 'count',
      reason: `${name} is named "${word}", like a count, but its values (${describeRange(facts)}) are not whole and non-negative; confirm it.`,
      confidence: 'low'
    };
  }
  return {
    level: 'count',
    reason: `${name} is named "${word}", like a count of things that adds up over areas${facts.integral ? ', and holds whole numbers' : ''}.`,
    confidence: weak && !facts.integral ? 'low' : 'high'
  };
}

function guessAmountLevel(
  name: string,
  word: string,
  facts: NumericFacts
): GPUAnalysisMeasurementGuess {
  const isNegative = facts.minimum !== undefined && facts.minimum < 0;
  if (AMBIGUOUS_AMOUNT_WORDS.includes(word)) {
    return {
      level: 'amount',
      reason: `${name} is named "${word}", which may be a total or a per-unit figure that does not add up over areas; confirm which.`,
      confidence: 'low'
    };
  }
  return {
    level: 'amount',
    reason: isNegative
      ? `${name} is named "${word}", like an amount that adds up, but it has negative values (${describeRange(facts)}); confirm it.`
      : `${name} is named "${word}", like a quantity that adds up over areas.`,
    confidence: isNegative ? 'low' : 'high'
  };
}

/** The level a word gives on its own, with the values taken into account. */
function guessLevelFromWord(
  name: string,
  word: string,
  facts: NumericFacts
): GPUAnalysisMeasurementGuess | undefined {
  const category = WORD_CATEGORY.get(word);
  switch (category) {
    case 'identifier':
      return {
        level: 'identifier',
        reason: `${name} is named "${word}", which labels rows or places; a code such as a zip or FIPS number is not an amount even when it is stored as a number.`,
        confidence: 'high'
      };
    case 'intensive':
      if (word === 'index' && facts.integral) {
        // An integer "index" is usually a row number or a position, not a ratio.
        return undefined;
      }
      return {
        level: 'rate',
        reason: `${name} is named "${word}", like a ratio, share or density that does not add up over areas.`,
        confidence: 'high'
      };
    case 'ordinal':
      return {
        level: 'ordinal',
        reason: `${name} is named "${word}", which orders rows or labels a year; differences and sums of it carry no amount.`,
        confidence: 'high'
      };
    case 'count':
      return guessCountLevel(name, word, facts);
    case 'amount':
      return guessAmountLevel(name, word, facts);
    case 'nominal':
      return {
        level: 'nominal',
        reason: `${name} is named "${word}" and holds numbers, which are most likely category codes.`,
        confidence: 'low'
      };
    default:
      return undefined;
  }
}

/** Order in which words anywhere in a name are tried, after the last word. */
const ANY_WORD_CATEGORY_ORDER: readonly NameCategory[] = [
  'intensive',
  'ordinal',
  'amount',
  'count',
  'identifier',
  'nominal'
];

function guessNumericLevelFromName(
  name: string,
  tokens: readonly string[],
  facts: NumericFacts
): GPUAnalysisMeasurementGuess | 'no-level' | undefined {
  const last = lastWord(tokens);
  if (last === undefined) {
    return undefined;
  }
  // 1. Coordinates and physical quantities have no level at all.
  if (tokens.some(word => isCoordinateWord(word)) || COORDINATE_LETTERS.includes(last)) {
    return 'no-level';
  }
  // 1b. A number that holds a date or a time stamp is a time, which has no level.
  if (
    tokens.some(word => TIME_WORDS.includes(word)) &&
    (UNAMBIGUOUS_TIME_WORDS.includes(last) || hasTimeLikeValues(facts))
  ) {
    return 'no-level';
  }
  // 2a. A code qualified by an ordinal word (`severity_code`) is a class, not a label.
  if (last === 'code' && tokens.some(word => WORD_CATEGORY.get(word) === 'ordinal')) {
    return {
      level: 'ordinal',
      reason: `${name} is a code for an ordered class (${tokens.filter(word => WORD_CATEGORY.get(word) === 'ordinal')[0]}); its order matters and its differences do not add up.`,
      confidence: 'high'
    };
  }
  // 2. A name that ends in an identifier word is an identifier, whatever comes before it.
  if (WORD_CATEGORY.get(last) === 'identifier') {
    return guessLevelFromWord(name, last, facts);
  }
  // 2c. Registry and administrative unit numbers (parcel, tract, MMSI, FIPS parts) are codes.
  if (isRegistryIdentifierName(tokens, last)) {
    return {
      level: 'identifier',
      reason: `${name} is named like a registry or administrative unit number, which labels rows or places; its digits carry no magnitude.`,
      confidence: 'high'
    };
  }
  // 2b. `community_area` and `service_area` number regions; the size is `area` alone.
  if (last === 'area' && REGION_QUALIFIER_WORDS.includes(tokens[tokens.length - 2] ?? '')) {
    return {
      level: 'nominal',
      reason: `${name} is named like a numbered region ("${tokens[tokens.length - 2]} area"), so its numbers most likely label regions and are not sizes.`,
      confidence: 'low'
    };
  }
  // 3. Clock and angle names, when the values fit.
  const cyclic = guessCyclicLevel(name, tokens, facts);
  if (cyclic) {
    return cyclic;
  }
  // 4. A ratio word anywhere makes the whole name a ratio: `median_income`, `cases_per_100k`.
  const intensiveWord = tokens.find(word => INTENSIVE_ANYWHERE.includes(word));
  if (intensiveWord !== undefined) {
    return guessLevelFromWord(name, intensiveWord, facts);
  }
  // A unit price or cost is per item even without an explicit `per` token.
  if (tokens.includes('unit') && (tokens.includes('price') || tokens.includes('cost'))) {
    return {
      level: 'rate',
      reason: `${name} names a price or cost per unit, which does not add up over areas.`,
      confidence: 'high'
    };
  }
  // 4b. An age or a magnitude is a measured quantity with no additive meaning.
  if (MEASURED_QUANTITY_LAST_WORDS.includes(last)) {
    return 'no-level';
  }
  // 5. The last word is the head of the name: `population_rank` is a rank.
  const fromLast = guessLevelFromWord(name, last, facts);
  if (fromLast) {
    return fromLast;
  }
  // 5b. A unit suffix makes a measured size or mass.
  if (
    UNIT_SUFFIX_WORDS.includes(last) &&
    tokens.length > 1 &&
    !tokens.some(word => WORD_CATEGORY.get(word) === 'intensive')
  ) {
    return {
      level: 'amount',
      reason: `${name} ends in the unit "${last}", so it holds a measured size or mass that adds up, not a share, even when its values are small.`,
      confidence: facts.minimum !== undefined && facts.minimum < 0 ? 'low' : 'high'
    };
  }
  // 6. Any other word, strongest category first.
  for (const category of ANY_WORD_CATEGORY_ORDER) {
    for (const word of tokens) {
      if (WORD_CATEGORY.get(word) === category) {
        const guess = guessLevelFromWord(name, word, facts);
        if (guess) {
          return guess;
        }
      }
    }
  }
  // 7. A name that says nothing (`value`) leaves the values unread.
  if (GENERIC_NAME_WORDS.includes(last)) {
    return 'no-level';
  }
  return undefined;
}

function guessNumericLevelFromValues(
  name: string,
  facts: NumericFacts
): GPUAnalysisMeasurementGuess | undefined {
  if (facts.nonNullCount === 0) {
    return undefined;
  }
  if (facts.integral) {
    if (facts.distinctCount === 1) {
      return undefined;
    }
    if (isUniqueColumn(facts)) {
      const contiguous =
        facts.minimum !== undefined &&
        facts.maximum !== undefined &&
        facts.maximum - facts.minimum + 1 === facts.nonNullCount;
      return {
        level: 'identifier',
        reason: contiguous
          ? `Every value of ${name} differs and they run without gaps from ${facts.minimum} to ${facts.maximum}, like a row number or identifier.`
          : `Every value of ${name} differs, so it most likely labels rows; if it is a measurement, such as a population, correct it.`,
        confidence: 'low'
      };
    }
    if (facts.distinctCount === 2) {
      return {
        level: 'nominal',
        reason: `${name} holds only two whole-number values, like a flag.`,
        confidence: 'low'
      };
    }
    if (facts.distinctCount !== undefined && facts.distinctCount <= ORDINAL_DISTINCT_LIMIT) {
      return {
        level: 'ordinal',
        reason: `${name} holds ${facts.distinctCount} different whole numbers, like classes or levels; it could also be a small count.`,
        confidence: 'low'
      };
    }
    if (facts.minimum !== undefined && facts.minimum >= 0) {
      return {
        level: 'count',
        reason: `${name} holds non-negative whole numbers, which most likely count things; values alone cannot tell a count from a code, so confirm it.`,
        confidence: 'low'
      };
    }
    return undefined;
  }
  if (facts.minimum === undefined || facts.maximum === undefined || facts.minimum < 0) {
    return undefined;
  }
  if (facts.maximum <= 1) {
    return {
      level: 'rate',
      reason: `${name} holds fractions between ${facts.minimum} and ${facts.maximum}, like a share or ratio; a measure that happens to be small would read the same.`,
      confidence: 'low'
    };
  }
  if (facts.maximum <= PERCENT_MAXIMUM) {
    return {
      level: 'rate',
      reason: `${name} holds fractions between ${facts.minimum} and ${facts.maximum}, like a percentage; a measure that happens to be small would read the same.`,
      confidence: 'low'
    };
  }
  return {
    level: 'amount',
    reason: `${name} holds non-negative fractions up to ${facts.maximum}, which most likely measure a size or quantity; values alone cannot tell it from a rate.`,
    confidence: 'low'
  };
}

function guessCategoryLevel(
  statistics: GPUAnalysisColumnStatistics,
  tokens: readonly string[]
): GPUAnalysisMeasurementGuess | undefined {
  const {name} = statistics;
  const facts = readNumericFacts(statistics);
  if (facts.nonNullCount === 0) {
    return undefined;
  }
  if (statistics.type === 'boolean') {
    return {
      level: 'nominal',
      reason: `${name} holds true or false values, which name two groups.`,
      confidence: 'high'
    };
  }
  const identifierWord = tokens.find(word => WORD_CATEGORY.get(word) === 'identifier');
  if (identifierWord !== undefined) {
    return {
      level: 'identifier',
      reason: `${name} is named "${identifierWord}", which labels rows or places.`,
      confidence: 'high'
    };
  }
  if (isUniqueColumn(facts)) {
    return {
      level: 'identifier',
      reason: `Every value of ${name} differs, so it labels rows.`,
      confidence: 'low'
    };
  }
  return {
    level: 'nominal',
    reason:
      statistics.type === 'category'
        ? `${name} holds categories.`
        : `${name} holds text that repeats between rows, which most likely names groups.`,
    confidence: statistics.type === 'category' ? 'high' : 'low'
  };
}

/**
 * Guesses what a column measures from its type, name, range, integrality, distinct count against
 * row count and null count. Returns undefined for time columns, for columns that are all null,
 * and where nothing is evident (a negative non-integer column named `tmax`).
 *
 * Rules, applied in this order for number and integer columns. Each returns a reason sentence.
 * 1. Names of coordinates and physical quantities (`lat`, `INTPTLAT`, `elevation`, `temperature`, a
 *    last word `x`) give no level: undefined. So do time stamps stored as numbers (a time word with
 *    epoch or yyyymmdd values, or a last word `date`, `timestamp`), a last word `age` or
 *    `magnitude` (measured, neither additive nor a ratio; `median_age` is still a rate), and a
 *    generic last word such as `value` or `data` once no other rule fired.
 * 2. A code qualified by an ordinal word (`severity_code`) is ordinal. A last word such as `id`,
 *    `fid`, `geoid`, `fips`, `zip`, `code` is an identifier, and so is a registry or administrative
 *    unit number: last word `apn`, `mmsi`, `parcel`, `puma`, `tract`, `block`, a trailing `no` or
 *    `number` after `order`, `invoice`, `parcel` and the like, a token ending in `fp`
 *    (`STATEFP`), or a name made only of registry words and short suffixes (`ZCTA5CE10`).
 *    `community_area` and similar numbered regions are nominal.
 * 3. A last word such as `bearing`, `heading`, `aspect`, `hour`, `month`, `dow` whose range fits
 *    (angles within [-360, 360], hours within [0, 24], months within [0, 12], weekdays within
 *    [0, 7]): cyclic. Low confidence when no range is known.
 * 4. A word such as `rate`, `ratio`, `share`, `pct`, `percent`, `per`, `density`, `mean`, `avg`,
 *    `median` anywhere: rate (`median_income`, `cases_per_100k`, `pct_change` even when negative).
 * 5. The last word, then a unit suffix (`_cm`, `_kg`, `_km`, `_m`, `_ft`: an amount, because a
 *    measured size adds up even when its values are small), then any word, from these
 *    categories, strongest first: `speed` or a non-integer `index` (rate); `year`, `rank`,
 *    `level`, `class`, `severity` (ordinal); `area`, `length`, `depth`, `revenue`, `distance`
 *    (amount; `income` is low confidence because it may be per person); `count`, `total`,
 *    `cases`, `population`, `bedrooms`, `employees` (count); a zone, `beat`, `precinct` or `type`
 *    word on numbers (nominal, low). `total` with fractional values reads as an amount, low.
 * 6. With no name evidence, the values, always low confidence: all distinct integers is an
 *    identifier; two integer values is nominal; at most 10 integer values is ordinal; other
 *    non-negative integers are a count; fractions within [0, 1] or (1, 100] are a rate; other
 *    non-negative fractions are an amount; negative values give undefined.
 *
 * For category and text columns: an identifier word anywhere, or all values distinct, gives
 * identifier; otherwise nominal. Boolean columns are nominal.
 *
 * With `sampleSize` set, row counts are taken from the sample. The confidence and the reason exist
 * so that a host can show the guess and let the user correct it.
 */
export function guessGPUAnalysisMeasurementLevel(
  statistics: GPUAnalysisColumnStatistics
): GPUAnalysisMeasurementGuess | undefined {
  if (statistics.type === 'time') {
    return undefined;
  }
  const tokens = tokenizeGPUAnalysisColumnName(statistics.name);
  if (
    statistics.type === 'category' ||
    statistics.type === 'text' ||
    statistics.type === 'boolean'
  ) {
    return guessCategoryLevel(statistics, tokens);
  }
  const facts = readNumericFacts(statistics);
  if (facts.nonNullCount === 0) {
    return undefined;
  }
  const fromName = guessNumericLevelFromName(statistics.name, tokens, facts);
  if (fromName === 'no-level') {
    return undefined;
  }
  if (fromName) {
    return fromName;
  }
  return guessNumericLevelFromValues(statistics.name, facts);
}

// ---------------------------------------------------------------------------------------------
// Rate pairs
// ---------------------------------------------------------------------------------------------

/** Numerator words with their strength: words that name events of interest score highest. */
const NUMERATOR_WORD_SCORES: Readonly<Record<string, number>> = {
  cases: 3,
  deaths: 3,
  incidents: 3,
  events: 3,
  crimes: 3,
  births: 3,
  infections: 3,
  positives: 3,
  crashes: 3,
  accidents: 3,
  arrests: 3,
  injuries: 2.5,
  fatalities: 2.5,
  casualties: 2.5,
  victims: 2.5,
  trips: 1.5,
  visits: 1.5
};

/** Denominator words with their strength: `population` is the plainest exposure. */
const DENOMINATOR_WORD_SCORES: Readonly<Record<string, number>> = {
  population: 3,
  pop: 3,
  residents: 2.5,
  inhabitants: 2.5,
  persons: 2.5,
  people: 2.5,
  households: 2,
  hh: 2,
  total: 1
};

// Only explicitly named proportions imply that the numerator is a subset of the denominator.
const PROPORTION_WORDS: ReadonlySet<string> = new Set([
  'proportion',
  'share',
  'fraction',
  'percent',
  'percentage',
  'pct'
]);

/** Mean ratio from which a denominator is clearly an exposure (10 times) or larger (2 times). */
const CLEAR_EXPOSURE_RATIO = 10;
const LARGER_EXPOSURE_RATIO = 2;
/** Score added when numerator and denominator end in the same suffix (`cases_2020`, `pop_2020`). */
const SHARED_SUFFIX_SCORE = 0.5;

function getKnownLevel(column: GPUAnalysisColumnProfile): GPUAnalysisMeasurementLevel | undefined {
  return column.measurement ?? guessGPUAnalysisMeasurementLevel(column)?.level;
}

/** A column counts only when its level is a count and is declared or guessed with confidence. */
function isConfidentCount(column: GPUAnalysisColumnProfile): boolean {
  if (column.type !== 'integer' && column.type !== 'number') {
    return false;
  }
  if (getKnownLevel(column) !== 'count') {
    return false;
  }
  return !isUnconfirmedGuess(describeColumnLevel(column));
}

function formatForReading(value: number): string {
  return Number.isInteger(value)
    ? value.toLocaleString('en-US', {maximumFractionDigits: 0})
    : value.toLocaleString('en-US', {maximumSignificantDigits: 4});
}

/** Suffix words after the name word, with digits kept (`2020` of `cases_2020`). */
function getTrailingDigits(tokens: readonly string[]): string | undefined {
  const last = tokens[tokens.length - 1];
  return last !== undefined && /^\p{N}+$/u.test(last) ? last : undefined;
}

/**
 * Finds likely cases-over-population pairs and ranks them, best first: the first element is the
 * best pair, and callers that take one should take the first.
 *
 * A pair needs, on both sides, a column that reads as a count and is declared or guessed with
 * high confidence (never a rate, an identifier, or a low-confidence guess), a numerator whose last
 * word is one of `cases`, `deaths`, `incidents`, `events`, `crimes`, `births`, ... and a
 * denominator whose last word is one of `population`, `pop`, `residents`, `persons`,
 * `households`, `total`, ... Relative size ranks the pairs; repeated-event rates can exceed one.
 * Only names explicitly describing proportions require a denominator at least as large in mean
 * and maximum. Columns without these statistics are not paired.
 *
 * Score (higher is more plausible): the numerator word's strength (1.5 to 3) plus the
 * denominator word's strength (1 to 3), plus 1 when the denominator's mean is at least 10 times
 * the numerator's (0.5 when at least 2 times), plus 0.5 when both names end in the same digits.
 * Ties keep column order. The score is on each result.
 */
export function suggestGPUAnalysisRatePairs(
  columns: readonly GPUAnalysisColumnProfile[]
): readonly GPUAnalysisRatePairSuggestion[] {
  const suggestions: (GPUAnalysisRatePairSuggestion & {order: number})[] = [];
  for (const numerator of columns) {
    const numeratorTokens = tokenizeGPUAnalysisColumnName(numerator.name);
    const numeratorLast = lastWord(numeratorTokens);
    const numeratorScore =
      numeratorLast === undefined ? undefined : NUMERATOR_WORD_SCORES[numeratorLast];
    if (numeratorScore === undefined || !isConfidentCount(numerator)) {
      continue;
    }
    for (const denominator of columns) {
      const denominatorTokens = tokenizeGPUAnalysisColumnName(denominator.name);
      const denominatorLast = lastWord(denominatorTokens);
      const denominatorScore =
        denominatorLast === undefined ? undefined : DENOMINATOR_WORD_SCORES[denominatorLast];
      if (
        denominator.name === numerator.name ||
        denominatorScore === undefined ||
        !isConfidentCount(denominator)
      ) {
        continue;
      }
      const numeratorTypical = numerator.mean ?? numerator.maximum;
      const denominatorTypical = denominator.mean ?? denominator.maximum;
      if (
        numeratorTypical === undefined ||
        denominatorTypical === undefined ||
        denominator.maximum === undefined ||
        numerator.maximum === undefined ||
        !Number.isFinite(numeratorTypical) ||
        !Number.isFinite(denominatorTypical) ||
        (numeratorTokens.some(word => PROPORTION_WORDS.has(word)) &&
          (denominatorTypical < numeratorTypical || denominator.maximum < numerator.maximum))
      ) {
        continue;
      }
      const statistic =
        numerator.mean !== undefined && denominator.mean !== undefined ? 'mean' : 'maximum';
      const numeratorValue = statistic === 'mean' ? numerator.mean : numerator.maximum;
      const denominatorValue = statistic === 'mean' ? denominator.mean : denominator.maximum;
      const ratio =
        numeratorValue !== undefined && denominatorValue !== undefined && numeratorValue > 0
          ? denominatorValue / numeratorValue
          : Number.POSITIVE_INFINITY;
      const sharedDigits = getTrailingDigits(numeratorTokens);
      const sharedSuffix =
        sharedDigits !== undefined && sharedDigits === getTrailingDigits(denominatorTokens);
      const score =
        numeratorScore +
        denominatorScore +
        (ratio >= CLEAR_EXPOSURE_RATIO ? 1 : ratio >= LARGER_EXPOSURE_RATIO ? 0.5 : 0) +
        (sharedSuffix ? SHARED_SUFFIX_SCORE : 0);
      suggestions.push({
        numerator: numerator.name,
        denominator: denominator.name,
        score,
        order: suggestions.length,
        reason: `${numerator.name} and ${denominator.name} both read as counts (${statistic} ${formatForReading(denominatorValue ?? 0)} against ${formatForReading(numeratorValue ?? 0)}), so ${numerator.name} per ${denominator.name} is a likely rate.`
      });
    }
  }
  suggestions.sort((left, right) => right.score - left.score || left.order - right.order);
  return suggestions.map(({order: _order, ...suggestion}) => suggestion);
}

/**
 * The best rate pair of `suggestGPUAnalysisRatePairs`, or undefined when none qualifies. A host
 * that fills a numerator role and a denominator role should use this, then show it as a guess.
 */
export function assignGPUAnalysisRatePair(
  columns: readonly GPUAnalysisColumnProfile[]
): GPUAnalysisRatePairSuggestion | undefined {
  return suggestGPUAnalysisRatePairs(columns)[0];
}

// ---------------------------------------------------------------------------------------------
// Coordinate model
// ---------------------------------------------------------------------------------------------

/** Longitude limits of a geographic extent, in degrees. */
const LONGITUDE_LIMIT = 180;
/** Latitude limits of a geographic extent, in degrees. */
const LATITUDE_LIMIT = 90;
/** Longitudes written 0 to 360 (Pacific-centred data) end here. */
const LONGITUDE_WRAPPED_LIMIT = 360;
/** An extent this close to (0, 0) in every coordinate could be local planar metres. */
const NEAR_ORIGIN_LIMIT = 1;

/**
 * Guesses whether an extent [minimumX, minimumY, maximumX, maximumY] is in degrees or in a planar
 * unit. This is only a guess from the numbers: an extent inside [-180, 180] x [-90, 90] reads as
 * geographic, but projected metres or local site coordinates can fall inside it too, and
 * unprojected data can fall outside it. Hosts must store the returned `confidence` on
 * `profile.coordinates.confidence` (with `source: 'guess'`) so that a low-confidence guess is
 * shown and confirmed before metric work. Confidence is high outside that box (it cannot be
 * degrees) and inside it for ordinary extents; it is low when every coordinate lies within one
 * unit of the origin or the extent starts exactly at (0, 0), where local planar coordinates are
 * common, and for longitudes written 0 to 360. Distances need the right answer, so an analysis that
 * depends on metres should ask for a `declared-coordinates` confirmation instead of trusting this.
 */
export function guessGPUAnalysisCoordinateModel(
  extent: readonly [number, number, number, number]
): GPUAnalysisCoordinateGuess {
  if (extent.length !== 4 || !extent.every(value => Number.isFinite(value))) {
    throw new Error(
      'guessGPUAnalysisCoordinateModel: extent must be four finite numbers [minimumX, minimumY, maximumX, maximumY]'
    );
  }
  const [minimumX, minimumY, maximumX, maximumY] = extent;
  const withinLatitude = minimumY >= -LATITUDE_LIMIT && maximumY <= LATITUDE_LIMIT;
  if (withinLatitude && minimumX >= -LONGITUDE_LIMIT && maximumX <= LONGITUDE_LIMIT) {
    const largest = Math.max(
      Math.abs(minimumX),
      Math.abs(minimumY),
      Math.abs(maximumX),
      Math.abs(maximumY)
    );
    if (largest <= NEAR_ORIGIN_LIMIT) {
      return {
        model: 'geographic',
        reason:
          'The extent fits within degrees but lies within one unit of (0, 0), where local planar metres are as likely; confirm the coordinate system.',
        confidence: 'low'
      };
    }
    if (minimumX === 0 && minimumY === 0) {
      return {
        model: 'geographic',
        reason:
          'The extent fits within degrees but starts exactly at (0, 0), as local planar or image coordinates do; confirm the coordinate system.',
        confidence: 'low'
      };
    }
    return {
      model: 'geographic',
      reason:
        'The extent lies within longitude -180 to 180 and latitude -90 to 90, so the coordinates read as degrees.',
      confidence: 'high'
    };
  }
  if (withinLatitude && minimumX >= 0 && maximumX <= LONGITUDE_WRAPPED_LIMIT) {
    return {
      model: 'geographic',
      reason:
        'Longitudes run from 0 to 360 and latitudes stay within -90 to 90, which reads as degrees with Pacific-centred longitudes; confirm the coordinate system.',
      confidence: 'low'
    };
  }
  return {
    model: 'planar',
    reason:
      'The extent lies outside longitude -180 to 180 and latitude -90 to 90, so the coordinates cannot be degrees and read as a planar unit such as metres.',
    confidence: 'high'
  };
}

// ---------------------------------------------------------------------------------------------
// Field roles
// ---------------------------------------------------------------------------------------------

type ColumnLevelInfo = {
  level: GPUAnalysisMeasurementLevel | undefined;
  source: 'declared' | 'guess';
  confidence: 'high' | 'low';
};

function describeColumnLevel(column: GPUAnalysisColumnProfile): ColumnLevelInfo {
  if (column.measurement !== undefined) {
    const source = column.measurementSource ?? 'declared';
    return {
      level: column.measurement,
      source,
      confidence: column.measurementConfidence ?? (source === 'declared' ? 'high' : 'low')
    };
  }
  const guess = guessGPUAnalysisMeasurementLevel(column);
  return {level: guess?.level, source: 'guess', confidence: guess?.confidence ?? 'low'};
}

/** True when the level is a guess that the profile does not vouch for. */
function isUnconfirmedGuess(info: ColumnLevelInfo): boolean {
  return info.level !== undefined && info.source === 'guess' && info.confidence === 'low';
}

const LEVEL_PHRASE: Readonly<Record<GPUAnalysisMeasurementLevel, string>> = {
  count: 'a count',
  amount: 'an amount',
  rate: 'a rate',
  ordinal: 'ordinal',
  nominal: 'nominal',
  cyclic: 'cyclic',
  identifier: 'an identifier'
};

const LEVEL_PLURAL: Readonly<Record<GPUAnalysisMeasurementLevel, string>> = {
  count: 'counts',
  amount: 'amounts',
  rate: 'rates',
  ordinal: 'ordinal values',
  nominal: 'nominal values',
  cyclic: 'cyclic values',
  identifier: 'identifiers'
};

function joinWithOr(items: readonly string[]): string {
  return items.length <= 1
    ? items.join('')
    : `${items.slice(0, -1).join(', ')} or ${items[items.length - 1]}`;
}

function isTypeAccepted(role: GPUAnalysisFieldRoleSpec, column: GPUAnalysisColumnProfile): boolean {
  if (role.accepts.includes(column.type)) {
    return true;
  }
  if (column.type === 'integer' && role.accepts.includes('number')) {
    return true;
  }
  return column.type === 'number' && column.integral === true && role.accepts.includes('integer');
}

/** Position of the column's type in `role.accepts`; earlier means more preferred by the role. */
function getTypePreference(
  role: GPUAnalysisFieldRoleSpec,
  column: GPUAnalysisColumnProfile
): number {
  const exact = role.accepts.indexOf(column.type);
  if (exact >= 0) {
    return exact;
  }
  const fallbackType = column.type === 'integer' ? 'number' : 'integer';
  return role.accepts.indexOf(fallbackType);
}

function isNumericRole(role: GPUAnalysisFieldRoleSpec): boolean {
  return role.accepts.includes('number') || role.accepts.includes('integer');
}

function levelWarning(
  column: GPUAnalysisColumnProfile,
  info: ColumnLevelInfo,
  meantFor: readonly GPUAnalysisMeasurementLevel[]
): string {
  const meant = joinWithOr(meantFor.map(level => LEVEL_PLURAL[level]));
  if (info.level === undefined) {
    return `${column.name} has no known measurement level; this method is meant for ${meant}`;
  }
  const guessNote =
    info.source === 'guess' && info.confidence === 'low' ? ' (a guess from the data)' : '';
  return `${column.name} is ${LEVEL_PHRASE[info.level]}${guessNote}; this method is meant for ${meant}`;
}

/**
 * True when the column's name contains the role id. Words are compared one by one, and also with
 * the separators removed, so the role `one-way` (`one`, `way`) matches `oneway`, `OneWay`,
 * `ONEWAY`, `one_way` and `is_oneway` but not `highway` or `way_id`.
 */
function hasNameAffinity(
  role: GPUAnalysisFieldRoleSpec,
  column: GPUAnalysisColumnProfile
): boolean {
  const roleWords = tokenizeGPUAnalysisColumnName(role.role);
  const columnWords = tokenizeGPUAnalysisColumnName(column.name);
  if (roleWords.length === 0) {
    return false;
  }
  if (roleWords.every(word => columnWords.includes(word))) {
    return true;
  }
  const roleJoined = roleWords.join('');
  return columnWords.join('') === roleJoined || columnWords.includes(roleJoined);
}

/**
 * Lists the columns that can fill a role, best first. A column qualifies when its type is accepted
 * (an integer column fits `number`; a number column known to be integral fits `integer`).
 * `preferred` columns have a measurement level the role lists, or the role lists none. `accepted`
 * columns fit by type only, or have a level the role lists that is only a low-confidence guess
 * (profile `measurementConfidence`, absent meaning certain for `declared` and low for `guess`),
 * which carries a warning that says it is a guess and is never `preferred`; their warning names what the column is and what the method is meant
 * for, for example `cases is a count; this method is meant for rates`. Within a group, the order
 * of `role.accepts` is the role's preference: a column whose type comes earlier in `accepts` is
 * offered first (`['time', 'integer', 'number']` offers a time column before an integer one).
 * Then columns whose name contains the role id (`population` for the role `population`) come first, then
 * declared levels before guessed ones, then column order. Identifier columns are not offered for
 * a numeric role unless nothing else fits, and then with a warning.
 */
export function matchGPUAnalysisFieldRole(
  role: GPUAnalysisFieldRoleSpec,
  columns: readonly GPUAnalysisColumnProfile[],
  options?: {excludedColumnNames?: readonly string[]}
): readonly GPUAnalysisFieldRoleMatch[] {
  const excluded = new Set(options?.excludedColumnNames ?? []);
  const identifierAllowed = role.measurement?.includes('identifier') === true;
  const numeric = isNumericRole(role);

  type Candidate = GPUAnalysisFieldRoleMatch & {rank: readonly number[]};
  const regular: Candidate[] = [];
  const identifiers: Candidate[] = [];

  columns.forEach((column, index) => {
    if (excluded.has(column.name) || !isTypeAccepted(role, column)) {
      return;
    }
    const info = describeColumnLevel(column);
    const affinityRank = hasNameAffinity(role, column) ? 0 : 1;
    const sourceRank = info.source === 'declared' ? 0 : info.confidence === 'high' ? 1 : 2;
    const rank = [getTypePreference(role, column), affinityRank, sourceRank, index];

    const levelFacts: Pick<
      GPUAnalysisFieldRoleMatch,
      'measurementSource' | 'measurementConfidence'
    > =
      info.level === undefined
        ? {}
        : {measurementSource: info.source, measurementConfidence: info.confidence};
    const unconfirmed = isUnconfirmedGuess(info);

    if (numeric && !identifierAllowed && info.level === 'identifier') {
      const candidate: Candidate = {
        name: column.name,
        fit: 'accepted',
        warning: unconfirmed
          ? `${column.name} may be an identifier (a guess from the data); confirm what it measures before using it as a measurement`
          : `${column.name} looks like an identifier, not a measurement; it labels rows and its values carry no magnitude`,
        ...levelFacts,
        rank
      };
      // A confident identifier is hidden unless nothing else fits; a guess stays in the list.
      (unconfirmed ? regular : identifiers).push(candidate);
      return;
    }
    const meantFor = role.measurement;
    if (meantFor === undefined) {
      regular.push({name: column.name, fit: 'preferred', ...levelFacts, rank});
    } else if (info.level !== undefined && meantFor.includes(info.level)) {
      regular.push(
        unconfirmed
          ? {
              name: column.name,
              fit: 'accepted',
              warning: `${column.name} looks like ${LEVEL_PHRASE[info.level]} (a guess from the data); confirm what it measures before using it`,
              ...levelFacts,
              rank
            }
          : {name: column.name, fit: 'preferred', ...levelFacts, rank}
      );
    } else {
      regular.push({
        name: column.name,
        fit: 'accepted',
        warning: levelWarning(column, info, meantFor),
        ...levelFacts,
        rank
      });
    }
  });

  const compare = (left: Candidate, right: Candidate): number => {
    for (let position = 0; position < left.rank.length; position++) {
      if (left.rank[position] !== right.rank[position]) {
        return left.rank[position] - right.rank[position];
      }
    }
    return 0;
  };
  const ordered = [
    ...regular.filter(candidate => candidate.fit === 'preferred').sort(compare),
    ...regular.filter(candidate => candidate.fit === 'accepted').sort(compare)
  ];
  if (regular.length === 0) {
    ordered.push(...identifiers.sort(compare));
  }
  return ordered.map(({rank: _rank, ...match}) => match);
}

function isRoleRequired(
  role: GPUAnalysisFieldRoleSpec,
  parameterValues: Readonly<Record<string, string | number | boolean>> | undefined
): boolean {
  if (typeof role.required === 'boolean') {
    return role.required;
  }
  const value = parameterValues?.[role.required.whenParameter];
  return value !== undefined && role.required.equals.includes(value);
}

/**
 * Assigns distinct columns to the roles of one input. Required roles go first, the role with the
 * fewest preferred (then fewest accepted) candidates first, then optional roles. A role is required when `required` is true, or when its `whenParameter` holds
 * one of the listed values in `parameterValues`. A role that takes several columns gets as many
 * as its minimum (at least one) and the user adds more; required roles are topped up with
 * `accepted` columns, optional roles are filled only from `preferred` columns so that an optional
 * role is never guessed from a poor fit or from a low-confidence guess of its level. Roles that list
 * measurement levels and were filled by such a guess are returned in `unconfirmedRoles`.
 * Required roles without a column are listed as unfilled.
 * Required roles use augmenting paths to move earlier assignments when needed to fill another
 * role. Free candidates are tried in preference order before displacing an earlier assignment,
 * so an already feasible preferred assignment is retained. A final repair pass locks earlier
 * roles to their best feasible candidates before later roles choose theirs. Optional roles use the
 * columns left. The order of `role.accepts` is followed here too. The result depends only on its arguments.
 */
export function assignGPUAnalysisFieldRoles(
  roles: readonly GPUAnalysisFieldRoleSpec[],
  columns: readonly GPUAnalysisColumnProfile[],
  parameterValues?: Readonly<Record<string, string | number | boolean>>
): GPUAnalysisFieldRoleAssignment {
  const indexed = roles.map((role, index) => {
    const candidates = matchGPUAnalysisFieldRole(role, columns);
    return {
      role,
      index,
      required: isRoleRequired(role, parameterValues),
      candidates,
      preferredCount: candidates.filter(match => match.fit === 'preferred').length,
      candidateCount: candidates.length
    };
  });
  indexed.sort((left, right) => {
    if (left.required !== right.required) {
      return left.required ? -1 : 1;
    }
    if (left.required && left.preferredCount !== right.preferredCount) {
      return left.preferredCount - right.preferredCount;
    }
    if (left.required && left.candidateCount !== right.candidateCount) {
      return left.candidateCount - right.candidateCount;
    }
    return left.index - right.index;
  });

  // One slot per minimum required column makes multi-column roles part of the same matching.
  const slots: {candidates: readonly GPUAnalysisFieldRoleMatch[]}[] = [];
  let columnOwners = new Map<string, number>();
  let slotColumns = new Map<number, string>();
  const protectedSlots = new Set<number>();
  const augmentSlot = (slot: number, visitedColumns: Set<string>): boolean => {
    const candidates = slots[slot].candidates;
    for (const candidate of candidates) {
      if (visitedColumns.has(candidate.name) || columnOwners.has(candidate.name)) continue;
      visitedColumns.add(candidate.name);
      columnOwners.set(candidate.name, slot);
      slotColumns.set(slot, candidate.name);
      return true;
    }
    for (const candidate of candidates) {
      if (visitedColumns.has(candidate.name)) continue;
      visitedColumns.add(candidate.name);
      const owner = columnOwners.get(candidate.name);
      if (owner !== undefined && !protectedSlots.has(owner) && augmentSlot(owner, visitedColumns)) {
        columnOwners.set(candidate.name, slot);
        slotColumns.set(slot, candidate.name);
        return true;
      }
    }
    return false;
  };
  const roleSlots = new Map<number, number[]>();
  for (const {role, index, required, candidates} of indexed) {
    if (!required) continue;
    const minimum = Math.max(role.multiple?.minimum ?? 1, 1);
    const wanted = Math.min(minimum, role.multiple?.maximum ?? 1);
    const previousOwners = new Map(columnOwners);
    const previousColumns = new Map(slotColumns);
    const firstSlot = slots.length;
    const addedSlots: number[] = [];
    for (let position = 0; position < wanted; position++) {
      const slot = slots.length;
      slots.push({candidates});
      if (!augmentSlot(slot, new Set())) break;
      addedSlots.push(slot);
    }
    if (addedSlots.length < minimum) {
      // An incomplete multiple role must not reserve columns that other roles can use.
      columnOwners = previousOwners;
      slotColumns = previousColumns;
      slots.length = firstSlot;
    } else {
      roleSlots.set(index, addedSlots);
    }
  }

  // Among equally complete matchings, preserve the earlier role's candidate preferences.
  // Lock slots in role order and try each candidate, repairing only the still-unlocked suffix.
  for (const assignedSlots of roleSlots.values()) {
    for (const slot of assignedSlots) {
      protectedSlots.add(slot);
      for (const candidate of slots[slot].candidates) {
        const current = slotColumns.get(slot);
        if (candidate.name === current) break;
        const owner = columnOwners.get(candidate.name);
        if (owner !== undefined && protectedSlots.has(owner)) continue;
        const previousOwners = new Map(columnOwners);
        const previousColumns = new Map(slotColumns);
        if (current !== undefined) columnOwners.delete(current);
        if (owner !== undefined) slotColumns.delete(owner);
        columnOwners.set(candidate.name, slot);
        slotColumns.set(slot, candidate.name);
        if (owner === undefined || augmentSlot(owner, new Set([candidate.name]))) break;
        columnOwners = previousOwners;
        slotColumns = previousColumns;
      }
    }
  }

  const usedColumnNames = new Set(columnOwners.keys());
  const assigned = new Map<string, string | readonly string[]>();
  const unfilled: {id: string; index: number}[] = [];
  const unconfirmed: {id: string; index: number}[] = [];
  const warnings: {text: string; index: number}[] = [];

  for (const {role, index, required, candidates} of indexed) {
    const minimum = Math.max(role.multiple?.minimum ?? 1, 1);
    const wanted = Math.min(minimum, role.multiple?.maximum ?? 1);
    const chosenNames = new Set((roleSlots.get(index) ?? []).map(slot => slotColumns.get(slot)));
    const chosen = required
      ? candidates.filter(candidate => chosenNames.has(candidate.name))
      : candidates
          .filter(
            candidate => candidate.fit === 'preferred' && !usedColumnNames.has(candidate.name)
          )
          .slice(0, wanted);
    if (chosen.length < minimum) {
      if (required) unfilled.push({id: role.role, index});
      continue;
    }
    if (
      role.measurement !== undefined &&
      chosen.some(
        match => match.measurementSource === 'guess' && match.measurementConfidence === 'low'
      )
    ) {
      unconfirmed.push({id: role.role, index});
    }
    for (const match of chosen) {
      usedColumnNames.add(match.name);
      if (match.warning !== undefined) {
        warnings.push({text: `${role.label}: ${match.warning}`, index});
      }
    }
    assigned.set(role.role, role.multiple ? chosen.map(match => match.name) : chosen[0].name);
  }

  const assignment: Record<string, string | readonly string[]> = {};
  for (const role of roles) {
    const value = assigned.get(role.role);
    if (value !== undefined) {
      assignment[role.role] = value;
    }
  }
  return {
    assignment,
    unfilledRoles: unfilled.sort((left, right) => left.index - right.index).map(entry => entry.id),
    unconfirmedRoles: unconfirmed
      .sort((left, right) => left.index - right.index)
      .map(entry => entry.id),
    warnings: warnings.sort((left, right) => left.index - right.index).map(entry => entry.text)
  };
}

// ---------------------------------------------------------------------------------------------
// Row semantics
// ---------------------------------------------------------------------------------------------

/** A guessed meaning of rows with the sentence that explains it. Always low confidence. */
export type GPUAnalysisRowSemanticsGuess = {
  value: GPUAnalysisRowSemantics;
  reason: string;
  confidence: 'low';
};

/** Words for quantities measured on a surface that exists everywhere (weather, air, ground). */
const SURFACE_MEASUREMENT_WORDS: readonly string[] = [
  'temperature',
  'temp',
  'precipitation',
  'precip',
  'rainfall',
  'snowfall',
  'elevation',
  'elev',
  'altitude',
  'depth',
  'concentration',
  'pm25',
  'pm10',
  'ozone',
  'no2',
  'co2',
  'reading',
  'readings',
  'humidity',
  'pressure',
  'salinity',
  'turbidity',
  'ph',
  'ndvi',
  'noise',
  'discharge',
  'level'
];

/** Unit suffixes of surface measurements (`pm25_ugm3` is covered by the word list). */
const SURFACE_UNIT_WORDS: readonly string[] = [
  'ppm',
  'ppb',
  'mg',
  'ug',
  'hpa',
  'kpa',
  'db',
  'celsius',
  'kelvin',
  'degc'
];

/** Words for attributes of a thing that has them (a shop, a hospital, a lot). */
const OBJECT_ATTRIBUTE_WORDS: readonly string[] = [
  'revenue',
  'sales',
  'capacity',
  'beds',
  'employees',
  'staff',
  'seats',
  'rooms',
  'bedrooms',
  'price',
  'rent',
  'cost',
  'budget',
  'assessed',
  'enrollment',
  'students',
  'floors',
  'stars',
  'rating',
  'sqft',
  'units'
];

/** Words for names that hold the row's own name: a row with a name is a thing. */
const OBJECT_NAME_WORDS: readonly string[] = ['name', 'title'];

/** Words of columns that hold the position itself, which says nothing about the rows. */
const POSITION_WORDS: readonly string[] = [
  'lat',
  'latitude',
  'lon',
  'lng',
  'longitude',
  'coord',
  'coords',
  'coordinate',
  'coordinates',
  'easting',
  'northing'
];

/** The words of a name plus each word joined to a following number (`pm` and `25` give `pm25`). */
function listNameWords(tokens: readonly string[]): readonly string[] {
  const words = [...tokens];
  tokens.forEach((token, index) => {
    const next = tokens[index + 1];
    if (next !== undefined && /^\p{N}+$/u.test(next)) {
      words.push(`${token}${next}`);
    }
  });
  return words;
}

function isPositionName(tokens: readonly string[], last: string | undefined): boolean {
  return (
    tokens.some(
      word =>
        POSITION_WORDS.includes(word) ||
        (word.length >= COORDINATE_SUFFIX_MINIMUM_LENGTH &&
          (word.endsWith('lat') || word.endsWith('lon') || word.endsWith('lng')))
    ) ||
    (last !== undefined && COORDINATE_LETTERS.includes(last))
  );
}

/** Share of points at an already used location from which locations count as repeated. */
const REPEATED_LOCATION_SHARE = 0.05;

/** Share of rows with their own name from which a name column marks the rows as things. */
const OBJECT_NAME_DISTINCT_SHARE = 0.5;

type RowEvidence = {
  surfaceColumns: string[];
  objectColumns: string[];
  hasValueColumn: boolean;
  hasTime: boolean;
  hasCategory: boolean;
};

function collectRowEvidence(columns: readonly GPUAnalysisColumnProfile[]): RowEvidence {
  const evidence: RowEvidence = {
    surfaceColumns: [],
    objectColumns: [],
    hasValueColumn: false,
    hasTime: false,
    hasCategory: false
  };
  for (const column of columns) {
    if (column.type === 'time') {
      evidence.hasTime = true;
      continue;
    }
    if (
      column.type === 'text' &&
      tokenizeGPUAnalysisColumnName(column.name).some(word => OBJECT_NAME_WORDS.includes(word)) &&
      (column.distinctCount ?? 0) >= column.rowCount * OBJECT_NAME_DISTINCT_SHARE
    ) {
      evidence.objectColumns.push(column.name);
    }
    if (column.type === 'category' || column.type === 'text' || column.type === 'boolean') {
      evidence.hasCategory ||= column.type !== 'text' || column.distinctCount !== column.rowCount;
      continue;
    }
    const tokens = tokenizeGPUAnalysisColumnName(column.name);
    const last = lastWord(tokens);
    if (isPositionName(tokens, last)) {
      continue;
    }
    if (
      listNameWords(tokens).some(word => SURFACE_MEASUREMENT_WORDS.includes(word)) ||
      (last !== undefined && SURFACE_UNIT_WORDS.includes(last))
    ) {
      evidence.surfaceColumns.push(column.name);
      evidence.hasValueColumn = true;
      continue;
    }
    const info = describeColumnLevel(column);
    if (
      listNameWords(tokens).some(word => OBJECT_ATTRIBUTE_WORDS.includes(word)) ||
      (info.level === 'amount' && info.confidence === 'high')
    ) {
      evidence.objectColumns.push(column.name);
      evidence.hasValueColumn = true;
      continue;
    }
    if (
      info.level === undefined ||
      info.level === 'count' ||
      info.level === 'amount' ||
      info.level === 'rate'
    ) {
      // A numeric column with no level is a value unless its name marks it as a time stamp.
      const isTimeLike = tokens.some(word => TIME_WORDS.includes(word));
      if (!isTimeLike) {
        evidence.hasValueColumn = true;
      }
    }
  }
  return evidence;
}

/**
 * Guesses what the rows of a dataset stand for. This is a question to ask the user, never a fact:
 * values cannot tell a rain gauge from an incident, so the confidence is always `low` and a host
 * should show the guess as a default for the user to confirm or change.
 *
 * Area rows (polygons, cells, regions) are `objects`. For point rows the evidence is the columns
 * and the repeated locations:
 * - A measured quantity (a surface measurement word such as temperature, precipitation, rainfall,
 *   elevation, depth, concentration, pm25, reading, level, or a unit suffix such as ppm) with few
 *   repeated locations (under 5 percent of points) reads as `surface-samples`.
 * - An attribute of a thing (revenue, sales, capacity, beds, employees, price, rooms, an amount
 *   by name) or a row's own name (a text column `name` that is mostly distinct) reads as
 *   `objects`.
 * - No numeric value column other than identifiers, coordinates and time, or many repeated
 *   locations, or a time column with categories, reads as `events`.
 * Evidence that conflicts (a measured quantity with an object attribute, a measured quantity at
 * many repeated locations, an object attribute with a time column) and every other kind of data
 * return undefined.
 */
export function guessGPUAnalysisRowSemantics(
  profile: GPUAnalysisDataProfile
): GPUAnalysisRowSemanticsGuess | undefined {
  if (profile.kind === 'polygons' || profile.kind === 'cells' || profile.kind === 'regions') {
    return {
      value: 'objects',
      reason: `Each ${profile.kind === 'cells' ? 'cell' : 'area'} row carries attributes of one area, so the rows most likely describe objects; confirm it.`,
      confidence: 'low'
    };
  }
  if (profile.kind !== 'points') {
    return undefined;
  }
  const evidence = collectRowEvidence(profile.columns);
  const spacing = profile.spacing;
  const repeatedShare =
    spacing === undefined || spacing.sampleSize === 0 || !(spacing.pointCount > 0)
      ? undefined
      : Math.min(1, Math.max(0, spacing.duplicateLocationCount / spacing.pointCount));
  const manyRepeated = repeatedShare !== undefined && repeatedShare >= REPEATED_LOCATION_SHARE;
  const measured = evidence.surfaceColumns.length > 0;
  const objectish = evidence.objectColumns.length > 0;

  if (measured && objectish) {
    return undefined;
  }
  // A thing with attributes that is also stamped with a time is as likely an occurrence.
  if (objectish && evidence.hasTime) {
    return undefined;
  }
  if (measured) {
    return manyRepeated
      ? undefined
      : {
          value: 'surface-samples',
          reason:
            repeatedShare === undefined
              ? `${evidence.surfaceColumns[0]} suggests surface samples; repeated locations have not been checked; confirm it.`
              : `${evidence.surfaceColumns[0]} is a measured quantity and few points share a location, so the rows most likely sample a surface; confirm it.`,
          confidence: 'low'
        };
  }
  if (objectish) {
    return {
      value: 'objects',
      reason: `${evidence.objectColumns[0]} describes a thing at each point, so the rows most likely are objects with attributes; confirm it.`,
      confidence: 'low'
    };
  }
  if (!evidence.hasValueColumn) {
    return {
      value: 'events',
      reason:
        'The rows carry no measured value, only places, times or labels, so each row most likely is an occurrence; confirm it.',
      confidence: 'low'
    };
  }
  if (manyRepeated && repeatedShare !== undefined) {
    return {
      value: 'events',
      reason: `${(repeatedShare * 100).toFixed(0)} percent of the points share a location with an earlier point, as repeated occurrences do; confirm it.`,
      confidence: 'low'
    };
  }
  if (evidence.hasTime && evidence.hasCategory) {
    return {
      value: 'events',
      reason: 'The rows have a time and categories, as occurrences do; confirm it.',
      confidence: 'low'
    };
  }
  return undefined;
}
