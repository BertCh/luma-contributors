// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ArrowAnalyticsTimeUnit} from '../graph-support/arrow-analytics-normalization';

/**
 * Physical form of a time column.
 * - `float32-relative`: one `float32` per row, ticks after the context epoch.
 * - `double-single`: a `float32` high and a `float32` low part per row, whose sum is the time.
 * - `int64-words`: one `uint32x2` Int64 `(low, high)` row per row, two's complement. Differences
 *   are taken exactly on the words (WGSL `timeWordsDifferenceToF32`) and only then rounded to f32.
 */
export type GPUTemporalEncoding = 'float32-relative' | 'double-single' | 'int64-words';

/**
 * Instant the stored numbers count from: `'unix'` (1970-01-01T00:00:00Z), or an origin given as
 * signed Int64 ticks of the context unit after the Unix epoch. `0n` is the same instant as `'unix'`.
 */
export type GPUTemporalEpoch = 'unix' | bigint;

/** Calendar unit of a regular axis whose steps have no fixed length in ticks. */
export type GPUTemporalCalendarUnit = 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour';

/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */
export type GPUTemporalCalendarStep = {
  calendarUnit: GPUTemporalCalendarUnit;
  interval?: number;
};

/**
 * Spacing of the times. A numeric `step` is a fixed width in the context unit (the unit a
 * `maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional
 * numbers are allowed only for the two float encodings. A calendar step is regular on the
 * calendar only.
 */
export type GPUTemporalAxis =
  | {kind: 'regular'; step: number | bigint | GPUTemporalCalendarStep}
  | {kind: 'irregular'};

/** Compile-time description of a time column; unit and time zone conversion are deliberately absent. */
export type GPUTemporalContext = {
  /** Unit of one tick, and of every duration parameter given with these times. */
  unit: ArrowAnalyticsTimeUnit;
  epoch: GPUTemporalEpoch;
  encoding: GPUTemporalEncoding;
  /**
   * Time zone name as declared by the source, for example an Arrow timestamp's `'UTC'` or
   * `'+05:30'`. Descriptive: stored instants stay UTC, the name is never parsed, it does not affect
   * compatibility, and local calendars take their offsets from their own inputs. Absent means
   * undeclared.
   */
  timeZone?: string;
  axis: GPUTemporalAxis;
};

/** Encodings, units and axes an operation accepts, declared next to its implementation. */
export type GPUTemporalContextSupport = {
  /** Accepted encodings. Omitted accepts any. */
  encodings?: readonly GPUTemporalEncoding[];
  /** Accepted units. Omitted accepts any. */
  units?: readonly ArrowAnalyticsTimeUnit[];
  /** Accepted axis kinds. Omitted accepts any. */
  axisKinds?: readonly GPUTemporalAxis['kind'][];
  /** True when the operation decodes civil dates, so times must count from the Unix epoch. */
  unixEpoch?: boolean;
};

/** Result of checking whether two time columns can be used together without conversion. */
export type GPUTemporalContextCompatibility =
  | {compatible: true}
  | {compatible: false; reason: string};

/** Axis of times with no declared spacing, which is what any uploaded column is until proven otherwise. */
export const GPU_IRREGULAR_TEMPORAL_AXIS: GPUTemporalAxis = Object.freeze({kind: 'irregular'});

const UNITS: readonly string[] = ['second', 'millisecond', 'microsecond', 'nanosecond'];
const ENCODINGS: readonly string[] = ['float32-relative', 'double-single', 'int64-words'];
const CALENDAR_UNITS: readonly string[] = ['year', 'quarter', 'month', 'week', 'day', 'hour'];
const INT64_MINIMUM = -(1n << 63n);
const INT64_MAXIMUM = (1n << 63n) - 1n;

/** Validates one temporal context without converting anything. */
export function validateGPUTemporalContext(id: string, context: GPUTemporalContext): void {
  const {unit, epoch, encoding, timeZone, axis} = context;
  if (!UNITS.includes(unit)) {
    throw new Error(`${id} temporal unit ${unit} must be one of ${UNITS.join(', ')}`);
  }
  if (!ENCODINGS.includes(encoding)) {
    throw new Error(`${id} temporal encoding ${encoding} must be one of ${ENCODINGS.join(', ')}`);
  }
  if (
    epoch !== 'unix' &&
    (typeof epoch !== 'bigint' || epoch < INT64_MINIMUM || epoch > INT64_MAXIMUM)
  ) {
    throw new Error(`${id} epoch must be 'unix' or a signed Int64 bigint origin`);
  }
  if (encoding === 'float32-relative' && epoch === 'unix') {
    // float32 cannot hold absolute time: epoch seconds round to 128 s.
    throw new Error(`${id} float32-relative times need a bigint origin epoch, not 'unix'`);
  }
  if (timeZone !== undefined && (typeof timeZone !== 'string' || timeZone.trim() === '')) {
    throw new Error(`${id} timeZone must be a non-empty string`);
  }
  validateAxis(id, encoding, axis);
}

/**
 * Validates an operation's temporal context against what it supports and returns it. There is no
 * default context: a unit or epoch must be declared, never assumed. Unsupported contexts throw at
 * construction, and data is never converted.
 */
export function resolveGPUTemporalContext(
  id: string,
  context: GPUTemporalContext,
  support: GPUTemporalContextSupport = {}
): GPUTemporalContext {
  validateGPUTemporalContext(id, context);
  if (support.encodings && !support.encodings.includes(context.encoding)) {
    throw new Error(
      `${id} does not support ${context.encoding} times; it reads ${support.encodings.join(' or ')}`
    );
  }
  if (support.units && !support.units.includes(context.unit)) {
    throw new Error(
      `${id} does not support ${context.unit} times; it reads ${support.units.join(' or ')}`
    );
  }
  if (support.axisKinds && !support.axisKinds.includes(context.axis.kind)) {
    throw new Error(`${id} does not support a ${context.axis.kind} temporal axis`);
  }
  if (support.unixEpoch && getEpochOrigin(context.epoch) !== 0n) {
    throw new Error(`${id} needs times counted from the Unix epoch`);
  }
  return context;
}

/**
 * Checks that two time columns share a unit, epoch and encoding, without changing either. Time zone
 * and axis are not compared: instants stay comparable across both.
 */
export function getGPUTemporalContextCompatibility(
  source: GPUTemporalContext,
  target: GPUTemporalContext
): GPUTemporalContextCompatibility {
  validateGPUTemporalContext('source temporal context', source);
  validateGPUTemporalContext('target temporal context', target);
  if (source.unit !== target.unit) {
    return {compatible: false, reason: `units differ (${source.unit} and ${target.unit})`};
  }
  if (getEpochOrigin(source.epoch) !== getEpochOrigin(target.epoch)) {
    return {compatible: false, reason: `epochs differ (${source.epoch} and ${target.epoch})`};
  }
  if (source.encoding !== target.encoding) {
    return {
      compatible: false,
      reason: `encodings differ (${source.encoding} and ${target.encoding})`
    };
  }
  return {compatible: true};
}

/** Throws a short graph-construction error when two temporal contexts are incompatible. */
export function assertCompatibleGPUTemporalContexts(
  id: string,
  source: GPUTemporalContext,
  target: GPUTemporalContext
): void {
  const compatibility = getGPUTemporalContextCompatibility(source, target);
  if ('reason' in compatibility) {
    throw new Error(`${id} incompatible temporal contexts: ${compatibility.reason}`);
  }
}

function getEpochOrigin(epoch: GPUTemporalEpoch): bigint {
  return epoch === 'unix' ? 0n : epoch;
}

function validateAxis(id: string, encoding: GPUTemporalEncoding, axis: GPUTemporalAxis): void {
  if (axis?.kind === 'irregular') {
    return;
  }
  if (axis?.kind !== 'regular') {
    throw new Error(`${id} axis must be regular or irregular`);
  }
  const {step} = axis;
  if (typeof step === 'object' && step !== null) {
    const {calendarUnit, interval = 1} = step;
    if (!CALENDAR_UNITS.includes(calendarUnit) || !Number.isInteger(interval) || interval < 1) {
      throw new Error(`${id} calendar step needs a calendar unit and a positive integer interval`);
    }
    return;
  }
  if (
    (typeof step !== 'bigint' && !(typeof step === 'number' && Number.isFinite(step))) ||
    step <= 0
  ) {
    throw new Error(`${id} axis step must be positive and finite`);
  }
  if (encoding === 'int64-words' && typeof step === 'number' && !Number.isSafeInteger(step)) {
    throw new Error(`${id} int64-words axis step must be a safe integer or bigint`);
  }
}
