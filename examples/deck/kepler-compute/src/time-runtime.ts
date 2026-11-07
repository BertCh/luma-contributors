// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CommandEncoder} from '@luma.gl/core';
import type {PreparedArrowAnalyticsTimeColumn} from '@luma.gl/graph/arrow';
import {
  GPUKeplerTimeSelection,
  convertKeplerTimeInstant,
  type GPUKeplerTimeUnit,
  type GPUKeplerTimeWindow,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type KeplerMemoryReservation,
  KeplerAllocationError
} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import type {ResidentExternalRangeBinding} from './resident-table-binding';
import type {IntervalBoundaryClosure} from './time-contract-interval';
import {
  makeContractIntervalTimeSelection,
  type TimeSelectionLike
} from './time-contract-interval-gpu';

/** Native Kepler Date controls use integral epoch milliseconds inside JavaScript's Date range. */
const MAXIMUM_DATE_MILLISECONDS = 8_640_000_000_000_000;
export const SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS = Date.UTC(2026, 0, 1);

/** Pure validation used by the native table planner before changing any active GPU selections. */
export function prepareKeplerTimeRangeMilliseconds(
  minimum: number,
  maximum: number,
  sourceUnit: GPUKeplerTimeUnit
): GPUKeplerTimeWindow {
  for (const endpoint of [minimum, maximum]) validateNativeMilliseconds(endpoint);
  if (minimum > maximum) throw new Error('Native time range endpoints must be ordered');
  return {
    start: convertKeplerTimeInstant(minimum, 'millisecond', sourceUnit),
    end: convertKeplerTimeInstant(maximum, 'millisecond', sourceUnit),
    unit: sourceUnit
  };
}

/**
 * Scans retained CPU time words once at import, never during interaction or from GPU readback.
 * Sub-millisecond extrema produce enclosing integer-ms slider bounds (floor min, ceil max),
 * using bigint division. Source ticks stay unchanged; each chosen filter endpoint converts exactly.
 * Empty/all-null columns return null; callers can disable their native slider explicitly.
 */
export function getPreparedTimeDomainMilliseconds(
  column: PreparedArrowAnalyticsTimeColumn
): readonly [number, number] | null {
  if (column.chunks.length !== column.validity.length)
    throw new Error('Timestamp validity must match source batches');
  let minimum: bigint | undefined;
  let maximum: bigint | undefined;
  for (const [batchIndex, words] of column.chunks.entries()) {
    const validity = column.validity[batchIndex];
    if (words.length !== validity.length * 2)
      throw new Error('Timestamp validity must match source batches');
    for (let rowIndex = 0; rowIndex < validity.length; rowIndex++) {
      if (!validity[rowIndex]) continue;
      const instant = BigInt.asIntN(
        64,
        BigInt(words[rowIndex * 2]) | (BigInt(words[rowIndex * 2 + 1]) << 32n)
      );
      if (minimum === undefined || instant < minimum) minimum = instant;
      if (maximum === undefined || instant > maximum) maximum = instant;
    }
  }
  if (minimum === undefined || maximum === undefined) return null;
  const minimumMilliseconds = Number(getEnclosingMillisecond(minimum, column.unit, 'minimum'));
  const maximumMilliseconds = Number(getEnclosingMillisecond(maximum, column.unit, 'maximum'));
  validateNativeMilliseconds(minimumMilliseconds);
  validateNativeMilliseconds(maximumMilliseconds);
  return [minimumMilliseconds, maximumMilliseconds];
}

/** Adds a real Arrow Timestamp field without changing the demo's existing minute-offset column. */
export function makeSampleTimestampColumn(
  minutes: Uint32Array,
  options: {epochMilliseconds?: number; timeZone?: string} = {}
): {
  column: arrow.Vector<arrow.TimestampMillisecond>;
  values: Float64Array;
  domainMilliseconds: readonly [number, number];
} {
  const epoch = options.epochMilliseconds ?? SAMPLE_TIMESTAMP_EPOCH_MILLISECONDS;
  validateNativeMilliseconds(epoch);
  const timestamps = new BigInt64Array(minutes.length);
  const values = new Float64Array(minutes.length);
  let minimum = minutes.length ? Infinity : epoch;
  let maximum = minutes.length ? -Infinity : epoch;
  for (const [rowIndex, minute] of minutes.entries()) {
    const milliseconds = BigInt(epoch) + BigInt(minute) * 60_000n;
    const nativeMilliseconds = Number(milliseconds);
    validateNativeMilliseconds(nativeMilliseconds);
    timestamps[rowIndex] = milliseconds;
    values[rowIndex] = nativeMilliseconds;
    minimum = Math.min(minimum, nativeMilliseconds);
    maximum = Math.max(maximum, nativeMilliseconds);
  }
  const type = new arrow.TimestampMillisecond(options.timeZone ?? 'UTC');
  return {
    column: arrow.makeVector(arrow.makeData({type, data: timestamps, length: timestamps.length})),
    values,
    domainMilliseconds: [minimum, maximum]
  };
}

/**
 * Native millisecond controls over exact signed Int64 source words. Display and dynamic-domain
 * selections own separate masks because fixed-domain filters must not change the domain population.
 * Both graphs borrow one versioned source; external dimensions encode before their consumers.
 */
export class TimeRuntime {
  readonly metadata: PreparedArrowAnalyticsTimeColumn;
  readonly domainMilliseconds: readonly [number, number] | null;
  readonly externalRange: ResidentExternalRangeBinding;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly session: KeplerComputeSession;
  private readonly reservation: KeplerMemoryReservation;
  private readonly display: TimeSelectionLike;
  private readonly domain: TimeSelectionLike;
  private readonly pendingReads = new Set<Promise<unknown>>();
  private destroyed = false;
  private completion?: Promise<void>;

  constructor(props: {
    session: KeplerComputeSession;
    datasetId: string;
    column: string;
    dimension?: string;
    /**
     * Rows are intervals [column, endColumn): selections run the time-contract overlap predicate
     * (closure, open-ended null ends, reversed rows refused) instead of the point predicate.
     */
    interval?: {endColumn: string; closure?: IntervalBoundaryClosure};
    /** Admission reservation for both selection masks, compaction buffers and compiled scratch. */
    workingMemoryBudgetBytes?: number;
  }) {
    const lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.datasetId);
    let reservation: KeplerMemoryReservation | undefined;
    let display: TimeSelectionLike | undefined;
    let domain: TimeSelectionLike | undefined;
    try {
      const source = lease.dataset.value.timeColumns[props.column];
      if (!source) throw new Error('Native time requires a prepared exact timestamp column');
      this.metadata = source.metadata;
      this.domainMilliseconds = getPreparedTimeDomainMilliseconds(source.metadata);
      reservation = props.session.registry.reserveMemory(
        props.workingMemoryBudgetBytes ?? source.timestamps.length * 256 + 2 * 1024 * 1024,
        'gpu',
        'scratch'
      );
      this.reservation = reservation;
      this.lease = lease;
      this.session = props.session;
      const dimension = props.dimension ?? props.column;
      const selectionProps = {
        device: props.session.device,
        timestamps: source.timestamps,
        validity: source.validity,
        unit: source.metadata.unit,
        window: {start: -(1n << 63n), end: (1n << 63n) - 1n, unit: source.metadata.unit}
      };
      const endSource = props.interval
        ? lease.dataset.value.timeColumns[props.interval.endColumn]
        : undefined;
      if (props.interval && (!endSource || endSource.metadata.unit !== source.metadata.unit))
        throw new Error('Interval time requires an end column with the same prepared unit');
      const makeSelection = (id: string): TimeSelectionLike =>
        endSource
          ? makeContractIntervalTimeSelection({
              device: props.session.device,
              id,
              start: source.timestamps,
              end: endSource.timestamps,
              startValidity: source.validity,
              endValidity: endSource.validity,
              window: {start: selectionProps.window.start, end: selectionProps.window.end},
              closure: props.interval!.closure
            })
          : new GPUKeplerTimeSelection({...selectionProps, id});
      display = makeSelection(`${dimension}-display-time`);
      domain = makeSelection(`${dimension}-domain-time`);
      this.display = display;
      this.domain = domain;
      if (this.accountedResidentByteLength > reservation.byteLength)
        throw new KeplerAllocationError('working-memory', {
          message: 'Exact time selection exceeded its working memory reservation'
        });
      this.externalRange = {
        displayDimension: {
          id: dimension,
          kind: 'external',
          mask: display.mask,
          encode: encoder => this.encode(this.display, encoder)
        },
        domainDimension: {
          id: dimension,
          kind: 'external',
          mask: domain.mask,
          encode: encoder => this.encode(this.domain, encoder)
        },
        setDisplayRange: (minimum, maximum) => this.setRange(this.display, minimum, maximum),
        setDomainRange: (minimum, maximum) => this.setRange(this.domain, minimum, maximum),
        setDisplayEmpty: () => this.setEmpty(this.display),
        setDomainEmpty: () => this.setEmpty(this.domain)
      };
    } catch (error) {
      display?.destroy();
      domain?.destroy();
      reservation?.release();
      void lease.release();
      throw error;
    }
  }

  /** Bound arrow function can be installed directly as a native field binding's validateRange. */
  readonly validateRange = (minimum: number, maximum: number): void => {
    prepareKeplerTimeRangeMilliseconds(minimum, maximum, this.metadata.unit);
  };

  get accountedResidentByteLength(): number {
    return this.display.accountedResidentByteLength + this.domain.accountedResidentByteLength;
  }

  get readbackByteLength(): number {
    return this.display.readbackByteLength + this.domain.readbackByteLength;
  }

  /** Optional bounded diagnostics after both external dimensions have been encoded and submitted. */
  readStatus(): Promise<{
    display: Awaited<ReturnType<TimeSelectionLike['readStatus']>>;
    domain: Awaited<ReturnType<TimeSelectionLike['readStatus']>>;
  }> {
    this.assertActive();
    const pending = Promise.all([this.display.readStatus(), this.domain.readStatus()]).then(
      ([display, domain]) => ({display, domain})
    );
    this.pendingReads.add(pending);
    void pending.then(
      () => this.pendingReads.delete(pending),
      () => this.pendingReads.delete(pending)
    );
    return pending;
  }

  destroy(completion: PromiseLike<unknown>): Promise<void> {
    if (this.completion) return this.completion;
    this.destroyed = true;
    this.completion = (async () => {
      const outcomes = await Promise.allSettled([
        Promise.resolve(completion),
        ...this.pendingReads
      ]);
      this.display.destroy();
      this.domain.destroy();
      this.reservation.release();
      await this.lease.release();
      const failed = outcomes.find(outcome => outcome.status === 'rejected');
      if (failed?.status === 'rejected') throw failed.reason;
    })();
    return this.completion;
  }

  private setRange(selection: TimeSelectionLike, minimum: number, maximum: number): void {
    this.assertActive();
    selection.setWindow(prepareKeplerTimeRangeMilliseconds(minimum, maximum, this.metadata.unit));
  }

  private setEmpty(selection: TimeSelectionLike): void {
    this.assertActive();
    selection.setWindow({start: 1n, end: 0n, unit: this.metadata.unit});
  }

  private encode(selection: TimeSelectionLike, encoder: CommandEncoder): void {
    this.assertActive();
    selection.encode(encoder);
    if (this.accountedResidentByteLength > this.reservation.byteLength)
      throw new KeplerAllocationError('working-memory', {
        message: 'Exact time selection exceeded its working memory reservation'
      });
  }

  private assertActive(): void {
    if (
      this.destroyed ||
      !this.lease.isValid() ||
      this.session.registry.getDataset(this.lease.dataset.id)?.revision !==
        this.lease.dataset.revision
    )
      throw new Error('Timestamp source is no longer active');
  }
}

function validateNativeMilliseconds(value: number): void {
  if (!Number.isSafeInteger(value) || Math.abs(value) > MAXIMUM_DATE_MILLISECONDS)
    throw new Error('Native time endpoints require integral milliseconds in the Date range');
}

function getEnclosingMillisecond(
  instant: bigint,
  sourceUnit: GPUKeplerTimeUnit,
  bound: 'minimum' | 'maximum'
): bigint {
  if (sourceUnit === 'second' || sourceUnit === 'millisecond')
    return convertKeplerTimeInstant(instant, sourceUnit, 'millisecond');
  const divisor = sourceUnit === 'microsecond' ? 1000n : 1_000_000n;
  const quotient = instant / divisor;
  const remainder = instant % divisor;
  if (bound === 'minimum' && remainder < 0n) return quotient - 1n;
  if (bound === 'maximum' && remainder > 0n) return quotient + 1n;
  return quotient;
}
