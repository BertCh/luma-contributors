// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation,
  KeplerSessionExplorer
} from '@luma-contributors/kepler';
import {
  MAXIMUM_RESIDENT_CHART_CELLS,
  MAXIMUM_RESIDENT_TIME_GROUPS,
  getResidentTimeGroupCapacity,
  type ResidentCalendarInterval,
  type ResidentTimeChartPlan
} from './resident-chart-plan';
import {
  ResidentCategoryChartRuntime,
  type ResidentCategoryChartResult,
  type ResidentChartSelection
} from './resident-chart-group-runtime';
import {ResidentIntegerChartDomain, INTEGER_DOMAIN_READ_BYTES} from './resident-integer-domain';
import {ResidentCalendarKeys} from './resident-calendar-keys';
import {ResidentTimeBinChartRuntime} from './resident-time-bin-runtime';
import {
  ResidentCalendarOverflowError,
  getNextCoarserResidentInterval
} from './resident-time-overflow';
import {makeResidentTimePeriods, type ResidentTimePeriod} from './resident-time-periods';
import {makeResidentHourPeriods} from './resident-hour-periods';
import {
  ResidentTimeExtent,
  TIME_EXTENT_READ_BYTES,
  chooseNativeTimeInterval
} from './resident-time-extent';

const CALENDAR_CAPACITY_REASON =
  'Calendar chart has more periods than its series cross-product can summarize within the 4,096-cell budget. Choose a coarser interval, fewer series categories or a narrower time range.';
const CALENDAR_OVERFLOW_REASON =
  'Calendar chart has more than 1,000 periods; native falls back to equal-width bins.';

export type ResidentTimeChartResult = Omit<ResidentCategoryChartResult, 'kind'> & {
  kind: 'time';
  periods: readonly ResidentTimePeriod[];
  /** The calendar interval that produced these periods; automatic plans resolve it per revision. */
  resolvedInterval: ResidentCalendarInterval | 'equal-width';
};

type ResidentFixedTimeChartPlan = Omit<ResidentTimeChartPlan, 'interval'> & {
  interval: ResidentCalendarInterval;
};

/** Calendar keys and bounded domain discovery, then native summaries over the same revision. */
class ResidentFixedTimeChartRuntime {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly calendar: ResidentCalendarKeys;
  private readonly domain: ResidentIntegerChartDomain;
  private readonly hourDomain?: ResidentIntegerChartDomain;
  private readonly grouped: ResidentCategoryChartRuntime;
  private tail: Promise<void> = Promise.resolve();
  private revision = 0;
  private destroyed = false;
  private destruction?: Promise<void>;
  private transferredBytes = 0;
  /** Native 1,000 periods whenever the series cross-product fits the cell budget. */
  private readonly groupCapacity: number;
  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      plan: ResidentFixedTimeChartPlan;
      selection?: ResidentChartSelection;
      workingMemoryBudgetBytes: number;
      /** Exact Float32 stand-in for an Int32 or Uint32 measure column. */
      measure?: GPUVector<'float32'>;
      /** An automatic parent owns the named selection mask across interval rebuilds. */
      retainSelectionScope?: boolean;
    }
  ) {
    this.props = {
      ...props,
      plan: {
        ...props.plan,
        excludedCpuFilterIds: [...props.plan.excludedCpuFilterIds],
        excludedGpuFieldNames: [...props.plan.excludedGpuFieldNames]
      }
    };
    this.groupCapacity = getResidentTimeGroupCapacity(props.plan.series?.categoryCount);
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.plan.datasetId);
    let reservation: KeplerMemoryReservation | undefined;
    let calendar: ResidentCalendarKeys | undefined;
    let domain: ResidentIntegerChartDomain | undefined;
    let hourDomain: ResidentIntegerChartDomain | undefined;
    let ownsScope = false;
    try {
      const {plan, session, explorer} = this.props;
      if (
        this.lease.dataset.id !== explorer.dataset.id ||
        this.lease.dataset.revision !== explorer.dataset.revision
      )
        throw new Error('Calendar chart and explorer require the same source revision');
      const source = this.lease.dataset.value.timeColumns[plan.timeColumn];
      if (
        !source ||
        (source.metadata.unit !== 'millisecond' &&
          source.metadata.unit !== 'second' &&
          source.metadata.unit !== 'microsecond' &&
          source.metadata.unit !== 'nanosecond') ||
        !/^(Timestamp|Date)/.test(source.metadata.sourceType) ||
        (plan.interval !== 'hour' &&
          plan.interval !== 'day' &&
          plan.interval !== 'week' &&
          plan.interval !== 'month' &&
          plan.interval !== 'year')
      )
        throw new Error(
          'Calendar chart requires a second, millisecond, microsecond or nanosecond Timestamp or Date and UTC hour/day/week/month/year interval'
        );
      const groupCount = this.groupCapacity;
      const cellCount = groupCount * (plan.series ? plan.series.categoryCount + 1 : 1);
      const normalizationBytesPerRow = source.metadata.unit === 'millisecond' ? 0 : 8;
      // Hour mode retains sanitized/dense-hour vectors and a second bounded domain reader.
      const hourBytes =
        plan.interval === 'hour'
          ? 65536 + source.timestamps.length * 32 + source.timestamps.data.length * 8192
          : 0;
      const ownBytes =
        65536 +
        source.timestamps.length * (64 + normalizationBytesPerRow) +
        source.timestamps.data.length * 8192 +
        hourBytes;
      const groupBytes =
        65536 +
        source.timestamps.length * 160 +
        source.timestamps.data.length * 8192 +
        cellCount * 128;
      if (
        !Number.isSafeInteger(props.workingMemoryBudgetBytes) ||
        ownBytes + groupBytes > props.workingMemoryBudgetBytes ||
        cellCount > MAXIMUM_RESIDENT_CHART_CELLS
      )
        throw new Error(
          'Calendar chart exceeds its working memory or cell budget before allocation'
        );
      reservation = session.registry.reserveMemory(ownBytes, 'gpu', 'scratch');
      if (plan.population === 'selected' && !props.selection)
        throw new Error('Calendar chart selection scopes are unavailable');
      const selected =
        plan.population === 'selected'
          ? props.selection!.getConsumerMask({
              id: `chart:${plan.chartId}`,
              excludedCpuFilterIds: plan.excludedCpuFilterIds,
              excludedGpuFieldNames: plan.excludedGpuFieldNames
            })
          : undefined;
      ownsScope = plan.population === 'selected' && !props.retainSelectionScope;
      calendar = new ResidentCalendarKeys({
        device: session.device,
        timestamps: source.timestamps,
        validity: source.validity,
        unit: source.metadata.unit,
        sourceType: source.metadata.sourceType,
        interval: plan.interval,
        capacity: groupCount
      });
      domain = new ResidentIntegerChartDomain({
        device: session.device,
        ordered: calendar.ordered,
        validity: calendar.validity,
        selected
      });
      if (plan.interval === 'hour') {
        // Hour keys and validity are owned by the same calendar helper as the day domain.
        assert(calendar.hourOrdered && calendar.hourValidity);
        hourDomain = new ResidentIntegerChartDomain({
          device: session.device,
          ordered: calendar.hourOrdered,
          validity: calendar.hourValidity,
          selected
        });
      }
      if (calendar.byteLength + domain.byteLength + (hourDomain?.byteLength ?? 0) > ownBytes)
        throw new Error('Calendar chart exceeds its reserved working memory');
      this.grouped = new ResidentCategoryChartRuntime({
        ...this.props,
        workingMemoryBudgetBytes: props.workingMemoryBudgetBytes - ownBytes,
        plan: {
          ...plan,
          kind: 'category',
          categoryColumn: plan.timeColumn,
          categoryCount: groupCount
        },
        groupingInput: {
          keys: calendar.groups,
          labels: Array.from({length: groupCount}, (_, index) => String(index))
        }
      });
      this.reservation = reservation;
      this.calendar = calendar;
      this.domain = domain;
      this.hourDomain = hourDomain;
    } catch (error) {
      hourDomain?.destroy();
      domain?.destroy();
      calendar?.destroy();
      reservation?.release();
      if (ownsScope) props.selection?.removeConsumerMask(`chart:${props.plan.chartId}`);
      void this.lease.release();
      throw error;
    }
  }
  get readbackByteLength(): number {
    return this.transferredBytes + this.grouped.readbackByteLength;
  }
  cancel(): void {
    this.revision++;
    this.grouped.cancel();
  }
  read(): Promise<ResidentTimeChartResult | null> {
    const revision = ++this.revision;
    const selectionRevision =
      this.props.plan.population === 'selected' ? this.props.selection!.currentRevision : null;
    const pending = this.tail.then(() => this.readCurrent(revision, selectionRevision));
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }
  private isCurrent(revision: number, selectionRevision: number | null): boolean {
    return (
      !this.destroyed &&
      !this.props.session.deviceLost &&
      this.lease.isValid() &&
      this.props.explorer.isCurrent &&
      revision === this.revision &&
      (selectionRevision === null ||
        (!this.props.selection!.isLayerRegistrationPending &&
          this.props.selection!.currentRevision === selectionRevision))
    );
  }
  private async readCurrent(
    revision: number,
    selectionRevision: number | null
  ): Promise<ResidentTimeChartResult | null> {
    if (!this.isCurrent(revision, selectionRevision)) return null;
    const {session, selection, plan} = this.props;
    const domainCount = plan.interval === 'hour' ? 2 : 1;
    const readbackBytes = INTEGER_DOMAIN_READ_BYTES * domainCount;
    const reservation = session.registry.reserveMemory(
      readbackBytes + MAXIMUM_RESIDENT_TIME_GROUPS * 64 * domainCount,
      'cpu',
      'readback'
    );
    try {
      const encoder = session.device.createCommandEncoder();
      if (selectionRevision !== null) selection!.encode(encoder);
      if (
        selectionRevision !== null &&
        (!selection!.canConsume || !this.isCurrent(revision, selectionRevision))
      ) {
        encoder.destroy();
        return null;
      }
      this.calendar.encodePrepare(encoder);
      this.domain.encode(encoder);
      session.device.submit(encoder.finish());
      this.transferredBytes += INTEGER_DOMAIN_READ_BYTES;
      const domain = await this.domain.read();
      if (
        !this.isCurrent(revision, selectionRevision) ||
        (selectionRevision !== null && !selection!.canConsume)
      )
        return null;
      if (domain.overflow || domain.keys.length > this.groupCapacity)
        throw this.makeCapacityError(
          domain.overflow || domain.keys.length > MAXIMUM_RESIDENT_TIME_GROUPS
        );
      let keys = domain.keys;
      let periods: ResidentTimePeriod[];
      if (plan.interval === 'hour') {
        // Keep each phase on the captured source/selection revision; no epoch-hour Int32 is formed.
        const hourDomain = this.hourDomain!;
        const hoursEncoder = session.device.createCommandEncoder();
        this.calendar.encodeHours(hoursEncoder, domain.keys);
        hourDomain.encode(hoursEncoder);
        session.device.submit(hoursEncoder.finish());
        this.transferredBytes += INTEGER_DOMAIN_READ_BYTES;
        const hours = await hourDomain.read();
        if (
          !this.isCurrent(revision, selectionRevision) ||
          (selectionRevision !== null && !selection!.canConsume)
        )
          return null;
        if (hours.overflow || hours.keys.length > this.groupCapacity)
          throw this.makeCapacityError(
            hours.overflow || hours.keys.length > MAXIMUM_RESIDENT_TIME_GROUPS
          );
        keys = hours.keys;
        periods = makeResidentHourPeriods(keys, domain.keys);
      } else {
        periods = makeResidentTimePeriods(keys, plan.interval);
        // A literal native "auto" interval floors to days but labels them with a clock time.
        if (plan.dateTimeLabels)
          periods = periods.map(period => ({...period, label: `${period.label} 00:00`}));
      }
      const groupsEncoder = session.device.createCommandEncoder();
      this.calendar.encodeGroups(groupsEncoder, keys);
      session.device.submit(groupsEncoder.finish());
      const grouped = await this.grouped.read();
      if (
        !grouped ||
        !this.isCurrent(revision, selectionRevision) ||
        (selectionRevision !== null && !selection!.canConsume)
      )
        return null;
      return {
        ...grouped,
        kind: 'time',
        periods,
        resolvedInterval: plan.interval,
        summaries: grouped.summaries.slice(0, periods.length),
        readbackBytes: readbackBytes + grouped.readbackBytes
      };
    } catch (error) {
      if (!this.isCurrent(revision, selectionRevision)) return null;
      throw error;
    } finally {
      reservation.release();
    }
  }
  /** Past native's 1,000 periods the caller falls back; otherwise the cell budget is the limit. */
  private makeCapacityError(beyondNativeLimit: boolean): Error {
    return beyondNativeLimit && this.groupCapacity === MAXIMUM_RESIDENT_TIME_GROUPS
      ? new ResidentCalendarOverflowError(CALENDAR_OVERFLOW_REASON)
      : new Error(CALENDAR_CAPACITY_REASON);
  }
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.cancel();
    this.destruction = (async () => {
      await this.tail;
      let fence: ReturnType<Device['createFence']> | undefined;
      try {
        if (!this.props.session.device.isLost) {
          fence = this.props.session.device.createFence();
          await Promise.race([fence.signaled, this.props.session.device.lost]);
        }
      } finally {
        fence?.destroy();
        this.hourDomain?.destroy();
        this.domain.destroy();
        try {
          await this.grouped.destroy();
        } finally {
          this.calendar.destroy();
          this.reservation.release();
          await this.lease.release();
        }
      }
    })();
    return this.destruction;
  }
}

/**
 * Fixed intervals delegate to one calendar runtime. Automatic intervals mirror native
 * chooseTimeInterval: an exact GPU Int64 extent over the current population selects hour to year,
 * and the calendar key vectors are rebuilt only when the chosen interval changes. A result is
 * returned only when the source, selection and cancellation revisions that selected the interval
 * are still current after the grouped read, so periods, legends and series share one revision.
 * Like native timeGroupMap, more than 1,000 periods coarsen an automatic interval one step at a
 * time and fall back to 1,000 equal-width bins (fixed intervals fall back immediately), provided
 * the series cross-product fits the cell budget.
 */
export class ResidentTimeChartRuntime {
  private child?: ResidentFixedTimeChartRuntime | ResidentTimeBinChartRuntime;
  private childInterval?: ResidentCalendarInterval;
  private readonly automatic: boolean;
  private fallbackBinCount = 0;
  private readonly lease?: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation?: KeplerMemoryReservation;
  private extent?: ResidentTimeExtent;
  private extentMask?: GPUVector<'uint32'>;
  private tail: Promise<void> = Promise.resolve();
  private revision = 0;
  private destroyed = false;
  private destruction?: Promise<void>;
  private transferredBytes = 0;
  private retiredReadbackBytes = 0;
  private readonly ownsScope: boolean;
  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      plan: ResidentTimeChartPlan;
      selection?: ResidentChartSelection;
      workingMemoryBudgetBytes: number;
      /** Exact Float32 stand-in for an Int32 or Uint32 measure column. */
      measure?: GPUVector<'float32'>;
    }
  ) {
    const {plan} = props;
    this.ownsScope = false;
    this.automatic = plan.interval === 'auto';
    if (plan.interval === 'equal-width') {
      this.child = new ResidentTimeBinChartRuntime(props);
      return;
    }
    if (plan.interval !== 'auto') {
      this.child = new ResidentFixedTimeChartRuntime({
        ...props,
        plan: {...plan, interval: plan.interval}
      });
      this.childInterval = plan.interval;
      return;
    }
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(plan.datasetId);
    let reservation: KeplerMemoryReservation | undefined;
    let extent: ResidentTimeExtent | undefined;
    let ownsScope = false;
    try {
      const source = this.lease!.dataset.value.timeColumns[plan.timeColumn];
      if (
        !source ||
        !/^(Timestamp|Date)/.test(source.metadata.sourceType) ||
        this.lease!.dataset.id !== props.explorer.dataset.id ||
        this.lease!.dataset.revision !== props.explorer.dataset.revision
      )
        throw new Error(
          'Automatic calendar chart requires a Timestamp or Date field and explorer revision'
        );
      const extentBytes =
        65536 +
        source.timestamps.length * (source.metadata.unit === 'millisecond' ? 0 : 8) +
        source.timestamps.data.length * 16384;
      if (extentBytes * 2 > props.workingMemoryBudgetBytes)
        throw new Error('Automatic calendar interval exceeds its working memory budget');
      reservation = props.session.registry.reserveMemory(extentBytes, 'gpu', 'scratch');
      if (plan.population === 'selected' && !props.selection)
        throw new Error('Calendar chart selection scopes are unavailable');
      const selected =
        plan.population === 'selected'
          ? props.selection!.getConsumerMask({
              id: `chart:${plan.chartId}`,
              excludedCpuFilterIds: plan.excludedCpuFilterIds,
              excludedGpuFieldNames: plan.excludedGpuFieldNames
            })
          : undefined;
      ownsScope = plan.population === 'selected';
      extent = this.makeExtent(selected);
      if (extent.byteLength > extentBytes)
        throw new Error('Automatic calendar interval exceeded its reserved working memory');
      this.reservation = reservation;
      this.extent = extent;
      this.extentMask = selected;
      this.ownsScope = ownsScope;
    } catch (error) {
      extent?.destroy();
      reservation?.release();
      if (ownsScope) props.selection?.removeConsumerMask(`chart:${plan.chartId}`);
      void this.lease!.release();
      throw error;
    }
  }
  private makeExtent(selected?: GPUVector<'uint32'>): ResidentTimeExtent {
    const source = this.lease!.dataset.value.timeColumns[this.props.plan.timeColumn];
    return new ResidentTimeExtent({
      device: this.props.session.device,
      timestamps: source.timestamps,
      validity: source.validity,
      selected,
      unit: source.metadata.unit,
      sourceType: source.metadata.sourceType
    });
  }
  /** A retired grouped child removes the named mask; the next extent must borrow the live one. */
  private refreshExtent(): number | null {
    const {plan, selection} = this.props;
    if (plan.population !== 'selected') return null;
    const mask = selection!.getConsumerMask({
      id: `chart:${plan.chartId}`,
      excludedCpuFilterIds: plan.excludedCpuFilterIds,
      excludedGpuFieldNames: plan.excludedGpuFieldNames
    });
    if (mask !== this.extentMask) {
      this.extent!.destroy();
      this.extent = this.makeExtent(mask);
      this.extentMask = mask;
    }
    return selection!.currentRevision;
  }
  get readbackByteLength(): number {
    return (
      this.transferredBytes + this.retiredReadbackBytes + (this.child?.readbackByteLength ?? 0)
    );
  }
  cancel(): void {
    this.revision++;
    this.child?.cancel();
  }
  /** Replace an over-capacity calendar child by the native 1,000-bin equal-width child. */
  private async useEqualWidthFallback(budget: number): Promise<void> {
    const previous = this.child;
    this.child = undefined;
    this.childInterval = undefined;
    if (previous) {
      this.retiredReadbackBytes += previous.readbackByteLength;
      await previous.destroy();
    }
    this.fallbackBinCount = MAXIMUM_RESIDENT_TIME_GROUPS;
    this.child = new ResidentTimeBinChartRuntime({
      ...this.props,
      plan: {
        ...this.props.plan,
        interval: 'equal-width',
        binCount: MAXIMUM_RESIDENT_TIME_GROUPS
      },
      workingMemoryBudgetBytes: budget
    });
  }
  private async readFixed(revision: number): Promise<ResidentTimeChartResult | null> {
    if (this.destroyed || revision !== this.revision) return null;
    try {
      return await this.child!.read();
    } catch (error) {
      if (!(error instanceof ResidentCalendarOverflowError) || this.fallbackBinCount) throw error;
      await this.useEqualWidthFallback(this.props.workingMemoryBudgetBytes);
      if (this.destroyed || revision !== this.revision) return null;
      return this.child!.read();
    }
  }
  read(): Promise<ResidentTimeChartResult | null> {
    if (!this.automatic) {
      const revision = ++this.revision;
      const pending = this.tail.then(() => this.readFixed(revision));
      this.tail = pending.then(
        () => undefined,
        () => undefined
      );
      return pending;
    }
    const revision = ++this.revision;
    const selectionRevision =
      this.props.plan.population === 'selected' ? this.props.selection!.currentRevision : null;
    const pending = this.tail.then(() => this.readAutomatic(revision, selectionRevision));
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }
  private isCurrent(revision: number, selectionRevision: number | null): boolean {
    const {selection, session, explorer} = this.props;
    return (
      !this.destroyed &&
      !session.deviceLost &&
      this.lease!.isValid() &&
      explorer.isCurrent &&
      revision === this.revision &&
      (selectionRevision === null ||
        (!selection!.isLayerRegistrationPending &&
          selection!.currentRevision === selectionRevision))
    );
  }
  private async readAutomatic(
    revision: number,
    selectionRevision: number | null
  ): Promise<ResidentTimeChartResult | null> {
    if (!this.isCurrent(revision, selectionRevision)) return null;
    const {session, selection, plan} = this.props;
    selectionRevision = this.refreshExtent();
    if (!this.isCurrent(revision, selectionRevision)) return null;
    const current = () =>
      this.isCurrent(revision, selectionRevision) &&
      (selectionRevision === null || selection!.canConsume);
    const reservation = session.registry.reserveMemory(TIME_EXTENT_READ_BYTES, 'cpu', 'readback');
    try {
      const encoder = session.device.createCommandEncoder();
      if (selectionRevision !== null) selection!.encode(encoder);
      if (selectionRevision !== null && !current()) {
        encoder.destroy();
        return null;
      }
      this.extent!.encode(encoder);
      session.device.submit(encoder.finish());
      this.transferredBytes += TIME_EXTENT_READ_BYTES;
      const extent = await this.extent!.read();
      if (!current()) return null;
      // Native returns no groups for an empty population; its zero-span interval is day.
      let interval = chooseNativeTimeInterval(extent ? extent.maximum - extent.minimum : 0);
      // A previous revision that exhausted every interval stays on 1,000 equal-width bins until
      // the span-chosen interval changes back to one that fits.
      const makeChild = async (next: ResidentCalendarInterval) => {
        const previous = this.child;
        this.child = undefined;
        this.childInterval = undefined;
        this.fallbackBinCount = 0;
        if (previous) {
          this.retiredReadbackBytes += previous.readbackByteLength;
          await previous.destroy();
        }
        if (!current()) return false;
        this.child = new ResidentFixedTimeChartRuntime({
          ...this.props,
          plan: {...plan, interval: next},
          workingMemoryBudgetBytes: this.props.workingMemoryBudgetBytes - this.reservationBytes(),
          retainSelectionScope: true
        });
        this.childInterval = next;
        return true;
      };
      if (this.fallbackBinCount || this.childInterval !== interval) {
        // The extent keeps its own reservation; the calendar runtime receives the remainder.
        if (!(await makeChild(interval))) return null;
      }
      let result: ResidentTimeChartResult | null;
      for (;;) {
        try {
          result = await this.child!.read();
          break;
        } catch (error) {
          if (!(error instanceof ResidentCalendarOverflowError) || this.fallbackBinCount)
            throw error;
          const coarser = getNextCoarserResidentInterval(interval);
          if (coarser) {
            interval = coarser;
            if (!(await makeChild(interval))) return null;
          } else {
            await this.useEqualWidthFallback(
              this.props.workingMemoryBudgetBytes - this.reservationBytes()
            );
            if (!current()) return null;
          }
        }
      }
      // The interval, periods and series belong to one revision only if nothing moved meanwhile.
      if (!result || !current()) return null;
      return result;
    } catch (error) {
      if (!this.isCurrent(revision, selectionRevision)) return null;
      throw error;
    } finally {
      reservation.release();
    }
  }
  private reservationBytes(): number {
    return this.extent!.byteLength;
  }
  destroy(): Promise<void> {
    if (!this.automatic) return this.child!.destroy();
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.cancel();
    this.destruction = (async () => {
      await this.tail;
      let fence: ReturnType<Device['createFence']> | undefined;
      try {
        if (!this.props.session.device.isLost) {
          fence = this.props.session.device.createFence();
          await Promise.race([fence.signaled, this.props.session.device.lost]);
        }
      } finally {
        fence?.destroy();
        try {
          await this.child?.destroy();
        } finally {
          this.extent!.destroy();
          if (this.ownsScope)
            this.props.selection?.removeConsumerMask(`chart:${this.props.plan.chartId}`);
          this.reservation!.release();
          await this.lease!.release();
        }
      }
    })();
    return this.destruction;
  }
}
