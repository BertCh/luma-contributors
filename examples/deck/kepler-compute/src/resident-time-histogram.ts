// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, GPUHistogram, getBoundedDispatchLayout, getBoundedInvocationIndexSource, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {TIME_WORDS_WGSL, getInt64TimeWords} from '@luma-contributors/analysis/dataframe-analysis';
import {
  convertKeplerTimeInstant,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type GPUKeplerTimeUnit,
  type KeplerMemoryReservation
} from '@luma-contributors/kepler';
import {
  utcMillisecond,
  utcSecond,
  utcMinute,
  utcHour,
  utcDay,
  utcWeek,
  utcMonth,
  utcYear
} from 'd3-time';
import {
  ResidentPlotGrouping,
  type ResidentPlotGroupingProps,
  type ResidentPlotGroupingSummary
} from './resident-plot-grouping';
import {
  makeResidentPlotMeasure,
  type ResidentPlotMeasureOperation,
  type ResidentPlotMeasureRunner,
  type ResidentPlotMeasureSummary
} from './resident-plot-measure';

import {getZonedBucketUnit, makeZonedTimeHistogramEdges} from './time-contract-histogram-plan';
import {getTimeContractSettings} from './time-contract-settings';

// The pinned native TIME_INTERVALS table uses these same d3 UTC primitives.
const TIME_INTERVALS = {
  millisecond: utcMillisecond,
  second: utcSecond,
  minute: utcMinute,
  hour: utcHour,
  day: utcDay,
  week: utcWeek,
  month: utcMonth,
  year: utcYear
};

export const MAXIMUM_NATIVE_TIME_HISTOGRAM_BINS = 16384;
export type ResidentTimeHistogramPlan = {interval: string; thresholds: readonly number[]};

/** Native UTC interval edges, bounded before allocation; no timestamps are inspected. */
export function makeResidentTimeHistogramPlan(
  domain: readonly [number, number],
  requestedInterval?: string
): ResidentTimeHistogramPlan {
  if (
    !domain.every(value => Number.isSafeInteger(value) && Math.abs(value) <= 8640000000000000) ||
    domain[0] > domain[1]
  )
    throw new Error('Time histogram requires a finite ordered native Date domain');
  if (!requestedInterval) {
    for (const interval of [
      '1-millisecond',
      '1-second',
      '5-second',
      '15-second',
      '1-minute',
      '5-minute',
      '15-minute',
      '1-hour',
      '6-hour',
      '1-day',
      '1-week',
      '1-month',
      '3-month',
      '1-year',
      '10-year',
      '100-year',
      '1000-year',
      '10000-year'
    ]) {
      try {
        const plan = makeResidentTimeHistogramPlan(domain, interval);
        if (plan.thresholds.length <= 65) return plan;
      } catch {}
    }
    throw new Error('Time histogram domain exceeds bounded native calendar intervals');
  }
  const match = /^(\d+)-(millisecond|second|minute|hour|day|week|month|year)$/.exec(
    requestedInterval
  );
  if (!match || !Number.isSafeInteger(Number(match[1])) || Number(match[1]) < 1)
    throw new Error('Unsupported native time histogram interval');
  const unit = match[2] as keyof typeof TIME_INTERVALS;
  const zonedUnit = getZonedBucketUnit(requestedInterval);
  const timeZone = getTimeContractSettings().timeZone;
  if (zonedUnit && timeZone !== 'UTC') {
    // Time contract extension: local wall-clock edges (native Kepler uses UTC edges).
    const zonedThresholds = makeZonedTimeHistogramEdges(domain, zonedUnit, timeZone);
    if (zonedThresholds.length < 2 || zonedThresholds.length > MAXIMUM_NATIVE_TIME_HISTOGRAM_BINS)
      throw new Error('Time histogram exceeds bounded calendar capacity');
    return {interval: requestedInterval, thresholds: zonedThresholds};
  }
  const interval = TIME_INTERVALS[unit].every(Number(match[1]));
  if (!interval) throw new Error('Unsupported native time histogram interval');
  const floor = Number(interval.floor(new Date(domain[0])));
  const ceiling = Number(interval.ceil(new Date(domain[1])));
  if (!Number.isSafeInteger(floor) || !Number.isSafeInteger(ceiling))
    throw new Error('Time histogram calendar boundary exceeds Date range');
  const thresholds: number[] = [];
  let current = floor;
  while (current <= ceiling) {
    if (thresholds.length >= MAXIMUM_NATIVE_TIME_HISTOGRAM_BINS)
      throw new Error('Time histogram exceeds 16384 bins; choose a coarser interval');
    thresholds.push(current);
    if (current === ceiling && current !== domain[1]) break;
    const next = Number(interval.offset(new Date(current), 1));
    if (!Number.isSafeInteger(next) || next <= current)
      throw new Error('Time histogram calendar boundary exceeds Date range');
    if (current === ceiling) {
      thresholds.push(next);
      break;
    }
    current = next;
  }
  if (thresholds.length < 2 || thresholds.length > MAXIMUM_NATIVE_TIME_HISTOGRAM_BINS)
    throw new Error('Time histogram exceeds bounded calendar capacity');
  return {interval: requestedInterval, thresholds};
}

/** Integer sources compare against ceiling edges; the inclusive final limit uses floor. */
export function makeTimeHistogramWordEdges(
  thresholds: readonly number[],
  unit: GPUKeplerTimeUnit
): BigInt64Array {
  const convert = (value: number, ceiling: boolean): bigint => {
    if (unit !== 'second') return convertKeplerTimeInstant(value, 'millisecond', unit);
    const milliseconds = BigInt(value);
    const quotient = milliseconds / 1000n;
    const remainder = milliseconds % 1000n;
    return quotient + (ceiling && remainder > 0n ? 1n : !ceiling && remainder < 0n ? -1n : 0n);
  };
  return BigInt64Array.from([
    ...thresholds.map(value => convert(value, true)),
    convert(thresholds[thresholds.length - 1], false)
  ]);
}

/** Exact word-time classification followed by the shared histogram contributor. */
export class ResidentTimeHistogram {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly compiled: CompiledGPUCommandGraph;
  private readonly keys: GPUVector<'uint32'>;
  private readonly measure?: ResidentPlotMeasureRunner;
  private readonly grouping?: ResidentPlotGrouping;
  private readonly edges: Buffer;
  private readonly counts: Buffer;
  private generation = 0;
  private encodedGeneration = -1;
  private pendingRead?: Promise<{
    histograms: Record<string, Uint32Array>;
    measure?: ResidentPlotMeasureSummary;
    groups?: ResidentPlotGroupingSummary;
  } | null>;
  private transferredBytes = 0;
  private destroyed = false;
  private completion?: Promise<void>;
  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      datasetId: string;
      field: string;
      plan: ResidentTimeHistogramPlan;
      mask: GPUVector<'uint32'>;
      workingMemoryBudgetBytes: number;
      allowIncompletePopulation?: boolean;
      /** Optional native y-axis reduction over the same bins and population. */
      measure?: {column: string; operation: ResidentPlotMeasureOperation};
      /** Native grouped series: per-label counts and measures over dense slot-and-bin keys. */
      grouping?: ResidentPlotGroupingProps;
      /** Working memory the y-axis measure may reserve; defaults to the remaining histogram budget. */
      measureWorkingMemoryBudgetBytes?: number;
    }
  ) {
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.datasetId, {
      allowIncomplete: props.allowIncompletePopulation
    });
    const buffers: Buffer[] = [];
    let reservation: KeplerMemoryReservation | undefined;
    let compiled: CompiledGPUCommandGraph | undefined;
    try {
      const source = this.lease.dataset.value.timeColumns[props.field];
      if (!source) throw new Error('Exact timestamp source is missing');
      const thresholds = props.plan.thresholds;
      if (thresholds.length < 2 || thresholds.length > MAXIMUM_NATIVE_TIME_HISTOGRAM_BINS)
        throw new Error('Time histogram bin capacity exceeded');
      if (
        !thresholds.every(
          (value, index) =>
            Number.isSafeInteger(value) && (index === 0 || value > thresholds[index - 1])
        )
      )
        throw new Error('Time histogram thresholds must be strictly increasing');
      const times = makeTimeHistogramWordEdges(thresholds, source.metadata.unit);
      const requiredBytes =
        65536 +
        source.timestamps.length * 8 +
        source.timestamps.data.length * 4096 +
        thresholds.length * 64;
      if (requiredBytes > props.workingMemoryBudgetBytes)
        throw new Error('Time histogram exceeds working memory budget');
      reservation = props.session.registry.reserveMemory(requiredBytes, 'gpu', 'scratch');
      this.reservation = reservation;
      const device = props.session.device;
      const allocate = (byteLength: number, data?: Uint32Array) => {
        const buffer = device.createBuffer({
          byteLength: Math.max(4, byteLength),
          data,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return buffer;
      };
      this.edges = allocate(times.length * 8, getInt64TimeWords(times));
      this.counts = allocate(thresholds.length * 4);
      this.keys = new GPUVector({
        name: 'time-bin-keys',
        type: 'data',
        format: 'uint32',
        ownsData: true,
        data: source.timestamps.data.map(
          chunk =>
            new GPUData({
              buffer: allocate(chunk.length * 4),
              format: 'uint32',
              length: chunk.length,
              ownsBuffer: true
            })
        )
      });
      const graph = new GPUCommandGraph(device, {id: 'native-time-histogram'});
      const timestamps = graph.importGPUVector('timestamps', source.timestamps);
      const validity = graph.importGPUVector('validity', source.validity);
      const keys = graph.importGPUVector('keys', this.keys);
      const edges = graph.createDataView(
        graph.importBuffer(
          {id: 'edges', byteLength: this.edges.byteLength, usage: this.edges.usage},
          this.edges
        ),
        {format: 'uint32x2', length: times.length}
      );
      const counts = graph.createDataView(
        graph.importBuffer(
          {id: 'counts', byteLength: this.counts.byteLength, usage: this.counts.usage},
          this.counts
        ),
        {format: 'uint32', length: thresholds.length}
      );
      for (const [chunkIndex, sourceChunk] of timestamps.data.entries()) {
        if (!sourceChunk.length) continue;
        const valid = validity.data[chunkIndex];
        const output = keys.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          'NativeTimeHistogram',
          sourceChunk.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `classify-${chunkIndex}`,
          resources: [
            {buffer: sourceChunk, usage: 'storage-read'},
            {buffer: valid, usage: 'storage-read'},
            {buffer: edges, usage: 'storage-read'},
            {buffer: output, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
${TIME_WORDS_WGSL}
@group(0) @binding(0) var<storage,read> times:array<vec2<u32>>;
@group(0) @binding(1) var<storage,read> validity:array<u32>;
@group(0) @binding(2) var<storage,read> edges:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> keys:array<u32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex:u32,@builtin(workgroup_id) workgroupId:vec3<u32>) {
${getBoundedInvocationIndexSource(dispatch, 256)}
if (index >= ${sourceChunk.length}u) {return;}
// Pinned native d3 histogram comparisons coerce null mapped timestamps to epoch zero.
let instant = select(vec2<u32>(0u),times[${sourceChunk.byteOffset / 8}u+index],validity[${valid.byteOffset / 4}u+index]!=0u);
var bucket = 0xffffffffu;
if (timeWordsCompare(instant,edges[0])>=0i && timeWordsCompare(instant,edges[${times.length - 1}u])<=0i) {
  var lower=0u; var upper=${thresholds.length}u;
  loop {if(lower>=upper){break;} let middle=lower+(upper-lower)/2u;
    if(timeWordsCompare(instant,edges[middle])>=0i){lower=middle+1u;}else{upper=middle;}}
  bucket=lower-1u;
}
keys[${output.byteOffset / 4}u+index]=bucket;
}`,
              shaderLayout: {
                bindings: ['times', 'validity', 'edges', 'keys'].map((name, location) => ({
                  name,
                  location,
                  group: 0,
                  type: 'storage' as const
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    times: getBuffer(sourceChunk.buffer),
                    validity: getBuffer(valid.buffer),
                    edges: getBuffer(edges.buffer),
                    keys: getBuffer(output.buffer)
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
      graph.add(
        new GPUHistogram({
          input: keys,
          output: counts,
          domain: [0, thresholds.length],
          mask: graph.importGPUVector('selected', props.mask)
        })
      );
      compiled = graph.compile();
      this.compiled = compiled;
      if (props.grouping)
        this.grouping = new ResidentPlotGrouping({
          session: props.session,
          datasetId: props.datasetId,
          allowIncompletePopulation: props.allowIncompletePopulation,
          grouping: props.grouping,
          keys: this.keys,
          mask: props.mask,
          binCount: thresholds.length,
          measure: props.measure,
          workingMemoryBudgetBytes: props.workingMemoryBudgetBytes - requiredBytes,
          measureWorkingMemoryBudgetBytes: props.measureWorkingMemoryBudgetBytes
        });
      else if (props.measure)
        this.measure = makeResidentPlotMeasure({
          session: props.session,
          lease: this.lease,
          keys: this.keys,
          mask: props.mask,
          binCount: thresholds.length,
          column: props.measure.column,
          operation: props.measure.operation,
          workingMemoryBudgetBytes:
            props.measureWorkingMemoryBudgetBytes ?? props.workingMemoryBudgetBytes - requiredBytes
        });
      if (
        buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
          compiled.stats.physicalTransientResourceBytes +
          thresholds.length * 4 >
        requiredBytes
      )
        throw new Error('Time histogram exceeded reserved working memory');
    } catch (error) {
      this.grouping?.destroy();
      this.measure?.destroy();
      compiled?.destroy();
      buffers.forEach(buffer => buffer.destroy());
      reservation?.release();
      void this.lease.release();
      throw error;
    }
  }
  get readbackByteLength(): number {
    return this.transferredBytes + (this.grouping?.readbackByteLength ?? 0);
  }
  private get isCurrent(): boolean {
    return (
      !this.destroyed &&
      !this.props.session.deviceLost &&
      this.props.session.registry.getDataset(this.props.datasetId)?.revision ===
        this.lease.dataset.revision
    );
  }
  /** Call when the borrowed population mask changes; unchanged render encodes stay readable. */
  invalidate(): number {
    this.encodedGeneration = -1;
    return ++this.generation;
  }
  encode(encoder: CommandEncoder): number {
    if (!this.isCurrent || encoder.device !== this.props.session.device)
      throw new Error('Time histogram source or encoder is stale');
    this.compiled.encode(encoder, {parameters: undefined});
    this.measure?.encode(encoder);
    this.grouping?.encode(encoder);
    this.encodedGeneration = this.generation;
    return this.generation;
  }
  readSummary(): Promise<{
    histograms: Record<string, Uint32Array>;
    measure?: ResidentPlotMeasureSummary;
    groups?: ResidentPlotGroupingSummary;
  } | null> {
    if (!this.isCurrent || this.encodedGeneration !== this.generation) return Promise.resolve(null);
    if (this.pendingRead) return this.pendingRead;
    const generation = this.generation;
    this.transferredBytes +=
      this.props.plan.thresholds.length * 4 + (this.measure?.readbackByteLength ?? 0);
    this.pendingRead = Promise.all([
      this.counts.readAsync(0, this.props.plan.thresholds.length * 4),
      this.measure?.readSummary(),
      this.grouping?.readSummary(() => this.isCurrent && generation === this.generation)
    ])
      .then(([bytes, measure, groups]) => {
        if (!this.isCurrent || generation !== this.generation || (this.grouping && !groups))
          return null;
        return {
          histograms: {
            'native-plot': new Uint32Array(
              bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
            )
          },
          ...(measure ? {measure} : {}),
          ...(groups ? {groups} : {})
        };
      })
      .finally(() => {
        this.pendingRead = undefined;
      });
    return this.pendingRead;
  }
  close(completion: PromiseLike<unknown>): Promise<void> {
    if (this.completion) return this.completion;
    this.destroyed = true;
    this.completion = (async () => {
      const outcomes = await Promise.allSettled([
        Promise.race([completion, this.props.session.device.lost]),
        this.pendingRead
      ]);
      this.grouping?.destroy();
      this.measure?.destroy();
      this.compiled.destroy();
      this.keys.destroy();
      this.edges.destroy();
      this.counts.destroy();
      this.reservation.release();
      await this.lease.release();
      const failure = outcomes.find(outcome => outcome.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
    })();
    return this.completion;
  }
}
