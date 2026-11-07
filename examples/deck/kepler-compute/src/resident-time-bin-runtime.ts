// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {TIME_WORDS_WGSL, getInt64TimeWords} from '@luma-contributors/analysis/dataframe-analysis';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation,
  KeplerSessionExplorer
} from '@luma-contributors/kepler';
import {
  MAXIMUM_RESIDENT_TIME_GROUPS,
  MAXIMUM_RESIDENT_CHART_CELLS,
  type ResidentTimeChartPlan
} from './resident-chart-plan';
import {
  ResidentCategoryChartRuntime,
  type ResidentChartSelection
} from './resident-chart-group-runtime';
import {makeResidentCalendarTimeView} from './resident-calendar-time';
import {ResidentTimeExtent, TIME_EXTENT_READ_BYTES} from './resident-time-extent';
import {makeResidentTimeBarPeriods} from './resident-time-bar-periods';
import type {ResidentTimeChartResult} from './resident-chart-time-runtime';

/**
 * Native time bars: exact GPU Int64 extent, native real-valued equal-width thresholds, then a
 * GPU classification of each row into a dense bin key reduced by the shared grouped runtime.
 * A row t lands in the bin counting thresholds <= t, which for integer millisecond t equals
 * counting ceil(threshold) <= t, so edges are exact Int64 words. Nulls are native epoch-zero
 * points; valid instants outside Date's range and unselected rows are excluded.
 */
export class ResidentTimeBinChartRuntime {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly extent: ResidentTimeExtent;
  private readonly classify: CompiledGPUCommandGraph;
  private readonly edges: Buffer;
  private readonly keys: GPUVector<'uint32'>;
  private readonly grouped: ResidentCategoryChartRuntime;
  private readonly binCount: number;
  private tail: Promise<void> = Promise.resolve();
  private revision = 0;
  private destroyed = false;
  private destruction?: Promise<void>;
  private transferredBytes = 0;
  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      plan: ResidentTimeChartPlan;
      selection?: ResidentChartSelection;
      workingMemoryBudgetBytes: number;
    }
  ) {
    const {plan, session} = props;
    this.lease = session.registry.acquireDataset<KeplerArrowDataset>(plan.datasetId);
    const buffers: Buffer[] = [];
    let reservation: KeplerMemoryReservation | undefined;
    let extent: ResidentTimeExtent | undefined;
    let classify: CompiledGPUCommandGraph | undefined;
    let ownsScope = false;
    try {
      const binCount = plan.binCount!;
      const source = this.lease.dataset.value.timeColumns[plan.timeColumn];
      if (
        !source ||
        !/^(Timestamp|Date)/.test(source.metadata.sourceType) ||
        !Number.isSafeInteger(binCount) ||
        binCount < 2 ||
        binCount > MAXIMUM_RESIDENT_TIME_GROUPS ||
        binCount * (plan.series ? plan.series.categoryCount + 1 : 1) >
          MAXIMUM_RESIDENT_CHART_CELLS ||
        this.lease.dataset.id !== props.explorer.dataset.id ||
        this.lease.dataset.revision !== props.explorer.dataset.revision
      )
        throw new Error('Time bar chart requires a bounded bin count and the explorer revision');
      this.binCount = binCount;
      const normalizationBytes = source.metadata.unit === 'millisecond' ? 0 : 8;
      const ownBytes =
        65536 +
        source.timestamps.length * (4 + 2 * normalizationBytes) +
        source.timestamps.data.length * 24576 +
        binCount * 64;
      const groupBytes =
        65536 +
        source.timestamps.length * 160 +
        source.timestamps.data.length * 8192 +
        binCount * (plan.series ? plan.series.categoryCount + 1 : 1) * 128;
      if (
        !Number.isSafeInteger(props.workingMemoryBudgetBytes) ||
        ownBytes + groupBytes > props.workingMemoryBudgetBytes
      )
        throw new Error('Time bar chart exceeds its working memory budget before allocation');
      reservation = session.registry.reserveMemory(ownBytes, 'gpu', 'scratch');
      if (plan.population === 'selected' && !props.selection)
        throw new Error('Time bar chart selection scopes are unavailable');
      const selected =
        plan.population === 'selected'
          ? props.selection!.getConsumerMask({
              id: `chart:${plan.chartId}`,
              excludedCpuFilterIds: plan.excludedCpuFilterIds,
              excludedGpuFieldNames: plan.excludedGpuFieldNames
            })
          : undefined;
      ownsScope = plan.population === 'selected';
      extent = new ResidentTimeExtent({
        device: session.device,
        timestamps: source.timestamps,
        validity: source.validity,
        selected,
        unit: source.metadata.unit,
        sourceType: source.metadata.sourceType
      });
      // Header, minimum, up to binCount - 1 thresholds and maximum, each as two Int64 words.
      this.edges = session.device.createBuffer({
        id: 'time-bar-edges',
        byteLength: (binCount + 3) * 8,
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
      buffers.push(this.edges);
      this.keys = new GPUVector({
        name: 'time-bar-keys',
        type: 'data',
        format: 'uint32',
        ownsData: true,
        data: source.timestamps.data.map(chunk => {
          const buffer = session.device.createBuffer({
            id: 'time-bar-keys',
            byteLength: Math.max(4, chunk.length * 4),
            usage: Buffer.STORAGE
          });
          buffers.push(buffer);
          return new GPUData({
            buffer,
            format: 'uint32',
            length: chunk.length,
            ownsBuffer: true
          });
        })
      });
      classify = this.makeClassifyGraph(session.device, source.timestamps, source.validity, {
        unit: source.metadata.unit,
        sourceType: source.metadata.sourceType
      });
      if (
        extent.byteLength +
          classify.stats.physicalTransientResourceBytes +
          buffers.reduce((total, buffer) => total + buffer.byteLength, 0) >
        ownBytes
      )
        throw new Error('Time bar chart exceeds its reserved working memory');
      this.grouped = new ResidentCategoryChartRuntime({
        ...props,
        workingMemoryBudgetBytes: props.workingMemoryBudgetBytes - ownBytes,
        plan: {
          ...plan,
          kind: 'category',
          categoryColumn: plan.timeColumn,
          categoryCount: binCount
        },
        groupingInput: {
          keys: this.keys,
          labels: Array.from({length: binCount}, (_, index) => String(index))
        }
      });
      this.reservation = reservation;
      this.extent = extent;
      this.classify = classify;
    } catch (error) {
      classify?.destroy();
      extent?.destroy();
      for (const buffer of buffers) buffer.destroy();
      reservation?.release();
      if (ownsScope) props.selection?.removeConsumerMask(`chart:${plan.chartId}`);
      void this.lease.release();
      throw error;
    }
  }
  private makeClassifyGraph(
    device: Device,
    timestamps: GPUVector<'uint32x2'>,
    validity: GPUVector<'uint32'>,
    source: {unit: Parameters<typeof makeResidentCalendarTimeView>[1]['unit']; sourceType: string}
  ): CompiledGPUCommandGraph {
    const graph = new GPUCommandGraph(device, {id: 'chart-time-bars'});
    const times = graph.importGPUVector('times', timestamps);
    const valid = graph.importGPUVector('validity', validity);
    const keys = graph.importGPUVector('keys', this.keys);
    const edges = graph.createDataView(
      graph.importBuffer(
        {id: 'edges', byteLength: this.edges.byteLength, usage: this.edges.usage},
        this.edges
      ),
      {format: 'uint32x2', length: this.binCount + 3}
    );
    for (const [chunkIndex, chunk] of times.data.entries()) {
      if (!chunk.length) continue;
      const normalized = makeResidentCalendarTimeView(graph, {
        id: `time-bar-time-${chunkIndex}`,
        timestamps: chunk,
        unit: source.unit,
        sourceType: source.sourceType
      });
      const present = valid.data[chunkIndex];
      const output = keys.data[chunkIndex];
      const dispatch = getBoundedDispatchLayout(
        'TimeBarKeys',
        normalized.length,
        256,
        device.limits.maxComputeWorkgroupsPerDimension
      );
      graph.addComputePass({
        id: `time-bar-classify-${chunkIndex}`,
        resources: [
          {buffer: normalized, usage: 'storage-read'},
          {buffer: present, usage: 'storage-read'},
          {buffer: edges, usage: 'storage-read'},
          {buffer: output, usage: 'storage-write'}
        ],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            source: `
${TIME_WORDS_WGSL}
@group(0) @binding(0) var<storage,read> times: array<u32>;
@group(0) @binding(1) var<storage,read> validity: array<u32>;
@group(0) @binding(2) var<storage,read> edges: array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> keys: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${normalized.length}u) {return;}
  let isPresent = validity[${present.byteOffset / 4}u + index] != 0u;
  let timeIndex = ${normalized.byteOffset / 4}u + index * 2u;
  // Native null mapped values become epoch zero; out-of-range valid instants stay outside the edges.
  let instant = select(vec2<u32>(0u), vec2<u32>(times[timeIndex], times[timeIndex + 1u]), isPresent);
  // edges[0].x is the retained threshold count, edges[1] the minimum, edges[count + 2] the maximum.
  let thresholdCount = edges[0].x;
  var bucket = 0xffffffffu;
  if (timeWordsCompare(instant, edges[1]) >= 0i && timeWordsCompare(instant, edges[thresholdCount + 2u]) <= 0i) {
    var lower = 0u;
    var upper = thresholdCount;
    loop {
      if (lower >= upper) {break;}
      let middle = lower + (upper - lower) / 2u;
      if (timeWordsCompare(instant, edges[middle + 2u]) >= 0i) {lower = middle + 1u;} else {upper = middle;}
    }
    bucket = lower;
  }
  keys[${output.byteOffset / 4}u + index] = bucket;
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
                  times: getBuffer(normalized.buffer),
                  validity: getBuffer(present.buffer),
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
    return graph.compile();
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
    const {selection} = this.props;
    return (
      !this.destroyed &&
      !this.props.session.deviceLost &&
      this.lease.isValid() &&
      this.props.explorer.isCurrent &&
      revision === this.revision &&
      (selectionRevision === null ||
        (!selection!.isLayerRegistrationPending &&
          selection!.currentRevision === selectionRevision &&
          selection!.canConsume))
    );
  }
  private async readCurrent(
    revision: number,
    selectionRevision: number | null
  ): Promise<ResidentTimeChartResult | null> {
    if (!this.isCurrent(revision, selectionRevision)) return null;
    const {session, selection} = this.props;
    const reservation = session.registry.reserveMemory(TIME_EXTENT_READ_BYTES, 'cpu', 'readback');
    try {
      const encoder = session.device.createCommandEncoder();
      if (selectionRevision !== null) selection!.encode(encoder);
      if (!this.isCurrent(revision, selectionRevision)) {
        encoder.destroy();
        return null;
      }
      this.extent.encode(encoder);
      session.device.submit(encoder.finish());
      this.transferredBytes += TIME_EXTENT_READ_BYTES;
      const extent = await this.extent.read();
      if (!this.isCurrent(revision, selectionRevision)) return null;
      if (!extent) {
        // Native has no points, hence no groups.
        return {
          kind: 'time',
          summaries: [],
          periods: [],
          resolvedInterval: 'equal-width',
          sourceRevision: this.lease.dataset.revision,
          selectionRevision,
          readbackBytes: TIME_EXTENT_READ_BYTES
        };
      }
      const layout = makeResidentTimeBarPeriods(extent.minimum, extent.maximum, this.binCount);
      // t >= threshold is t >= ceil(threshold) for integer milliseconds; both extrema are integers.
      const edges = BigInt64Array.from([
        BigInt(layout.thresholds.length),
        BigInt(extent.minimum),
        ...layout.thresholds.map(value => BigInt(Math.ceil(value))),
        BigInt(extent.maximum)
      ]);
      const words = getInt64TimeWords(edges);
      this.edges.write(words);
      const classifyEncoder = session.device.createCommandEncoder();
      this.classify.encode(classifyEncoder, {parameters: undefined});
      session.device.submit(classifyEncoder.finish());
      const grouped = await this.grouped.read();
      if (!grouped || !this.isCurrent(revision, selectionRevision)) return null;
      return {
        ...grouped,
        kind: 'time',
        periods: layout.periods,
        resolvedInterval: 'equal-width',
        summaries: layout.sourceBins.map(index => grouped.summaries[index]),
        readbackBytes: TIME_EXTENT_READ_BYTES + grouped.readbackBytes
      };
    } catch (error) {
      if (!this.isCurrent(revision, selectionRevision)) return null;
      throw error;
    } finally {
      reservation.release();
    }
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
        this.classify.destroy();
        this.extent.destroy();
        try {
          await this.grouped.destroy();
        } finally {
          this.keys.destroy();
          this.edges.destroy();
          this.reservation.release();
          await this.lease.release();
        }
      }
    })();
    return this.destruction;
  }
}
