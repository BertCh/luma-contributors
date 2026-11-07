// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUHistogram,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';
import {
  makeResidentNumericEdgeWords,
  MAXIMUM_RESIDENT_NUMERIC_PLOT_BINS,
  type ResidentNumericPlotEdges
} from './resident-numeric-plot-edges';
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

export type ResidentNumericHistogramSummary = {
  histograms: Record<string, Uint32Array>;
  measure?: ResidentPlotMeasureSummary;
  groups?: ResidentPlotGroupingSummary;
};

/**
 * Native range-filter plot over a numeric source column: bins use the pinned native edges (d3
 * `ticks` or numeric interval thresholds, computed on the CPU) and every source row is classified
 * on the GPU by comparing sortable source words with those edges, so resident and CPU series
 * share their edges exactly. A null value counts as zero, as native d3 comparisons coerce it, a
 * NaN or infinite value is outside every bin, and only `domain[0] <= x <= domain[1]` is binned.
 * An optional y-axis measure reduces over the same keys.
 */
export class ResidentNumericHistogram {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly compiled: CompiledGPUCommandGraph;
  private readonly keys: GPUVector<'uint32'>;
  private readonly measure?: ResidentPlotMeasureRunner;
  private readonly grouping?: ResidentPlotGrouping;
  private readonly edges: Buffer;
  private readonly counts: Buffer;
  private readonly binCount: number;
  private generation = 0;
  private encodedGeneration = -1;
  private pendingRead?: Promise<ResidentNumericHistogramSummary | null>;
  private transferredBytes = 0;
  private destroyed = false;
  private completion?: Promise<void>;
  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      datasetId: string;
      field: string;
      plan: ResidentNumericPlotEdges;
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
      const column = this.lease.dataset.value.columns[props.field];
      if (!column || this.lease.dataset.value.prepared.dictionaries[props.field])
        throw new Error('Numeric plot source column is missing');
      const format = column.format;
      if (format !== 'float32' && format !== 'uint32' && format !== 'sint32')
        throw new Error('Numeric plots support Float32, Int32 and Uint32 columns');
      const binCount = props.plan.thresholds.length + 1;
      this.binCount = binCount;
      if (binCount > MAXIMUM_RESIDENT_NUMERIC_PLOT_BINS)
        throw new KeplerAllocationError('capacity', {
          message: 'Numeric plot bin capacity exceeded'
        });
      const edgeWords = makeResidentNumericEdgeWords(props.plan, format);
      const requiredBytes =
        65536 + column.length * 8 + column.data.length * 4096 + (binCount + 2) * 64;
      if (requiredBytes > props.workingMemoryBudgetBytes)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Numeric histogram exceeds working memory budget'
        });
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
      this.edges = allocate(edgeWords.words.byteLength, edgeWords.words);
      this.counts = allocate(binCount * 4);
      this.keys = new GPUVector({
        name: 'numeric-bin-keys',
        type: 'data',
        format: 'uint32',
        ownsData: true,
        data: column.data.map(
          chunk =>
            new GPUData({
              buffer: allocate(chunk.length * 4),
              format: 'uint32',
              length: chunk.length,
              ownsBuffer: true
            })
        )
      });
      const graph = new GPUCommandGraph(device, {id: 'native-numeric-histogram'});
      const source = graph.importGPUVector('values', column);
      const validityVector = this.lease.dataset.value.analytics.validity[props.field];
      const validity = validityVector
        ? graph.importGPUVector('validity', validityVector)
        : undefined;
      const keys = graph.importGPUVector('keys', this.keys);
      const edgeView = graph.createDataView(
        graph.importBuffer(
          {id: 'edges', byteLength: this.edges.byteLength, usage: this.edges.usage},
          this.edges
        ),
        {format: 'uint32x2', length: binCount + 1}
      );
      const counts = graph.createDataView(
        graph.importBuffer(
          {id: 'counts', byteLength: this.counts.byteLength, usage: this.counts.usage},
          this.counts
        ),
        {format: 'uint32', length: binCount}
      );
      const thresholdCount = binCount - 1;
      const orderSource =
        format === 'float32'
          ? `var order = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);`
          : format === 'sint32'
            ? 'var order = bits ^ 0x80000000u;'
            : 'var order = bits;';
      // Null counts as zero (native d3 coercion); its sortable word depends on the format.
      const zeroOrder = format === 'uint32' ? '0u' : '0x80000000u';
      for (const [chunkIndex, sourceChunk] of source.data.entries()) {
        if (!sourceChunk.length) continue;
        const validChunk = validity?.data[chunkIndex];
        const output = keys.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          'NativeNumericHistogram',
          sourceChunk.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `classify-${chunkIndex}`,
          resources: [
            {buffer: sourceChunk, usage: 'storage-read'},
            ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
            {buffer: edgeView, usage: 'storage-read'},
            {buffer: output, usage: 'storage-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> values:array<u32>;
@group(0) @binding(1) var<storage,read> edges:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read_write> keys:array<u32>;
${validChunk ? '@group(0) @binding(3) var<storage,read> validity:array<u32>;' : ''}
fn atLeast(order:u32,edge:vec2<u32>)->bool {return edge.y==1u || (edge.y==0u && order>=edge.x);}
fn atMost(order:u32,edge:vec2<u32>)->bool {return edge.y==1u || (edge.y==0u && order<=edge.x);}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex:u32,@builtin(workgroup_id) workgroupId:vec3<u32>) {
${getBoundedInvocationIndexSource(dispatch, 256)}
if (index >= ${sourceChunk.length}u) {return;}
var bits = values[${sourceChunk.byteOffset / 4}u+index];
var present = ${validChunk ? `validity[${validChunk.byteOffset / 4}u+index]!=0u` : 'true'};
var bucket = 0xffffffffu;
var binnable = true;
${
  format === 'float32'
    ? `if ((bits & 0x7fffffffu) >= 0x7f800000u) {binnable = present == false;}
if ((bits & 0x7fffffffu) == 0u) {bits = 0u;}`
    : ''
}
${orderSource}
if (!present) {order = ${zeroOrder};}
if (binnable && atLeast(order,edges[0]) && atMost(order,edges[${thresholdCount + 1}u])) {
  var lower=0u; var upper=${thresholdCount}u;
  loop {if(lower>=upper){break;} let middle=lower+(upper-lower)/2u;
    if(atLeast(order,edges[1u+middle])){lower=middle+1u;}else{upper=middle;}}
  bucket=lower;
}
keys[${output.byteOffset / 4}u+index]=bucket;
}`,
              shaderLayout: {
                bindings: ['values', 'edges', 'keys', ...(validChunk ? ['validity'] : [])].map(
                  (name, location) => ({name, location, group: 0, type: 'storage' as const})
                )
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    values: getBuffer(sourceChunk.buffer),
                    edges: getBuffer(edgeView.buffer),
                    keys: getBuffer(output.buffer),
                    ...(validChunk ? {validity: getBuffer(validChunk.buffer)} : {})
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
          domain: [0, binCount],
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
          binCount: binCount,
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
          binCount,
          column: props.measure.column,
          operation: props.measure.operation,
          workingMemoryBudgetBytes:
            props.measureWorkingMemoryBudgetBytes ?? props.workingMemoryBudgetBytes - requiredBytes
        });
      if (
        buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
          compiled.stats.physicalTransientResourceBytes >
        requiredBytes
      )
        throw new Error('Numeric histogram exceeded reserved working memory');
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
      throw new Error('Numeric histogram source or encoder is stale');
    this.compiled.encode(encoder, {parameters: undefined});
    this.measure?.encode(encoder);
    this.grouping?.encode(encoder);
    this.encodedGeneration = this.generation;
    return this.generation;
  }
  readSummary(): Promise<ResidentNumericHistogramSummary | null> {
    if (!this.isCurrent || this.encodedGeneration !== this.generation) return Promise.resolve(null);
    if (this.pendingRead) return this.pendingRead;
    const generation = this.generation;
    this.transferredBytes += this.binCount * 4 + (this.measure?.readbackByteLength ?? 0);
    this.pendingRead = Promise.all([
      this.counts.readAsync(0, this.binCount * 4),
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
