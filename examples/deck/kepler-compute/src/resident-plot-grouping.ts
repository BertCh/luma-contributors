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
  makeResidentPlotMeasure,
  type ResidentPlotMeasureOperation,
  type ResidentPlotMeasureRunner,
  type ResidentPlotMeasureSummary
} from './resident-plot-measure';

/** Declared bound on dictionary labels of a grouping field (first-seen tables are read back). */
export const MAXIMUM_RESIDENT_PLOT_GROUP_LABELS = 4096;
/** Pinned native `MAX_PLOT_NUM_GROUPS` and `DEFAULT_PLOT_NUM_GROUPS` of `groupLimit`. */
const MAXIMUM_NATIVE_PLOT_GROUPS = 20;
const DEFAULT_NATIVE_PLOT_GROUPS = 10;
const NO_KEY = 0xffffffff;

/** Native `groupLimit(numGroups)`: ALL is the maximum, an invalid count the default. */
export function getResidentPlotGroupLimit(numGroups: unknown): number {
  if (numGroups === 'ALL') return MAXIMUM_NATIVE_PLOT_GROUPS;
  if (typeof numGroups !== 'number' || !(numGroups > 0)) return DEFAULT_NATIVE_PLOT_GROUPS;
  return Math.min(numGroups, MAXIMUM_NATIVE_PLOT_GROUPS);
}

export type ResidentPlotGroupingProps = {
  /** Dictionary column (uint32 codes) whose `String(label)` names the series. */
  column: string;
  numGroups: unknown;
  groupOthers: boolean;
};

export type ResidentPlotGroupingSummary = {
  /** Series in native first-seen order; `others` marks a final merged series of the remaining labels. */
  labels: readonly string[];
  others: boolean;
  /** Dense `slot * binCount + bin` row counts of the plot population. */
  counts: Uint32Array;
  /** Optional y-axis reduction over the same dense keys. */
  measure?: ResidentPlotMeasureSummary;
};

/**
 * Native grouped line series. Native discovers groups by walking the plot bins in order, and the
 * rows of each bin in index order, keeps the first `groupLimit(numGroups)` labels, and optionally
 * merges the remaining labels into Others. The GPU finds each label's earliest (bin, row), the CPU
 * ranks at most 4096 labels, and a second pass maps every row to a dense `slot * binCount + bin`
 * key over which the counts and the optional y measure are reduced. Capacity is fixed at
 * `groupLimit + 1` slots, so the slot map is a per-frame parameter and nothing is recompiled.
 *
 * Phase one (`encode`) shares the host frame; phase two runs in `readSummary`, which submits once
 * after reading the small first-seen tables. The caller discards the result when its population
 * changed in between.
 */
export class ResidentPlotGrouping {
  readonly slotCapacity: number;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly firstSeen: CompiledGPUCommandGraph;
  private readonly assignment: CompiledGPUCommandGraph;
  private readonly composite: GPUVector<'uint32'>;
  private readonly measure?: ResidentPlotMeasureRunner;
  private readonly buffers: Buffer[] = [];
  private readonly firstBins: Buffer;
  private readonly firstRows: Buffer;
  private readonly slotMap: Buffer;
  private readonly counts: Buffer;
  private readonly labels: readonly string[];
  private readonly limit: number;
  private transferredBytes = 0;
  private destroyed = false;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      datasetId: string;
      allowIncompletePopulation?: boolean;
      grouping: ResidentPlotGroupingProps;
      /** Per-row plot bin key (`0xffffffff` outside the plot). */
      keys: GPUVector<'uint32'>;
      mask: GPUVector<'uint32'>;
      binCount: number;
      measure?: {column: string; operation: ResidentPlotMeasureOperation};
      workingMemoryBudgetBytes: number;
      measureWorkingMemoryBudgetBytes?: number;
    }
  ) {
    const {session, grouping, binCount} = props;
    this.lease = session.registry.acquireDataset<KeplerArrowDataset>(props.datasetId, {
      allowIncomplete: props.allowIncompletePopulation
    });
    let reservation: KeplerMemoryReservation | undefined;
    const compiled: CompiledGPUCommandGraph[] = [];
    try {
      const source = this.lease.dataset.value;
      const codes = source.columns[grouping.column];
      const labels = source.prepared.dictionaries[grouping.column]?.values;
      if (!codes || codes.format !== 'uint32' || !labels)
        throw new Error('Grouped line charts need a prepared string dictionary field');
      if (labels.length < 1 || labels.length > MAXIMUM_RESIDENT_PLOT_GROUP_LABELS)
        throw new Error(
          `Grouped line charts support at most ${MAXIMUM_RESIDENT_PLOT_GROUP_LABELS} labels`
        );
      this.labels = labels.map(label => String(label));
      this.limit = getResidentPlotGroupLimit(grouping.numGroups);
      this.slotCapacity = this.limit + 1;
      const keyCount = this.slotCapacity * binCount;
      const requiredBytes =
        65536 + codes.length * 4 + codes.data.length * 8192 + keyCount * 16 + labels.length * 32;
      if (requiredBytes > props.workingMemoryBudgetBytes)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Grouped line chart exceeds its working memory budget'
        });
      reservation = session.registry.reserveMemory(requiredBytes, 'gpu', 'scratch');
      this.reservation = reservation;
      const device = session.device;
      const allocate = (byteLength: number, label: string) => {
        const buffer = device.createBuffer({
          id: `plot-group-${label}`,
          byteLength: Math.max(4, byteLength),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        this.buffers.push(buffer);
        return buffer;
      };
      this.firstBins = allocate(labels.length * 4, 'first-bins');
      this.firstRows = allocate(labels.length * 4, 'first-rows');
      this.slotMap = allocate(labels.length * 4, 'slot-map');
      this.counts = allocate(keyCount * 4, 'counts');
      const composite = new GPUVector<'uint32'>({
        name: 'plot-group-keys',
        type: 'data',
        format: 'uint32',
        ownsData: true,
        data: codes.data.map(
          chunk =>
            new GPUData({
              buffer: allocate(chunk.length * 4, 'composite'),
              format: 'uint32',
              length: chunk.length,
              ownsBuffer: true
            })
        )
      });
      this.composite = composite;
      const validity = source.analytics.validity[grouping.column];
      this.firstSeen = this.makeFirstSeenGraph(codes, validity, labels.length);
      compiled.push(this.firstSeen);
      this.assignment = this.makeAssignmentGraph(codes, validity, keyCount);
      compiled.push(this.assignment);
      if (props.measure)
        this.measure = makeResidentPlotMeasure({
          session,
          lease: this.lease,
          keys: this.composite,
          mask: props.mask,
          binCount: keyCount,
          column: props.measure.column,
          operation: props.measure.operation,
          workingMemoryBudgetBytes:
            props.measureWorkingMemoryBudgetBytes ?? props.workingMemoryBudgetBytes - requiredBytes
        });
      const accountedBytes =
        this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
        compiled.reduce((total, graph) => total + graph.stats.physicalTransientResourceBytes, 0);
      if (accountedBytes > requiredBytes)
        throw new Error('Grouped line chart exceeded reserved working memory');
    } catch (error) {
      this.measure?.destroy();
      compiled.forEach(graph => graph.destroy());
      this.buffers.forEach(buffer => buffer.destroy());
      reservation?.release();
      void this.lease.release();
      throw error;
    }
  }

  get readbackByteLength(): number {
    return this.transferredBytes;
  }

  /** Phase one: earliest bin, then earliest row within that bin, per dictionary label. */
  encode(encoder: CommandEncoder): void {
    this.firstSeen.encode(encoder, {parameters: undefined});
  }

  /**
   * Phase two. Returns null when the caller's population changed (`isCurrent` false) before or
   * after the small first-seen tables were read.
   */
  async readSummary(isCurrent: () => boolean): Promise<ResidentPlotGroupingSummary | null> {
    const labelCount = this.labels.length;
    const [binBytes, rowBytes] = await Promise.all([
      this.firstBins.readAsync(0, labelCount * 4),
      this.firstRows.readAsync(0, labelCount * 4)
    ]);
    this.transferredBytes += labelCount * 8;
    if (!isCurrent()) return null;
    const firstBin = new Uint32Array(
      binBytes.buffer.slice(binBytes.byteOffset, binBytes.byteOffset + labelCount * 4)
    );
    const firstRow = new Uint32Array(
      rowBytes.buffer.slice(rowBytes.byteOffset, rowBytes.byteOffset + labelCount * 4)
    );
    // Native groups skip null and empty labels, and walk bins then rows, so rank by (bin, row).
    const ranked = Array.from({length: labelCount}, (_, code) => code)
      .filter(code => firstBin[code] !== NO_KEY && this.labels[code] !== '')
      .sort((left, right) => firstBin[left] - firstBin[right] || firstRow[left] - firstRow[right]);
    const kept = ranked.slice(0, this.limit);
    const hasOthers = this.props.grouping.groupOthers && ranked.length > kept.length;
    const slots = new Uint32Array(labelCount).fill(NO_KEY);
    kept.forEach((code, slot) => {
      slots[code] = slot;
    });
    if (hasOthers) for (const code of ranked.slice(kept.length)) slots[code] = kept.length;
    this.slotMap.write(slots);
    const encoder = this.props.session.device.createCommandEncoder();
    this.assignment.encode(encoder, {parameters: undefined});
    this.measure?.encode(encoder);
    this.props.session.device.submit(encoder.finish());
    const keyCount = this.slotCapacity * this.props.binCount;
    this.transferredBytes += keyCount * 4 + (this.measure?.readbackByteLength ?? 0);
    const [countBytes, measure] = await Promise.all([
      this.counts.readAsync(0, keyCount * 4),
      this.measure?.readSummary()
    ]);
    if (!isCurrent()) return null;
    return {
      labels: kept.map(code => this.labels[code]),
      others: hasOthers,
      counts: new Uint32Array(
        countBytes.buffer.slice(countBytes.byteOffset, countBytes.byteOffset + keyCount * 4)
      ),
      ...(measure ? {measure} : {})
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.measure?.destroy();
    this.firstSeen.destroy();
    this.assignment.destroy();
    this.buffers.forEach(buffer => buffer.destroy());
    this.reservation.release();
    void this.lease.release();
  }

  private makeFirstSeenGraph(
    codes: GPUVector,
    validity: GPUVector<'uint32'> | undefined,
    labelCount: number
  ): CompiledGPUCommandGraph {
    const device = this.props.session.device;
    const graph = new GPUCommandGraph(device, {id: 'plot-group-first-seen'});
    const view = (buffer: Buffer, id: string) =>
      graph.createDataView(
        graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
        {format: 'uint32', length: labelCount}
      );
    const firstBins = view(this.firstBins, 'first-bins');
    const firstRows = view(this.firstRows, 'first-rows');
    graph.addComputePass({
      id: 'initialize',
      resources: [
        {buffer: firstBins, usage: 'storage-write'},
        {buffer: firstRows, usage: 'storage-write'}
      ],
      compile: ({device: compileDevice}) => {
        const kernel = new Kernel(compileDevice, {
          source: `@group(0) @binding(0) var<storage,read_write> bins: array<u32>;
@group(0) @binding(1) var<storage,read_write> rows: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {if (invocation.x < ${labelCount}u) {bins[invocation.x] = 0xffffffffu; rows[invocation.x] = 0xffffffffu;}}`,
          shaderLayout: {
            bindings: ['bins', 'rows'].map((name, location) => ({
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
              bindings: {bins: getBuffer(firstBins.buffer), rows: getBuffer(firstRows.buffer)},
              x: Math.ceil(labelCount / 256)
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
    const keys = graph.importGPUVector('keys', this.props.keys);
    const masks = graph.importGPUVector('selected', this.props.mask);
    const codeVector = graph.importGPUVector('codes', codes);
    const valid = validity ? graph.importGPUVector('validity', validity) : undefined;
    // Pass one: earliest bin. Pass two: earliest row among the rows of that bin.
    for (const phase of ['bin', 'row'] as const) {
      let base = 0;
      for (const [chunkIndex, codeChunk] of codeVector.data.entries()) {
        const offset = base;
        base += codeChunk.length;
        if (!codeChunk.length) continue;
        const keyChunk = keys.data[chunkIndex];
        const maskChunk = masks.data[chunkIndex];
        const validChunk = valid?.data[chunkIndex];
        const dispatch = getBoundedDispatchLayout(
          `PlotGroupFirst${phase}`,
          codeChunk.length,
          256,
          device.limits.maxComputeWorkgroupsPerDimension
        );
        graph.addComputePass({
          id: `first-${phase}-${chunkIndex}`,
          resources: [
            {buffer: codeChunk, usage: 'storage-read'},
            {buffer: keyChunk, usage: 'storage-read'},
            {buffer: maskChunk, usage: 'storage-read'},
            ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
            {buffer: firstBins, usage: 'storage-read-write'},
            {buffer: firstRows, usage: 'storage-read-write'}
          ],
          compile: ({device: compileDevice}) => {
            const kernel = new Kernel(compileDevice, {
              source: `
@group(0) @binding(0) var<storage,read> codes: array<u32>;
@group(0) @binding(1) var<storage,read> keys: array<u32>;
@group(0) @binding(2) var<storage,read> selected: array<u32>;
@group(0) @binding(3) var<storage,read_write> bins: array<atomic<u32>>;
${phase === 'row' ? '@group(0) @binding(4) var<storage,read_write> rows: array<atomic<u32>>;' : ''}
${validChunk ? `@group(0) @binding(${phase === 'row' ? 5 : 4}) var<storage,read> validity: array<u32>;` : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
${getBoundedInvocationIndexSource(dispatch, 256)}
if (index >= ${codeChunk.length}u || selected[${maskChunk.byteOffset / 4}u + index] == 0u) {return;}
${validChunk ? `if (validity[${validChunk.byteOffset / 4}u + index] == 0u) {return;}` : ''}
let code = codes[${codeChunk.byteOffset / 4}u + index];
let bin = keys[${keyChunk.byteOffset / 4}u + index];
if (code >= ${labelCount}u || bin == 0xffffffffu) {return;}
${
  phase === 'bin'
    ? 'atomicMin(&bins[code], bin);'
    : `if (bin == atomicLoad(&bins[code])) {atomicMin(&rows[code], ${offset}u + index);}`
}
}`,
              shaderLayout: {
                bindings: [
                  'codes',
                  'keys',
                  'selected',
                  'bins',
                  ...(phase === 'row' ? ['rows'] : []),
                  ...(validChunk ? ['validity'] : [])
                ].map((name, location) => ({name, location, group: 0, type: 'storage' as const}))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    codes: getBuffer(codeChunk.buffer),
                    keys: getBuffer(keyChunk.buffer),
                    selected: getBuffer(maskChunk.buffer),
                    bins: getBuffer(firstBins.buffer),
                    ...(phase === 'row' ? {rows: getBuffer(firstRows.buffer)} : {}),
                    ...(validChunk ? {validity: getBuffer(validChunk.buffer)} : {})
                  },
                  ...dispatch
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      }
    }
    return graph.compile();
  }

  private makeAssignmentGraph(
    codes: GPUVector,
    validity: GPUVector<'uint32'> | undefined,
    keyCount: number
  ): CompiledGPUCommandGraph {
    const device = this.props.session.device;
    const graph = new GPUCommandGraph(device, {id: 'plot-group-assignment'});
    const slotMap = graph.createDataView(
      graph.importBuffer(
        {id: 'slot-map', byteLength: this.slotMap.byteLength, usage: this.slotMap.usage},
        this.slotMap
      ),
      {format: 'uint32', length: this.labels.length}
    );
    const counts = graph.createDataView(
      graph.importBuffer(
        {id: 'counts', byteLength: this.counts.byteLength, usage: this.counts.usage},
        this.counts
      ),
      {format: 'uint32', length: keyCount}
    );
    const keys = graph.importGPUVector('keys', this.props.keys);
    const output = graph.importGPUVector('composite', this.composite);
    const codeVector = graph.importGPUVector('codes', codes);
    const valid = validity ? graph.importGPUVector('validity', validity) : undefined;
    const binCount = this.props.binCount;
    for (const [chunkIndex, codeChunk] of codeVector.data.entries()) {
      if (!codeChunk.length) continue;
      const keyChunk = keys.data[chunkIndex];
      const outputChunk = output.data[chunkIndex];
      const validChunk = valid?.data[chunkIndex];
      const dispatch = getBoundedDispatchLayout(
        'PlotGroupAssign',
        codeChunk.length,
        256,
        device.limits.maxComputeWorkgroupsPerDimension
      );
      graph.addComputePass({
        id: `assign-${chunkIndex}`,
        resources: [
          {buffer: codeChunk, usage: 'storage-read'},
          {buffer: keyChunk, usage: 'storage-read'},
          {buffer: slotMap, usage: 'storage-read'},
          ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
          {buffer: outputChunk, usage: 'storage-write'}
        ],
        compile: ({device: compileDevice}) => {
          const kernel = new Kernel(compileDevice, {
            source: `
@group(0) @binding(0) var<storage,read> codes: array<u32>;
@group(0) @binding(1) var<storage,read> keys: array<u32>;
@group(0) @binding(2) var<storage,read> slots: array<u32>;
@group(0) @binding(3) var<storage,read_write> output: array<u32>;
${validChunk ? '@group(0) @binding(4) var<storage,read> validity: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
${getBoundedInvocationIndexSource(dispatch, 256)}
if (index >= ${codeChunk.length}u) {return;}
var key = 0xffffffffu;
let code = codes[${codeChunk.byteOffset / 4}u + index];
let bin = keys[${keyChunk.byteOffset / 4}u + index];
var present = ${validChunk ? `validity[${validChunk.byteOffset / 4}u + index] != 0u` : 'true'};
if (present && code < ${this.labels.length}u && bin != 0xffffffffu) {
  let slot = slots[code];
  if (slot != 0xffffffffu) {key = slot * ${binCount}u + bin;}
}
output[${outputChunk.byteOffset / 4}u + index] = key;
}`,
            shaderLayout: {
              bindings: [
                'codes',
                'keys',
                'slots',
                'output',
                ...(validChunk ? ['validity'] : [])
              ].map((name, location) => ({name, location, group: 0, type: 'storage' as const}))
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {
                  codes: getBuffer(codeChunk.buffer),
                  keys: getBuffer(keyChunk.buffer),
                  slots: getBuffer(slotMap.buffer),
                  output: getBuffer(outputChunk.buffer),
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
        input: output,
        output: counts,
        domain: [0, keyCount],
        mask: graph.importGPUVector('selected', this.props.mask)
      })
    );
    return graph.compile();
  }
}
