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
import {
  GPUDataFrame,
  column,
  literal,
  type CompiledGPUDataFrameAggregation,
  type GPUDataFrameQueryParameters
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerMemoryReservation,
  KeplerSessionExplorer
} from '@luma-contributors/kepler';
import {KeplerAllocationError} from '@luma-contributors/kepler';
import {
  ResidentCategoryChartRuntime,
  type ResidentCategoryChartResult,
  type ResidentChartSelection
} from './resident-chart-group-runtime';
import {MAXIMUM_RESIDENT_GROUPS} from './resident-chart-group-limits';
import {
  MAXIMUM_RESIDENT_CHART_CELLS,
  getResidentNumericGroupCount,
  type ResidentNumericChartPlan
} from './resident-chart-plan';
import {
  decodeResidentNumericOrder,
  encodeResidentNumericThreshold,
  makeResidentNumericBoundaries
} from './resident-numeric-domain';

import {ResidentIntegerChartDomain, INTEGER_DOMAIN_READ_BYTES} from './resident-integer-domain';
import {MeteredCommandGraph} from '@luma-contributors/kepler';

export type ResidentNumericChartResult = Omit<ResidentCategoryChartResult, 'kind'> & {
  kind: 'numeric';
  boundaries: readonly number[];
  categoryValues?: readonly number[];
};
type ExtentColumns = {ordered: 'uint32'; selected: 'uint32'};

/** Two bounded phases: exact source-format extrema, then bins over the same source/selection revision. */
export class ResidentNumericChartRuntime {
  private readonly device: Device;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly extent: CompiledGPUDataFrameAggregation<{minimum: 'uint32'; maximum: 'uint32'}>;
  private readonly prepare: CompiledGPUCommandGraph;
  private readonly binning: CompiledGPUCommandGraph;
  private readonly vectors: readonly GPUVector<'uint32'>[];
  private readonly thresholds: Buffer;
  private readonly integerDomain?: ResidentIntegerChartDomain;
  private readonly grouped: ResidentCategoryChartRuntime;
  private tail: Promise<void> = Promise.resolve();
  private requestRevision = 0;
  private destroyed = false;
  private destruction?: Promise<void>;
  private transferredBytes = 0;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      selection?: ResidentChartSelection;
      plan: ResidentNumericChartPlan;
      workingMemoryBudgetBytes: number;
      /** Exact Float32 stand-in for an Int32 or Uint32 measure column. */
      measure?: GPUVector<'float32'>;
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
    this.device = props.session.device;
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.plan.datasetId);
    const vectors: GPUVector<'uint32'>[] = [];
    let reservation: KeplerMemoryReservation | undefined;
    let extent: CompiledGPUDataFrameAggregation<{minimum: 'uint32'; maximum: 'uint32'}> | undefined;
    let prepare: CompiledGPUCommandGraph | undefined;
    let binning: CompiledGPUCommandGraph | undefined;
    let thresholds: Buffer | undefined;
    let integerDomain: ResidentIntegerChartDomain | undefined;
    let grouped: ResidentCategoryChartRuntime | undefined;
    let frame: GPUDataFrame<ExtentColumns> | undefined;
    let ownsScope = false;
    try {
      const {plan} = props;
      if (
        this.lease.dataset.id !== props.explorer.dataset.id ||
        this.lease.dataset.revision !== props.explorer.dataset.revision
      )
        throw new Error('Numeric chart and explorer require the same source revision');
      const source = this.lease.dataset.value;
      const values = source.columns[plan.binColumn];
      if (
        !values ||
        values.format !== plan.binFormat ||
        source.prepared.dictionaries[plan.binColumn] ||
        !Number.isSafeInteger(plan.binCount) ||
        plan.binCount < 2 ||
        plan.binCount > MAXIMUM_RESIDENT_GROUPS
      )
        throw new Error('Numeric chart requires a supported source and bounded bin count');
      const groupCount = getResidentNumericGroupCount(plan);
      if (
        groupCount * (plan.series ? plan.series.categoryCount + 1 : 1) >
        MAXIMUM_RESIDENT_CHART_CELLS
      )
        throw new KeplerAllocationError('memory-budget', {
          message: 'Chart series exceed the bounded cell budget before allocation'
        });
      const ownEstimatedBytes =
        65536 + values.length * 160 + values.data.length * 8192 + groupCount * 128;
      const groupEstimatedBytes =
        65536 +
        values.length * 160 +
        values.data.length * 8192 +
        groupCount * (plan.series ? plan.series.categoryCount + 1 : 1) * 128;
      if (
        !Number.isSafeInteger(props.workingMemoryBudgetBytes) ||
        ownEstimatedBytes + groupEstimatedBytes > props.workingMemoryBudgetBytes
      )
        throw new KeplerAllocationError('working-memory', {
          message: 'Numeric chart exceeds its working memory budget before allocation'
        });
      reservation = props.session.registry.reserveMemory(ownEstimatedBytes, 'gpu', 'scratch');
      const ordered = allocateVector(this.device, 'numeric-ordered', values);
      vectors.push(ordered);
      const finite = allocateVector(this.device, 'numeric-finite', values);
      vectors.push(finite);
      const keys = allocateVector(this.device, 'numeric-bins', values);
      vectors.push(keys);
      prepare = makePrepareGraph(
        this.device,
        values,
        source.analytics.validity[plan.binColumn],
        ordered,
        finite,
        plan.binFormat
      );
      thresholds = this.device.createBuffer({
        id: 'numeric-chart-thresholds',
        byteLength: (groupCount + 2) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
      binning = makeBinningGraph(
        this.device,
        ordered,
        finite,
        keys,
        thresholds,
        source.analytics.validity[plan.binColumn]
      );
      if (plan.population === 'selected' && !props.selection)
        throw new Error('Numeric chart selection scopes are unavailable');
      const selected =
        plan.population === 'selected'
          ? props.selection!.getConsumerMask({
              id: `chart:${plan.chartId}`,
              excludedCpuFilterIds: plan.excludedCpuFilterIds,
              excludedGpuFieldNames: plan.excludedGpuFieldNames
            })
          : props.explorer.selectionMask;
      ownsScope = plan.population === 'selected';
      if (plan.integerGroupLimit !== undefined)
        integerDomain = new ResidentIntegerChartDomain({
          device: this.device,
          ordered,
          validity: source.analytics.validity[plan.binColumn],
          selected: plan.population === 'selected' ? selected : undefined
        });
      const table = new GPUTable<ExtentColumns>({
        vectors: {
          ordered: new GPUVector({
            name: 'ordered',
            type: 'data',
            format: 'uint32',
            data: ordered.data,
            ownsData: false
          }),
          selected: new GPUVector({
            name: 'selected',
            type: 'data',
            format: 'uint32',
            data: selected.data,
            ownsData: false
          })
        }
      });
      for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
        const field = schema.fields.find(candidate => candidate.name === 'ordered');
        if (field) field.nullable = true;
      }
      frame = new GPUDataFrame({table, validity: {ordered: finite}});
      const graph = new MeteredCommandGraph<GPUDataFrameQueryParameters>(this.device, {
        id: `numeric-chart-extent-${plan.chartId}`
      });
      extent = frame
        .filter(
          plan.population === 'selected' ? column('selected').notEqual(literal(0)) : literal(true)
        )
        .aggregate({minimum: {min: 'ordered'}, maximum: {max: 'ordered'}})
        .compile(graph);
      const buffers = new Set<Buffer>([thresholds]);
      for (const vector of [
        ...vectors,
        ...Object.values(extent.table.gpuVectors),
        ...Object.values(extent.validity),
        extent.selectionMask,
        extent.rowIndices,
        extent.selectedCounts
      ])
        if (vector) for (const data of vector.data) buffers.add(getBuffer(data));
      const accountedBytes =
        [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0) +
        graph.transientByteLength +
        prepare.stats.physicalTransientResourceBytes +
        binning.stats.physicalTransientResourceBytes +
        (integerDomain?.byteLength ?? 0);
      if (accountedBytes > ownEstimatedBytes)
        throw new KeplerAllocationError('working-memory', {
          message: 'Numeric chart exceeds its reserved working memory'
        });
      grouped = new ResidentCategoryChartRuntime({
        ...props,
        workingMemoryBudgetBytes: props.workingMemoryBudgetBytes - ownEstimatedBytes,
        plan: {
          kind: 'category',
          chartId: plan.chartId,
          datasetId: plan.datasetId,
          population: plan.population,
          operation: plan.operation,
          ...(plan.column ? {column: plan.column} : {}),
          categoryColumn: plan.binColumn,
          categoryCount: groupCount,
          series: plan.series,
          excludedCpuFilterIds: plan.excludedCpuFilterIds,
          excludedGpuFieldNames: plan.excludedGpuFieldNames
        },
        groupingInput: {
          keys,
          labels: Array.from({length: groupCount}, (_, index) => String(index))
        }
      });
      this.reservation = reservation;
      this.extent = extent;
      this.prepare = prepare;
      this.binning = binning;
      this.thresholds = thresholds;
      this.integerDomain = integerDomain;
      this.vectors = vectors;
      this.grouped = grouped;
    } catch (error) {
      // Construction has submitted no GPU work. A failed grouped constructor already retires its scope.
      integerDomain?.destroy();
      extent?.destroy();
      prepare?.destroy();
      binning?.destroy();
      thresholds?.destroy();
      for (const vector of vectors) vector.destroy();
      if (ownsScope) props.selection?.removeConsumerMask(`chart:${props.plan.chartId}`);
      reservation?.release();
      void this.lease.release();
      throw error;
    } finally {
      frame?.destroy();
    }
  }
  get readbackByteLength(): number {
    return this.transferredBytes + this.grouped.readbackByteLength;
  }
  cancel(): void {
    this.requestRevision++;
    this.grouped.cancel();
  }
  read(): Promise<ResidentNumericChartResult | null> {
    const requestRevision = ++this.requestRevision;
    const selectionRevision =
      this.props.plan.population === 'selected' ? this.props.selection!.currentRevision : null;
    const pending = this.tail.then(() => this.readCurrent(requestRevision, selectionRevision));
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.cancel();
    this.destruction = (async () => {
      await this.tail;
      let fence: ReturnType<Device['createFence']> | undefined;
      try {
        if (!this.device.isLost) {
          fence = this.device.createFence();
          await Promise.race([fence.signaled, this.device.lost]);
        }
      } finally {
        fence?.destroy();
        try {
          this.integerDomain?.destroy();
          this.extent.destroy();
          this.prepare.destroy();
          this.binning.destroy();
          await this.grouped.destroy();
        } finally {
          this.thresholds.destroy();
          for (const vector of this.vectors) vector.destroy();
          this.reservation.release();
          await this.lease.release();
        }
      }
    })();
    return this.destruction;
  }
  private isCurrent(requestRevision: number, selectionRevision: number | null): boolean {
    return (
      !this.destroyed &&
      !this.props.session.deviceLost &&
      this.lease.isValid() &&
      this.props.explorer.isCurrent &&
      requestRevision === this.requestRevision &&
      (selectionRevision === null ||
        (!this.props.selection!.isLayerRegistrationPending &&
          this.props.selection!.currentRevision === selectionRevision))
    );
  }
  private async readCurrent(
    requestRevision: number,
    selectionRevision: number | null
  ): Promise<ResidentNumericChartResult | null> {
    if (!this.isCurrent(requestRevision, selectionRevision)) return null;
    const reservation = this.props.session.registry.reserveMemory(
      12 +
        this.props.plan.binCount * 12 +
        (this.integerDomain ? INTEGER_DOMAIN_READ_BYTES + MAXIMUM_RESIDENT_GROUPS * 32 : 0),
      'cpu',
      'readback'
    );
    try {
      const encoder = this.device.createCommandEncoder();
      if (selectionRevision !== null) this.props.selection!.encode(encoder);
      if (
        selectionRevision !== null &&
        (!this.props.selection!.canConsume || !this.isCurrent(requestRevision, selectionRevision))
      ) {
        encoder.destroy();
        return null;
      }
      this.prepare.encode(encoder, {parameters: undefined});
      this.extent.encode(encoder);
      this.integerDomain?.encode(encoder);
      this.device.submit(encoder.finish());
      const minimumVector = this.extent.table.gpuVectors['minimum'];
      const maximumVector = this.extent.table.gpuVectors['maximum'];
      const validity = this.extent.validity.minimum!;
      const domainReadBytes = 12 + (this.integerDomain ? INTEGER_DOMAIN_READ_BYTES : 0);
      this.transferredBytes += domainReadBytes;
      const [minimumBytes, maximumBytes, validBytes, integerDomain] = await Promise.all([
        getBuffer(minimumVector.data[0]).readAsync(minimumVector.data[0].byteOffset, 4),
        getBuffer(maximumVector.data[0]).readAsync(maximumVector.data[0].byteOffset, 4),
        getBuffer(validity.data[0]).readAsync(validity.data[0].byteOffset, 4),
        this.integerDomain?.read()
      ]);
      if (
        !this.isCurrent(requestRevision, selectionRevision) ||
        (selectionRevision !== null && !this.props.selection!.canConsume)
      )
        return null;
      const valid = new DataView(validBytes.buffer, validBytes.byteOffset, 4).getUint32(0, true);
      const useUnique =
        integerDomain &&
        !(
          this.props.plan.integerGroupLimit! > 0 &&
          integerDomain.keys.length > this.props.plan.integerGroupLimit!
        );
      if (
        useUnique &&
        (integerDomain.overflow || integerDomain.keys.length > MAXIMUM_RESIDENT_GROUPS)
      )
        throw new Error(
          'Integer chart distinct values exceed the bounded summary budget; choose Numeric intervals or a smaller group limit.'
        );
      if (!valid && !useUnique)
        return {
          kind: 'numeric',
          summaries: [],
          boundaries: [],
          sourceRevision: this.lease.dataset.revision,
          selectionRevision,
          readbackBytes: domainReadBytes
        };
      const minimum = decodeResidentNumericOrder(
        new DataView(minimumBytes.buffer, minimumBytes.byteOffset, 4).getUint32(0, true),
        this.props.plan.binFormat
      );
      const maximum = decodeResidentNumericOrder(
        new DataView(maximumBytes.buffer, maximumBytes.byteOffset, 4).getUint32(0, true),
        this.props.plan.binFormat
      );
      const categoryValues = useUnique
        ? integerDomain.keys.map(key => decodeResidentNumericOrder(key, this.props.plan.binFormat))
        : undefined;
      const boundaries = useUnique
        ? []
        : makeResidentNumericBoundaries(minimum, maximum, this.props.plan.binCount);
      const words = new Uint32Array(this.thresholds.byteLength / 4);
      words[0] = useUnique ? 1 : 0;
      words[1] = useUnique ? integerDomain.keys.length : boundaries.length - 2;
      if (useUnique) words.set(integerDomain.keys, 2);
      else
        for (let index = 1; index < boundaries.length - 1; index++)
          words[index + 1] = encodeResidentNumericThreshold(
            boundaries[index],
            this.props.plan.binFormat
          );
      this.thresholds.write(words);
      const binEncoder = this.device.createCommandEncoder();
      this.binning.encode(binEncoder, {parameters: undefined});
      this.device.submit(binEncoder.finish());
      const grouped = await this.grouped.read();
      if (
        !grouped ||
        !this.isCurrent(requestRevision, selectionRevision) ||
        (selectionRevision !== null && !this.props.selection!.canConsume)
      )
        return null;
      return {
        ...grouped,
        kind: 'numeric',
        boundaries,
        ...(categoryValues ? {categoryValues} : {}),
        readbackBytes: grouped.readbackBytes + domainReadBytes
      };
    } catch (error) {
      if (!this.isCurrent(requestRevision, selectionRevision)) return null;
      throw error;
    } finally {
      reservation.release();
    }
  }
}

export function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}
export function allocateVector(
  device: Device,
  name: string,
  source: GPUVector
): GPUVector<'uint32'> {
  const chunks: GPUData<'uint32'>[] = [];
  try {
    for (const chunk of source.data)
      chunks.push(
        new GPUData({
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            id: name,
            byteLength: Math.max(4, chunk.length * 4),
            usage: Buffer.STORAGE | Buffer.COPY_SRC
          })
        })
      );
    return new GPUVector({name, type: 'data', format: 'uint32', data: chunks, ownsData: true});
  } catch (error) {
    for (const chunk of chunks) chunk.destroy();
    throw error;
  }
}

/** Sortable integer encodings preserve Float32 subnormals, signed zero, Int32 and Uint32 exactly. */
export function makePrepareGraph(
  device: Device,
  source: GPUVector,
  validity: GPUVector<'uint32'> | undefined,
  ordered: GPUVector<'uint32'>,
  finite: GPUVector<'uint32'>,
  format: ResidentNumericChartPlan['binFormat']
): CompiledGPUCommandGraph {
  const graph = new GPUCommandGraph(device, {id: 'numeric-chart-prepare'});
  const values = graph.importGPUVector('values', source);
  const valid = validity ? graph.importGPUVector('valid', validity) : undefined;
  const orders = graph.importGPUVector('ordered', ordered);
  const masks = graph.importGPUVector('finite', finite);
  for (const [index, input] of values.data.entries()) {
    if (!input.length) continue;
    const output = orders.data[index];
    const mask = masks.data[index];
    const validChunk = valid?.data[index];
    const dispatch = getBoundedDispatchLayout(
      'NumericChartPrepare',
      input.length,
      256,
      device.limits.maxComputeWorkgroupsPerDimension
    );
    graph.addComputePass({
      id: `prepare-${index}`,
      resources: [
        {buffer: input, usage: 'storage-read'},
        ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
        {buffer: output, usage: 'storage-write'},
        {buffer: mask, usage: 'storage-write'}
      ],
      compile: ({device: compileDevice}) => {
        const kernel = new Kernel(compileDevice, {
          source: `
@group(0) @binding(0) var<storage,read> input: array<u32>;
@group(0) @binding(1) var<storage,read_write> output: array<u32>;
@group(0) @binding(2) var<storage,read_write> finite: array<u32>;
${validChunk ? '@group(0) @binding(3) var<storage,read> valid: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if(index >= ${input.length}u) {return;}
  var bits = input[${input.byteOffset / 4}u + index];
  var present = ${validChunk ? `valid[${validChunk.byteOffset / 4}u + index] != 0u` : 'true'};
  ${format === 'float32' ? 'present = present && (bits & 0x7fffffffu) < 0x7f800000u; if((bits & 0x7fffffffu) > 0x7f800000u) {bits = 0x7fc00000u;} if((bits & 0x7fffffffu) == 0u) {bits = 0u;} bits = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);' : format === 'sint32' ? 'bits = bits ^ 0x80000000u;' : ''}
  output[index] = bits; finite[index] = select(0u, 1u, present);
}`,
          shaderLayout: {
            bindings: ['input', 'output', 'finite', ...(validChunk ? ['valid'] : [])].map(
              (name, location) => ({name, location, group: 0, type: 'storage' as const})
            )
          }
        });
        return {
          encode: ({computePass, getBuffer: resolve}) =>
            kernel.dispatch(computePass, {
              bindings: {
                input: resolve(input.buffer),
                output: resolve(output.buffer),
                finite: resolve(mask.buffer),
                ...(validChunk ? {valid: resolve(validChunk.buffer)} : {})
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

export function makeBinningGraph(
  device: Device,
  ordered: GPUVector<'uint32'>,
  finite: GPUVector<'uint32'>,
  keys: GPUVector<'uint32'>,
  thresholds: Buffer,
  validity?: GPUVector<'uint32'>
): CompiledGPUCommandGraph {
  const graph = new GPUCommandGraph(device, {id: 'numeric-chart-binning'});
  const inputs = graph.importGPUVector('ordered', ordered);
  const masks = graph.importGPUVector('finite', finite);
  const outputs = graph.importGPUVector('keys', keys);
  const valid = validity ? graph.importGPUVector('valid', validity) : undefined;
  const edges = graph.createDataView(
    graph.importBuffer(
      {id: 'thresholds', byteLength: thresholds.byteLength, usage: thresholds.usage},
      thresholds
    ),
    {format: 'uint32', length: thresholds.byteLength / 4}
  );
  for (const [index, input] of inputs.data.entries()) {
    if (!input.length) continue;
    const mask = masks.data[index];
    const output = outputs.data[index];
    const validChunk = valid?.data[index];
    const dispatch = getBoundedDispatchLayout(
      'NumericChartBins',
      input.length,
      256,
      device.limits.maxComputeWorkgroupsPerDimension
    );
    graph.addComputePass({
      id: `bin-${index}`,
      resources: [
        {buffer: input, usage: 'storage-read'},
        {buffer: mask, usage: 'storage-read'},
        {buffer: edges, usage: 'storage-read'},
        ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
        {buffer: output, usage: 'storage-write'}
      ],
      compile: ({device: compileDevice}) => {
        const kernel = new Kernel(compileDevice, {
          source: `
@group(0) @binding(0) var<storage,read> input: array<u32>;
@group(0) @binding(1) var<storage,read> finite: array<u32>;
@group(0) @binding(2) var<storage,read> edges: array<u32>;
@group(0) @binding(3) var<storage,read_write> output: array<u32>;
${validChunk ? '@group(0) @binding(4) var<storage,read> valid: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if(index >= ${input.length}u) {return;}
  let unique = edges[0] == 1u;
  let present = select(finite[index] != 0u, ${validChunk ? `valid[${validChunk.byteOffset / 4}u + index] != 0u` : 'true'}, unique);
  if(!present) {output[index] = 0xffffffffu; return;}
  let value = input[index]; var lower = 0u; var upper = edges[1];
  loop {if(lower >= upper) {break;} let middle = (lower + upper) / 2u; let edge = edges[middle + 2u]; if(select(value >= edge, value > edge, unique)) {lower = middle + 1u;} else {upper = middle;}}
  if(unique) {if(lower >= edges[1]) {output[index] = 0xffffffffu; return;} if(edges[lower + 2u] != value) {output[index] = 0xffffffffu; return;}}
  output[index] = lower;
}`,
          shaderLayout: {
            bindings: ['input', 'finite', 'edges', 'output', ...(validChunk ? ['valid'] : [])].map(
              (name, location) => ({
                name,
                location,
                group: 0,
                type: 'storage' as const
              })
            )
          }
        });
        return {
          encode: ({computePass, getBuffer: resolve}) =>
            kernel.dispatch(computePass, {
              bindings: {
                input: resolve(input.buffer),
                finite: resolve(mask.buffer),
                edges: resolve(edges.buffer),
                output: resolve(output.buffer),
                ...(validChunk ? {valid: resolve(validChunk.buffer)} : {})
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
