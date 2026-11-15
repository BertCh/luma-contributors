// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {GPUVector, type GPUData} from '@luma.gl/graph/gpu-data';
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
  type CompiledGPUDataFrameGroupedAggregation,
  type GPUDataFrameQueryParameters,
  type GPUDataFrameAggregationValue
} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerSessionExplorer,
  KeplerDatasetLease,
  KeplerMemoryReservation
} from '@luma-contributors/kepler';
import {
  MAXIMUM_RESIDENT_CHART_CATEGORIES,
  MAXIMUM_RESIDENT_CHART_CELLS,
  MAXIMUM_RESIDENT_TIME_GROUPS,
  isResidentChartOrderOperation,
  type ResidentCategoryChartPlan
} from './resident-chart-plan';
import {ResidentPlotOrderMeasure} from './resident-plot-order-measure';
import type {ResidentCategorySummary} from './resident-chart-bins';
import {ResidentSeriesKeys} from './resident-series-keys';
import {foldResidentChartSeries} from './resident-chart-series';
import type {ResidentScopeBinding} from './resident-scope-binding';
import {MeteredCommandGraph} from '@luma-contributors/kepler';

export type ResidentChartSelection = Pick<
  ResidentScopeBinding,
  | 'getConsumerMask'
  | 'removeConsumerMask'
  | 'encode'
  | 'currentRevision'
  | 'isLayerRegistrationPending'
  | 'canConsume'
>;
type GroupColumns = {category: 'uint32'; selected: 'uint32'; value: 'float32'};
type GroupQuery =
  | CompiledGPUDataFrameGroupedAggregation<{category: 'uint32'; result: 'float32'; count: 'uint32'}>
  | CompiledGPUDataFrameGroupedAggregation<{category: 'uint32'; result: 'uint32'; count: 'uint32'}>;
export type ResidentCategoryChartResult = {
  kind: 'category';
  summaries: ResidentCategorySummary[];
  sourceRevision: number;
  selectionRevision: number | null;
  readbackBytes: number;
};

/** Bounded dense dictionary reductions, retaining source chunks and exact source-order ties. */
export class ResidentCategoryChartRuntime {
  private readonly device: Device;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation: KeplerMemoryReservation;
  private readonly query: GroupQuery;
  private readonly orderMeasure?: ResidentPlotOrderMeasure;
  private readonly orderGraph: CompiledGPUCommandGraph;
  private readonly firstRows: Buffer;
  private readonly labels: readonly string[];
  private readonly groupCount: number;
  private readonly seriesLabels?: readonly string[];
  private readonly seriesKeys?: ResidentSeriesKeys;
  private readonly scopeId: string;
  private tail: Promise<void> = Promise.resolve();
  private requestRevision = 0;
  private destruction?: Promise<void>;
  private destroyed = false;
  private transferredBytes = 0;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      plan: ResidentCategoryChartPlan;
      selection?: ResidentChartSelection;
      workingMemoryBudgetBytes: number;
      /** Exact Float32 stand-in for an Int32 or Uint32 measure column. */
      measure?: GPUVector<'float32'>;
      /** Explicit derived keys owned by a numeric-bin adapter; retain until this reader is destroyed. */
      groupingInput?: {keys: GPUVector<'uint32'>; labels: readonly string[]};
      /** Raised group/cell ceiling for the top-N counting pass over a very large dictionary (no series). */
      maximumGroups?: number;
      /** Derived series keys (numeric-series bins) replacing the dictionary series column; invalid is 0xffffffff. */
      seriesInput?: {keys: GPUVector<'uint32'>; labels: readonly string[]};
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
    this.scopeId = `chart:${props.plan.chartId}`;
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.plan.datasetId);
    let frame: GPUDataFrame<GroupColumns> | undefined;
    let reservation: KeplerMemoryReservation | undefined;
    let query: GroupQuery | undefined;
    let firstRows: Buffer | undefined;
    let orderGraph: CompiledGPUCommandGraph | undefined;
    let ownsScope = false;
    let orderMeasure: ResidentPlotOrderMeasure | undefined;
    let seriesKeys: ResidentSeriesKeys | undefined;
    try {
      if (
        this.lease.dataset.id !== props.explorer.dataset.id ||
        this.lease.dataset.revision !== props.explorer.dataset.revision
      )
        throw new Error('Chart and explorer require the same source revision');
      const source = this.lease.dataset.value;
      let category = props.groupingInput?.keys ?? source.columns[props.plan.categoryColumn];
      const labels =
        props.groupingInput?.labels ??
        source.prepared.dictionaries[props.plan.categoryColumn]?.values;
      if (
        !category ||
        category.format !== 'uint32' ||
        labels?.length !== props.plan.categoryCount ||
        labels.length < 1 ||
        labels.length > (props.maximumGroups ?? MAXIMUM_RESIDENT_TIME_GROUPS) ||
        category.length > 0xffffffff
      )
        throw new Error('Chart dictionary changed or is unavailable');
      this.labels = labels;
      const seriesPlan = props.plan.series;
      const series = seriesPlan
        ? (props.seriesInput?.keys ?? source.columns[seriesPlan.column])
        : undefined;
      const seriesLabels = seriesPlan
        ? (props.seriesInput?.labels ?? source.prepared.dictionaries[seriesPlan.column]?.values)
        : undefined;
      if (
        seriesPlan &&
        (!series ||
          series.format !== 'uint32' ||
          seriesLabels?.length !== seriesPlan.categoryCount ||
          !seriesLabels.length ||
          seriesLabels.length > MAXIMUM_RESIDENT_CHART_CATEGORIES)
      )
        throw new Error('Chart series dictionary changed or is unavailable');
      const groupCount = labels.length * (seriesLabels ? seriesLabels.length + 1 : 1);
      if (groupCount > (props.maximumGroups ?? MAXIMUM_RESIDENT_CHART_CELLS))
        throw new Error('Chart series exceed the bounded cell budget before allocation');
      this.groupCount = groupCount;
      this.seriesLabels = seriesLabels;
      const estimatedBytes =
        65536 + category.length * 160 + category.data.length * 8192 + groupCount * 128;
      if (
        !Number.isSafeInteger(props.workingMemoryBudgetBytes) ||
        estimatedBytes > props.workingMemoryBudgetBytes
      )
        throw new Error('Category chart exceeds its working memory budget before allocation');
      reservation = props.session.registry.reserveMemory(estimatedBytes, 'gpu', 'scratch');
      if (seriesPlan && series?.format === 'uint32' && seriesLabels) {
        seriesKeys = new ResidentSeriesKeys({
          device: this.device,
          categories: category,
          categoryCount: labels.length,
          categoryValidity: props.groupingInput
            ? undefined
            : source.analytics.validity[props.plan.categoryColumn],
          series,
          seriesLabels,
          seriesValidity: props.seriesInput
            ? undefined
            : source.analytics.validity[seriesPlan.column]
        });
        category = seriesKeys.keys;
      }
      if (props.plan.population === 'selected' && !props.selection)
        throw new Error('Category chart selection scopes are unavailable');
      const selected =
        props.plan.population === 'selected'
          ? props.selection!.getConsumerMask({
              id: this.scopeId,
              excludedCpuFilterIds: props.plan.excludedCpuFilterIds,
              excludedGpuFieldNames: props.plan.excludedGpuFieldNames
            })
          : props.explorer.selectionMask;
      ownsScope = props.plan.population === 'selected';
      const vectors: Record<string, GPUVector> = {
        category: new GPUVector({
          name: 'category',
          type: 'data',
          format: 'uint32',
          data: category.data,
          ownsData: false
        }),
        selected: new GPUVector({
          name: 'selected',
          type: 'data',
          format: 'uint32',
          data: selected.data,
          ownsData: false
        })
      };
      const validity: {category?: GPUVector<'uint32'>; value?: GPUVector<'uint32'>} = {};
      const categoryValidity =
        props.groupingInput || seriesKeys
          ? undefined
          : source.analytics.validity[props.plan.categoryColumn];
      if (categoryValidity) validity.category = categoryValidity;
      const operation = props.plan.operation;
      const isOrder = isResidentChartOrderOperation(operation);
      if (props.plan.column && !isOrder) {
        const value = props.measure ?? source.columns[props.plan.column];
        // Integer measures arrive widened to exact Float32 from the provider.
        if (!value || value.format !== 'float32')
          throw new Error('Grouped numeric measures require Float32 values');
        vectors['value'] = new GPUVector({
          name: 'value',
          type: 'data',
          format: 'float32',
          data: value.data,
          ownsData: false
        });
        const valueValidity = source.analytics.validity[props.plan.column];
        if (valueValidity) validity.value = valueValidity;
      }
      const table = new GPUTable<GroupColumns>({vectors});
      for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)])
        for (const field of schema.fields)
          field.nullable =
            field.name === 'category'
              ? Boolean(validity.category)
              : field.name === 'value' && Boolean(validity.value);
      frame = new GPUDataFrame({table, validity});
      const graph = new MeteredCommandGraph<GPUDataFrameQueryParameters>(this.device, {
        id: `kepler-category-${props.plan.chartId}`
      });
      const result: GPUDataFrameAggregationValue<GroupColumns, keyof GroupColumns> =
        isOrder || props.plan.operation === 'count'
          ? 'count'
          : props.plan.operation === 'sum'
            ? {sum: 'value'}
            : props.plan.operation === 'mean'
              ? {mean: 'value'}
              : props.plan.operation === 'min'
                ? {min: 'value'}
                : {max: 'value'};
      const selectedFrame = frame.filter(
        props.plan.population === 'selected'
          ? column('selected').notEqual(literal(0))
          : literal(true)
      );
      query = selectedFrame
        .groupBy('category', {groupCount})
        .aggregate({result, count: 'count'})
        .compile(graph);
      firstRows = this.device.createBuffer({
        id: 'chart-first-rows',
        byteLength: groupCount * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC
      });
      orderGraph = makeFirstRowsGraph(
        this.device,
        category,
        query.selectionMask,
        validity.category,
        firstRows,
        groupCount
      );
      if (isOrder && props.plan.column) {
        // Exact ordered statistics share the chart's bin keys and selection, so membership agrees.
        orderMeasure = new ResidentPlotOrderMeasure({
          session: props.session,
          lease: this.lease,
          // Validated above: dictionary keys and derived bin keys are always Uint32.
          keys: category as GPUVector<'uint32'>,
          ...(props.plan.population === 'selected' ? {mask: selected} : {}),
          ...(categoryValidity ? {keyValidity: categoryValidity} : {}),
          ...(props.measure ? {values: props.measure} : {}),
          keyCount: groupCount,
          column: props.plan.column,
          operation,
          workingMemoryBudgetBytes: props.workingMemoryBudgetBytes - estimatedBytes
        });
      }
      const buffers = new Set<Buffer>([firstRows]);
      for (const vector of [
        ...Object.values(query.table.gpuVectors),
        ...Object.values(query.validity),
        query.selectionMask,
        query.rowIndices,
        query.selectedCounts
      ])
        if (vector) for (const data of vector.data) buffers.add(getBuffer(data));
      const accountedBytes =
        (seriesKeys?.byteLength ?? 0) +
        graph.transientByteLength +
        orderGraph.stats.physicalTransientResourceBytes +
        [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0);
      if (accountedBytes > estimatedBytes)
        throw new Error('Category chart exceeds its reserved working memory');
      this.seriesKeys = seriesKeys;
      this.orderMeasure = orderMeasure;
      this.query = query;
      this.orderGraph = orderGraph;
      this.firstRows = firstRows;
      this.reservation = reservation;
    } catch (error) {
      orderMeasure?.destroy();
      orderGraph?.destroy();
      firstRows?.destroy();
      query?.destroy();
      seriesKeys?.destroy();
      reservation?.release();
      if (ownsScope) props.selection?.removeConsumerMask(this.scopeId);
      void this.lease.release();
      throw error;
    } finally {
      frame?.destroy();
    }
  }

  get readbackByteLength(): number {
    return this.transferredBytes;
  }
  cancel(): void {
    this.requestRevision++;
  }
  read(): Promise<ResidentCategoryChartResult | null> {
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
        this.orderMeasure?.destroy();
        this.orderGraph.destroy();
        this.firstRows.destroy();
        this.query.destroy();
        this.seriesKeys?.destroy();
        if (this.props.plan.population === 'selected' && this.props.explorer.isCurrent)
          this.props.selection?.removeConsumerMask(this.scopeId);
        this.reservation.release();
        await this.lease.release();
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
          selectionRevision === this.props.selection!.currentRevision))
    );
  }
  private async readCurrent(
    requestRevision: number,
    selectionRevision: number | null
  ): Promise<ResidentCategoryChartResult | null> {
    if (!this.isCurrent(requestRevision, selectionRevision)) return null;
    const result = this.query.table.gpuVectors['result'];
    const count = this.query.table.gpuVectors['count'];
    const validity = this.query.validity.result;
    const byteLength = this.groupCount * 4;
    const readbackBytes =
      byteLength * (validity ? 4 : 3) + (this.orderMeasure?.readbackByteLength ?? 0);
    const reservation = this.props.session.registry.reserveMemory(readbackBytes, 'cpu', 'readback');
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
      this.seriesKeys?.encode(encoder);
      this.query.encode(encoder);
      this.orderGraph.encode(encoder, {parameters: undefined});
      this.orderMeasure?.encode(encoder);
      this.device.submit(encoder.finish());
      this.transferredBytes += readbackBytes;
      const [values, counts, firstRows, valid, orderSummary] = await Promise.all([
        getBuffer(result.data[0]).readAsync(result.data[0].byteOffset, byteLength),
        getBuffer(count.data[0]).readAsync(count.data[0].byteOffset, byteLength),
        this.firstRows.readAsync(0, byteLength),
        validity
          ? getBuffer(validity.data[0]).readAsync(validity.data[0].byteOffset, byteLength)
          : undefined,
        this.orderMeasure?.readSummary()
      ]);
      if (
        !this.isCurrent(requestRevision, selectionRevision) ||
        (selectionRevision !== null && !this.props.selection!.canConsume)
      )
        return null;
      const valueView = new DataView(values.buffer, values.byteOffset, byteLength);
      const countView = new DataView(counts.buffer, counts.byteOffset, byteLength);
      const firstView = new DataView(firstRows.buffer, firstRows.byteOffset, byteLength);
      const validView = valid
        ? new DataView(valid.buffer, valid.byteOffset, byteLength)
        : undefined;
      const cells = Array.from({length: this.groupCount}, (_, index) => {
        const category = this.seriesLabels ? String(index) : this.labels[index];
        const value = orderSummary
          ? orderSummary.valid[index]
            ? orderSummary.values[index]
            : 0
          : validView && !validView.getUint32(index * 4, true)
            ? 0
            : result.format === 'float32'
              ? valueView.getFloat32(index * 4, true)
              : valueView.getUint32(index * 4, true);
        if (!Number.isFinite(value))
          throw new Error('Category chart aggregate overflowed its numeric representation');
        return {
          category,
          value,
          count: countView.getUint32(index * 4, true),
          firstRow: firstView.getUint32(index * 4, true)
        };
      });
      return {
        kind: 'category',
        summaries: this.seriesLabels
          ? foldResidentChartSeries(cells, this.labels, this.seriesLabels)
          : cells,
        sourceRevision: this.lease.dataset.revision,
        selectionRevision,
        readbackBytes
      };
    } catch (error) {
      if (!this.isCurrent(requestRevision, selectionRevision)) return null;
      throw error;
    } finally {
      reservation.release();
    }
  }
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

/** Exact first-row indices avoid dictionary-order colors and unstable equal-value ordering. */
function makeFirstRowsGraph(
  device: Device,
  categories: GPUVector,
  selection: GPUVector<'uint32'>,
  validity: GPUVector<'uint32'> | undefined,
  firstRows: Buffer,
  groupCount: number
): CompiledGPUCommandGraph {
  const graph = new GPUCommandGraph(device, {id: 'chart-category-first-rows'});
  const output = graph.createDataView(
    graph.importBuffer(
      {id: 'first-rows', byteLength: firstRows.byteLength, usage: firstRows.usage},
      firstRows
    ),
    {format: 'uint32', length: groupCount}
  );
  graph.addComputePass({
    id: 'initialize',
    resources: [{buffer: output, usage: 'storage-write'}],
    compile: ({device: compileDevice}) => {
      const kernel = new Kernel(compileDevice, {
        source: `@group(0) @binding(0) var<storage,read_write> output: array<u32>; @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {if (invocation.x < ${groupCount}u) {output[invocation.x] = 0xffffffffu;}}`,
        shaderLayout: {bindings: [{name: 'output', group: 0, location: 0, type: 'storage'}]}
      });
      return {
        encode: ({computePass, getBuffer: resolveBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {output: resolveBuffer(output.buffer)},
            x: Math.ceil(groupCount / 256)
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
  const keys = graph.importGPUVector('categories', categories);
  const masks = graph.importGPUVector('selection', selection);
  const valid = validity ? graph.importGPUVector('validity', validity) : undefined;
  let sourceOffset = 0;
  for (const [chunkIndex, key] of keys.data.entries()) {
    const offset = sourceOffset;
    sourceOffset += key.length;
    if (!key.length) continue;
    const mask = masks.data[chunkIndex];
    const validChunk = valid?.data[chunkIndex];
    const dispatch = getBoundedDispatchLayout(
      'ChartFirstRows',
      key.length,
      256,
      device.limits.maxComputeWorkgroupsPerDimension
    );
    graph.addComputePass({
      id: `first-${chunkIndex}`,
      resources: [
        {buffer: key, usage: 'storage-read'},
        {buffer: mask, usage: 'storage-read'},
        ...(validChunk ? [{buffer: validChunk, usage: 'storage-read' as const}] : []),
        {buffer: output, usage: 'storage-read-write'}
      ],
      compile: ({device: compileDevice}) => {
        const kernel = new Kernel(compileDevice, {
          source: `
@group(0) @binding(0) var<storage,read> keys: array<u32>;
@group(0) @binding(1) var<storage,read> mask: array<u32>;
@group(0) @binding(2) var<storage,read_write> output: array<atomic<u32>>;
${validChunk ? '@group(0) @binding(3) var<storage,read> valid: array<u32>;' : ''}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) localInvocationIndex: u32, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if(index >= ${key.length}u || mask[${mask.byteOffset / 4}u + index] == 0u) {return;}
  ${validChunk ? `if(valid[${validChunk.byteOffset / 4}u + index] == 0u) {return;}` : ''}
  let category = keys[${key.byteOffset / 4}u + index];
  if(category < ${groupCount}u) {atomicMin(&output[category], ${offset}u + index);}
}`,
          shaderLayout: {
            bindings: ['keys', 'mask', 'output', ...(validChunk ? ['valid'] : [])].map(
              (name, location) => ({name, location, group: 0, type: 'storage' as const})
            )
          }
        });
        return {
          encode: ({computePass, getBuffer: resolveBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: {
                keys: resolveBuffer(key.buffer),
                mask: resolveBuffer(mask.buffer),
                output: resolveBuffer(output.buffer),
                ...(validChunk ? {valid: resolveBuffer(validChunk.buffer)} : {})
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
