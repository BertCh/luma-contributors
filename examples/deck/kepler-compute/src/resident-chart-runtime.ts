// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector, type GPUData} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDataFrame, column, literal, type CompiledGPUDataFrameAggregation, type GPUDataFrameGlobalAggregationValue, type GPUDataFrameQueryParameters} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerSessionExplorer,
  KeplerDatasetLease,
  KeplerMemoryReservation,
  GPUResidentScalarFormat
} from '@luma-contributors/kepler';
import {KeplerAllocationError} from '@luma-contributors/kepler';
import {isResidentChartOrderOperation, type ResidentScalarChartPlan} from './resident-chart-plan';
import {ResidentPlotOrderMeasure} from './resident-plot-order-measure';

type ChartColumns = {value: GPUResidentScalarFormat; selected: 'uint32'};
export type ResidentScalarChartResult = {
  value: number;
  /** Native charts display zero for empty/all-invalid scalar aggregates. */
  hasValue: boolean;
  sourceRevision: number;
  selectionRevision: number | null;
  readbackBytes: number;
};

/** Own per chart, and destroy before closing its explorer. Only one scalar and validity return. */
export class ResidentScalarChartRuntime {
  private readonly session: KeplerComputeSession;
  private readonly device: Device;
  private readonly explorer: KeplerSessionExplorer;
  private readonly plan: ResidentScalarChartPlan;
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private readonly reservation?: KeplerMemoryReservation;
  private readonly query?: CompiledGPUDataFrameAggregation<{result: GPUResidentScalarFormat}>;
  private readonly orderMeasure?: ResidentPlotOrderMeasure;
  private readTail: Promise<void> = Promise.resolve();
  private requestRevision = 0;
  private destruction?: Promise<void>;
  private destroyed = false;
  private transferredBytes = 0;

  constructor(props: {
    session: KeplerComputeSession;
    explorer: KeplerSessionExplorer;
    plan: ResidentScalarChartPlan;
    workingMemoryBudgetBytes: number;
    /** Exact Float32 stand-in for an Int32 or Uint32 measure column. */
    measure?: GPUVector<'float32'>;
  }) {
    if (!Number.isSafeInteger(props.workingMemoryBudgetBytes) || props.workingMemoryBudgetBytes < 0)
      throw new Error('Invalid chart working memory budget');
    if (props.plan.datasetId !== props.explorer.dataset.id)
      throw new Error('Chart and explorer require the same dataset');
    this.session = props.session;
    this.device = props.session.device;
    this.explorer = props.explorer;
    this.plan = {...props.plan};
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.plan.datasetId);
    let reservation: KeplerMemoryReservation | undefined;
    let frame: GPUDataFrame<ChartColumns> | undefined;
    let query: CompiledGPUDataFrameAggregation<{result: GPUResidentScalarFormat}> | undefined;
    let orderMeasure: ResidentPlotOrderMeasure | undefined;
    try {
      if (this.lease.dataset.revision !== this.explorer.dataset.revision)
        throw new Error('Chart and explorer require the same source revision');
      const source = this.lease.dataset.value;
      if (isResidentChartOrderOperation(this.plan.operation)) {
        if (!this.plan.column || !source.columns[this.plan.column])
          throw new Error('Chart source column is unavailable');
        // One global key: the same exact ordered statistics as grouped charts, without a frame.
        orderMeasure = new ResidentPlotOrderMeasure({
          session: this.session,
          lease: this.lease,
          ...(this.plan.population === 'selected' ? {mask: this.explorer.selectionMask} : {}),
          ...(props.measure ? {values: props.measure} : {}),
          keyCount: 1,
          column: this.plan.column,
          operation: this.plan.operation,
          workingMemoryBudgetBytes: props.workingMemoryBudgetBytes
        });
        this.orderMeasure = orderMeasure;
        return;
      }
      const sourceColumn = this.plan.column
        ? source.columns[this.plan.column]
        : this.explorer.selectionMask;
      if (!sourceColumn) throw new Error('Chart source column is unavailable');
      const measureColumn =
        this.plan.column &&
        (this.plan.operation === 'sum' || this.plan.operation === 'mean') &&
        props.measure
          ? props.measure
          : sourceColumn;
      if (
        (this.plan.operation === 'sum' || this.plan.operation === 'mean') &&
        measureColumn.format !== 'float32'
      )
        throw new Error('Chart sum and average require Float32 measurements');
      const validity = this.plan.column ? source.analytics.validity[this.plan.column] : undefined;
      // Reserve conservative selection, reduction, sanitation and per-chunk overhead before compile.
      const estimatedBytes =
        65536 +
        sourceColumn.data.length * 4096 +
        sourceColumn.length * (this.plan.operation === 'count' ? 32 : 128);
      if (estimatedBytes > props.workingMemoryBudgetBytes)
        throw new KeplerAllocationError('working-memory', {
          message: 'Chart exceeds its working memory budget before allocation'
        });
      reservation = this.session.registry.reserveMemory(estimatedBytes, 'gpu', 'scratch');
      const table = new GPUTable<ChartColumns>({
        vectors: {
          value: borrowVector('value', measureColumn),
          selected: borrowVector('selected', this.explorer.selectionMask)
        }
      });
      for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
        const field = schema.fields.find(candidate => candidate.name === 'value');
        if (field) field.nullable = Boolean(validity);
      }
      frame = new GPUDataFrame({table, ...(validity ? {validity: {value: validity}} : {})});
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(this.device, {
        id: `kepler-chart-${this.plan.chartId}`
      });
      const definitions: {
        result: GPUDataFrameGlobalAggregationValue<ChartColumns, keyof ChartColumns>;
      } = {
        result:
          this.plan.operation === 'count'
            ? 'count'
            : this.plan.operation === 'sum'
              ? {sum: 'value'}
              : this.plan.operation === 'mean'
                ? {mean: 'value'}
                : this.plan.operation === 'min'
                  ? {min: 'value'}
                  : {max: 'value'}
      };
      query =
        this.plan.population === 'selected'
          ? frame
              .filter(column('selected').notEqual(literal(0)))
              .aggregate(definitions)
              .compile(graph)
          : frame.aggregate(definitions).compile(graph);
      const buffers = new Set<Buffer>();
      for (const vector of [
        ...Object.values(query.table.gpuVectors),
        ...Object.values(query.validity),
        query.selectionMask,
        query.rowIndices,
        query.selectedCounts
      ]) {
        if (vector) for (const data of vector.data) buffers.add(getBuffer(data));
      }
      const accountedBytes =
        query.compilationStats.physicalTransientResourceBytes +
        [...buffers].reduce((total, buffer) => total + buffer.byteLength, 0);
      if (accountedBytes > reservation.byteLength)
        throw new KeplerAllocationError('working-memory', {
          message: 'Chart exceeds its reserved working memory'
        });
      this.reservation = reservation;
      this.query = query;
    } catch (error) {
      orderMeasure?.destroy();
      query?.destroy();
      reservation?.release();
      void this.lease.release();
      throw error;
    } finally {
      frame?.destroy();
    }
  }

  get readbackByteLength(): number {
    return this.transferredBytes;
  }

  read(): Promise<ResidentScalarChartResult | null> {
    const requestRevision = ++this.requestRevision;
    const selectionRevision =
      this.plan.population === 'selected' ? this.explorer.currentRevision : null;
    const result = this.readTail.then(() => this.readCurrent(requestRevision, selectionRevision));
    this.readTail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  cancel(): void {
    this.requestRevision++;
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.cancel();
    this.destruction = (async () => {
      await this.readTail;
      let fence: ReturnType<Device['createFence']> | undefined;
      try {
        if (!this.device.isLost) {
          fence = this.device.createFence();
          await Promise.race([fence.signaled, this.device.lost]);
        }
      } finally {
        fence?.destroy();
        this.orderMeasure?.destroy();
        this.query?.destroy();
        this.reservation?.release();
        await this.lease.release();
      }
    })();
    return this.destruction;
  }

  private isCurrent(requestRevision: number, selectionRevision: number | null): boolean {
    return (
      !this.destroyed &&
      !this.session.deviceLost &&
      this.lease.isValid() &&
      this.explorer.isCurrent &&
      requestRevision === this.requestRevision &&
      (selectionRevision === null || selectionRevision === this.explorer.currentRevision)
    );
  }

  private async readOrderCurrent(
    requestRevision: number,
    selectionRevision: number | null
  ): Promise<ResidentScalarChartResult | null> {
    const measure = this.orderMeasure!;
    const readbackBytes = measure.readbackByteLength;
    const reservation = this.session.registry.reserveMemory(readbackBytes, 'cpu', 'readback');
    try {
      const encoder = this.device.createCommandEncoder();
      if (selectionRevision !== null) this.explorer.encode(encoder);
      measure.encode(encoder);
      this.device.submit(encoder.finish());
      this.transferredBytes += readbackBytes;
      const summary = await measure.readSummary();
      if (!this.isCurrent(requestRevision, selectionRevision)) return null;
      const hasValue = summary.valid[0] === 1;
      return {
        value: hasValue ? summary.values[0] : 0,
        hasValue,
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

  private async readCurrent(
    requestRevision: number,
    selectionRevision: number | null
  ): Promise<ResidentScalarChartResult | null> {
    if (!this.isCurrent(requestRevision, selectionRevision)) return null;
    if (this.orderMeasure) return this.readOrderCurrent(requestRevision, selectionRevision);
    const query = this.query!;
    const output = query.table.gpuVectors['result'];
    const validity = query.validity.result;
    const readbackBytes = validity ? 8 : 4;
    const reservation = this.session.registry.reserveMemory(readbackBytes, 'cpu', 'readback');
    try {
      const encoder = this.device.createCommandEncoder();
      if (selectionRevision !== null) this.explorer.encode(encoder);
      query.encode(encoder);
      this.device.submit(encoder.finish());
      this.transferredBytes += readbackBytes;
      const [values, valid] = await Promise.all([
        getBuffer(output.data[0]).readAsync(output.data[0].byteOffset, 4),
        validity ? getBuffer(validity.data[0]).readAsync(validity.data[0].byteOffset, 4) : undefined
      ]);
      if (!this.isCurrent(requestRevision, selectionRevision)) return null;
      const hasValue =
        !valid || new DataView(valid.buffer, valid.byteOffset, 4).getUint32(0, true) !== 0;
      const view = new DataView(values.buffer, values.byteOffset, 4);
      const value =
        output.format === 'float32'
          ? view.getFloat32(0, true)
          : output.format === 'sint32'
            ? view.getInt32(0, true)
            : view.getUint32(0, true);
      if (hasValue && !Number.isFinite(value))
        throw new Error('Chart aggregate overflowed its numeric representation');
      return {
        value: hasValue ? value : 0,
        hasValue,
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

function borrowVector<Format extends GPUResidentScalarFormat>(
  name: string,
  vector: GPUVector<Format>
): GPUVector<Format> {
  return new GPUVector({
    name,
    type: 'data',
    format: vector.format,
    data: vector.data,
    ownsData: false
  });
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}
