// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDataFrame, literal, type CompiledGPUDataFrameAggregation, type GPUDataFrameQueryParameters} from '@luma.gl/graph/gpu-dataframe';
import {GPUTable} from '@luma.gl/graph/gpu-tables';
import {GPUVector as Vector} from '@luma.gl/graph/gpu-data';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetLease,
  KeplerSessionExplorer
} from '@luma-contributors/kepler';
import {
  ResidentCategoryChartRuntime,
  type ResidentCategoryChartResult,
  type ResidentChartSelection
} from './resident-chart-group-runtime';
import {
  allocateVector,
  getBuffer,
  makeBinningGraph,
  makePrepareGraph
} from './resident-chart-numeric-runtime';
import type {ResidentCategoryChartPlan} from './resident-chart-plan';
import {
  decodeResidentNumericOrder,
  encodeResidentNumericThreshold,
  makeResidentNumericBoundaries
} from './resident-numeric-domain';

/** Label of one equal-width interval; the last interval includes its upper bound. */
export function formatResidentNumericBinLabel(lower: number, upper: number): string {
  const format = (value: number) => String(Number(value.toPrecision(6)));
  return `${format(lower)} to ${format(upper)}`;
}

/** Labels padded to exactly `binCount` entries (unused bins are empty and never receive rows). */
export function makeResidentNumericSeriesLabels(
  boundaries: readonly number[],
  binCount: number
): string[] {
  const labels: string[] = [];
  for (let index = 0; index + 1 < boundaries.length; index++)
    labels.push(formatResidentNumericBinLabel(boundaries[index], boundaries[index + 1]));
  while (labels.length < binCount) labels.push('');
  return labels.slice(0, binCount);
}

type ExtentColumns = {ordered: 'uint32'};

/**
 * Derived series keys for a numeric field: one GPU pass maps values to sortable integers, the
 * range over the whole dataset (not the selection, so bins stay stable) is reduced on the GPU,
 * and a second pass assigns each row an equal-width bin. Only the range returns to the CPU.
 */
export class ResidentNumericSeriesKeys {
  private readonly vectors: GPUVector<'uint32'>[] = [];
  private readonly prepare: CompiledGPUCommandGraph;
  private readonly binning: CompiledGPUCommandGraph;
  private readonly extent: CompiledGPUDataFrameAggregation<{
    minimum: 'uint32';
    maximum: 'uint32';
  }>;
  private readonly thresholds: Buffer;
  readonly keys: GPUVector<'uint32'>;

  constructor(
    private readonly props: {
      device: Device;
      values: GPUVector;
      validity?: GPUVector<'uint32'>;
      binFormat: 'float32' | 'sint32' | 'uint32';
      binCount: number;
    }
  ) {
    const {device, values, validity} = props;
    const ordered = allocateVector(device, 'numeric-series-ordered', values);
    const finite = allocateVector(device, 'numeric-series-finite', values);
    this.keys = allocateVector(device, 'numeric-series-bins', values);
    this.vectors.push(ordered, finite, this.keys);
    this.prepare = makePrepareGraph(device, values, validity, ordered, finite, props.binFormat);
    this.thresholds = device.createBuffer({
      id: 'numeric-series-thresholds',
      byteLength: (props.binCount + 2) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.binning = makeBinningGraph(device, ordered, finite, this.keys, this.thresholds, validity);
    const table = new GPUTable<ExtentColumns>({
      vectors: {
        ordered: new Vector({
          name: 'ordered',
          type: 'data',
          format: 'uint32',
          data: ordered.data,
          ownsData: false
        })
      }
    });
    for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)]) {
      const field = schema.fields.find(candidate => candidate.name === 'ordered');
      if (field) field.nullable = true;
    }
    const frame = new GPUDataFrame({table, validity: {ordered: finite}});
    try {
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
        id: 'numeric-series-extent'
      });
      this.extent = frame
        .filter(literal(true))
        .aggregate({minimum: {min: 'ordered'}, maximum: {max: 'ordered'}})
        .compile(graph);
    } finally {
      frame.destroy();
    }
  }

  /** Runs the passes and returns the interval labels, or null when the device or source changed. */
  async bin(isCurrent: () => boolean): Promise<string[] | null> {
    const {device, binFormat, binCount} = this.props;
    const encoder = device.createCommandEncoder();
    this.prepare.encode(encoder, {parameters: undefined});
    this.extent.encode(encoder);
    device.submit(encoder.finish());
    const minimumVector = this.extent.table.gpuVectors['minimum'];
    const maximumVector = this.extent.table.gpuVectors['maximum'];
    const validVector = this.extent.validity.minimum!;
    const [minimumBytes, maximumBytes, validBytes] = await Promise.all([
      getBuffer(minimumVector.data[0]).readAsync(minimumVector.data[0].byteOffset, 4),
      getBuffer(maximumVector.data[0]).readAsync(maximumVector.data[0].byteOffset, 4),
      getBuffer(validVector.data[0]).readAsync(validVector.data[0].byteOffset, 4)
    ]);
    if (!isCurrent()) return null;
    const read = (bytes: Uint8Array) =>
      new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
    const words = new Uint32Array(this.thresholds.byteLength / 4);
    let boundaries: number[] = [];
    if (read(validBytes)) {
      boundaries = makeResidentNumericBoundaries(
        decodeResidentNumericOrder(read(minimumBytes), binFormat),
        decodeResidentNumericOrder(read(maximumBytes), binFormat),
        binCount
      );
      words[1] = boundaries.length - 2;
      for (let index = 1; index < boundaries.length - 1; index++)
        words[index + 1] = encodeResidentNumericThreshold(boundaries[index], binFormat);
    }
    this.thresholds.write(words);
    const binEncoder = device.createCommandEncoder();
    this.binning.encode(binEncoder, {parameters: undefined});
    device.submit(binEncoder.finish());
    return makeResidentNumericSeriesLabels(boundaries, binCount);
  }

  destroy(): void {
    this.extent.destroy();
    this.prepare.destroy();
    this.binning.destroy();
    this.thresholds.destroy();
    for (const vector of this.vectors) vector.destroy();
  }
}

/**
 * Category chart whose series is a numeric field: series bins are derived on the GPU the first
 * time the chart is read, then the usual dense group x series summary runs over them.
 */
export class ResidentNumericSeriesChartRuntime {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private seriesKeys?: ResidentNumericSeriesKeys;
  private grouped?: ResidentCategoryChartRuntime;
  private tail: Promise<void> = Promise.resolve();
  private requestRevision = 0;
  private destruction?: Promise<void>;
  private failure?: unknown;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      plan: ResidentCategoryChartPlan;
      selection?: ResidentChartSelection;
      workingMemoryBudgetBytes: number;
      measure?: GPUVector<'float32'>;
    }
  ) {
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(props.plan.datasetId);
    const {plan} = props;
    const source = this.lease.dataset.value;
    const values = plan.series ? source.columns[plan.series.column] : undefined;
    try {
      if (!values || !plan.series || !plan.numericSeries)
        throw new Error('Numeric series column is unavailable');
      this.seriesKeys = new ResidentNumericSeriesKeys({
        device: props.session.device,
        values,
        validity: source.analytics.validity[plan.series.column],
        binFormat: plan.numericSeries.binFormat as 'float32' | 'sint32' | 'uint32',
        binCount: plan.series.categoryCount
      });
    } catch (error) {
      void this.lease.release();
      throw error;
    }
  }

  get readbackByteLength(): number {
    return 12 + (this.grouped?.readbackByteLength ?? 0);
  }
  cancel(): void {
    this.requestRevision++;
    this.grouped?.cancel();
  }
  read(): Promise<ResidentCategoryChartResult | null> {
    const requestRevision = ++this.requestRevision;
    const pending = this.tail.then(async () => {
      if (this.failure) throw this.failure;
      if (!this.grouped) {
        try {
          const labels = await this.seriesKeys!.bin(
            () => this.requestRevision === requestRevision && this.lease.isValid()
          );
          if (!labels) return null;
          this.grouped = new ResidentCategoryChartRuntime({
            ...this.props,
            plan: {...this.props.plan, numericSeries: undefined},
            seriesInput: {keys: this.seriesKeys!.keys, labels}
          });
        } catch (error) {
          this.failure = error;
          throw error;
        }
      }
      return this.requestRevision === requestRevision ? this.grouped.read() : null;
    });
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }
  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.cancel();
    this.destruction = (async () => {
      await this.tail;
      try {
        await this.grouped?.destroy();
      } finally {
        this.seriesKeys?.destroy();
        await this.lease.release();
      }
    })();
    return this.destruction;
  }
}
