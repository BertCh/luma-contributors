// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '@luma-contributors/kepler';
import type {LayerProps} from '@deck.gl/core';
import type {GPUSolidPolygonLayerProps} from '../gpu-layers/index';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma.gl/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUGather,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUClassBreaks,
  GPUColorScale,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  getGPUPointDensityHexagonCenter,
  getGPUPointDensityD3HexagonCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUKeplerAggregation,
  getKeplerAggregationLayout,
  type GPUKeplerAggregationTopology,
  type GPUKeplerAggregationLayout,
  type KeplerDatasetRegistry,
  type KeplerMemoryReservation,
  type GPUResidentScalarFormat
} from '@luma-contributors/kepler';
import {getDistanceScales, lngLatToWorld, MAX_LATITUDE} from '@math.gl/web-mercator';
import {AggregationStatistics} from './aggregation-statistics-gpu';
import {
  widenResidentChartMeasure,
  type ResidentChartWidenedMeasure
} from './resident-chart-measure-widening';
import {getAggregateCountLabels, type AggregationCountLabel} from './aggregation-count-labels';
import {getJenksBreaks, getOrdinalLookup} from './aggregation-scale-lookup';
import {
  getAggregationStylePlan,
  isAdvancedAggregationLayer,
  type AggregationStylePlan,
  type NativeAggregationName
} from './aggregation-statistics-plan';

export type NativeAggregationLayer = {
  id: string;
  type: 'grid' | 'hexagon';
  config: {
    columnMode?: string;
    columns: Record<string, {value?: string | null} | undefined>;
    colorField?: {name: string; type?: string} | null;
    colorScale?: PropertyKey;
    sizeField?: {name: string; type?: string} | null;
    sizeScale?: PropertyKey;
    textLabel?: readonly {field?: unknown}[];
    visConfig: {
      worldUnitSize: number;
      coverage?: number;
      enable3d?: boolean;
      colorAggregation?: string;
      sizeAggregation?: string;
      percentile?: readonly number[];
      elevationPercentile?: readonly number[];
      sizeRange?: readonly number[];
      colorRange?: {colors: readonly string[]; reversed?: boolean; colorMap?: unknown};
    };
  };
};
export type NativeAggregationSettings = {
  kind: 'grid' | 'hexagon';
  sizeMeters: number;
  coverage: number;
  statistic: 'count' | 'sum' | 'mean';
  weightColumn?: string;
  colorScale: 'linear' | 'quantize' | 'quantile';
  palette: readonly number[];
  /** Present for statistics, percentile, custom-scale or elevation controls beyond count/sum/mean. */
  advanced?: AggregationStylePlan;
};
export type AggregationProjection = {
  positions: Float32Array;
  /** Rebased metric bounds; meters use the dataset centroid's Mercator distance scale. */
  bounds: [number, number, number, number];
  geographicBounds: [number, number, number, number];
  commonOrigin: [number, number, number];
  commonUnitsPerMeter: number;
};
export type ResidentAggregationBinding = Pick<
  GPUSolidPolygonLayerProps,
  | 'positions'
  | 'indices'
  | 'rowIndices'
  | 'selectionMask'
  | 'getFillColor'
  | 'getElevation'
  | 'extrusionRings'
> &
  Pick<LayerProps, 'coordinateSystem' | 'coordinateOrigin' | 'modelMatrix'> & {
    layout: GPUKeplerAggregationLayout;
    counts: GPUVector<'uint32'>;
    values: GPUVector<'float32'>;
    contract: 'mercator-centroid-meters-float32';
    /** Native extrusion: heights are interpolated in meters; multiply by the native elevationScale. */
    extruded: boolean;
    /**
     * Native point-count labels (cell center + count). The first call requests them; they are read
     * back once per selection revision (bounded by the cell budget) and the layer is redrawn.
     */
    getCountLabels?: () => readonly AggregationCountLabel[] | undefined;
  };
type CountLabelState = {
  wanted: boolean;
  stale: boolean;
  sequence: number;
  labels?: readonly AggregationCountLabel[];
};
export type AggregationCell = {
  layerId: string;
  cellId: number;
  count: number;
  value: number;
  statistic: NativeAggregationName | 'mean';
  /** Dictionary label of a categorical mode result; native shows the label, not the code. */
  label?: string;
  revision: number;
};
type Program = {
  key: string;
  sourceSelectionMask: GPUVector<'uint32'>;
  sourceBinOrderMask?: GPUVector<'uint32'>;
  selectionRevision?: number;
  readable: boolean;
  aggregation: GPUKeplerAggregation;
  compiled: CompiledGPUCommandGraph;
  statistics?: AggregationStatistics;
  buffers: Buffer[];
  binding: ResidentAggregationBinding;
  settings: NativeAggregationSettings;
  revision: number;
  /** Jenks/ordinal lookups are re-derived from a bounded cell-value readback when the inputs change. */
  scaleLookupStale: boolean;
  scaleLookupSequence: number;
  countLabels: CountLabelState;
  destroy(): void;
};

/** Native controls are interpreted here so unsupported options never silently alter their meaning. */
export function getNativeAggregationSettings(
  layer: NativeAggregationLayer
): NativeAggregationSettings {
  const {config} = layer;
  const {visConfig} = config;
  if (config.columnMode && config.columnMode !== 'points')
    throw new Error('Resident aggregation requires latitude/longitude point columns.');
  // Aggregate count/text labels are not drawn by the resident aggregation; the layer still renders.
  const sizeMeters = visConfig.worldUnitSize * 1000;
  if (!Number.isFinite(sizeMeters) || sizeMeters <= 0)
    throw new Error('Aggregate cell size must be positive kilometers.');
  const coverage = visConfig.coverage ?? 1;
  if (!Number.isFinite(coverage) || coverage < 0 || coverage > 1)
    throw new Error('Aggregate coverage must lie between zero and one.');
  const advanced = isAdvancedAggregationLayer(layer) ? getAggregationStylePlan(layer) : undefined;
  const operation = config.colorField
    ? (visConfig.colorAggregation ?? 'average').toLowerCase()
    : 'count';
  const statistic = advanced
    ? 'count'
    : operation === 'average' || operation === 'mean'
      ? 'mean'
      : operation === 'sum'
        ? 'sum'
        : operation === 'count'
          ? 'count'
          : undefined;
  if (!statistic) throw new Error(`Resident aggregation does not support ${operation}.`);
  const colorScale = String(config.colorScale ?? 'quantize');
  if (
    !advanced &&
    colorScale !== 'quantize' &&
    colorScale !== 'quantile' &&
    colorScale !== 'linear'
  )
    throw new Error(`Resident aggregation does not support the ${colorScale} color scale.`);
  const colors = visConfig.colorRange?.colors ?? ['#21918c', '#fde725'];
  if (colors.length < 1 || colors.length > 64)
    throw new Error('Resident aggregate palettes require 1 to 64 colors.');
  const palette = colors.map(parseHexColor);
  if (visConfig.colorRange?.reversed) palette.reverse();
  return {
    kind: layer.type,
    sizeMeters,
    coverage,
    statistic,
    weightColumn: statistic === 'count' ? undefined : config.colorField?.name,
    colorScale: colorScale as NativeAggregationSettings['colorScale'],
    palette,
    ...(advanced ? {advanced} : {})
  };
}

/** Initial CPU projection only. Repeated filtering never projects or uploads source positions again. */
export function prepareAggregationProjection(
  longitude: ArrayLike<number | null | undefined>,
  latitude: ArrayLike<number | null | undefined>
): AggregationProjection {
  if (!longitude.length || longitude.length !== latitude.length)
    throw new Error('Aggregation requires equally sized nonempty position columns.');
  const geographicBounds: [number, number, number, number] = [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ];
  for (let row = 0; row < longitude.length; row++) {
    const east = longitude[row];
    const north = latitude[row];
    if (
      typeof east !== 'number' ||
      typeof north !== 'number' ||
      !Number.isFinite(east) ||
      !Number.isFinite(north)
    )
      continue;
    // Native Kepler skips rows outside the Web Mercator domain.
    if (Math.abs(east) > 180 || Math.abs(north) > MAX_LATITUDE) continue;
    geographicBounds[0] = Math.min(geographicBounds[0], east);
    geographicBounds[1] = Math.min(geographicBounds[1], north);
    geographicBounds[2] = Math.max(geographicBounds[2], east);
    geographicBounds[3] = Math.max(geographicBounds[3], north);
  }
  if (!geographicBounds.every(Number.isFinite) || geographicBounds[2] - geographicBounds[0] > 360)
    throw new Error(
      'Aggregation requires finite data in one longitude interval without antimeridian wrapping.'
    );
  const center = [
    (geographicBounds[0] + geographicBounds[2]) / 2,
    (geographicBounds[1] + geographicBounds[3]) / 2
  ];
  const commonOrigin: [number, number, number] = [...lngLatToWorld(center), 0];
  const commonUnitsPerMeter = getDistanceScales({longitude: center[0], latitude: center[1]})
    .unitsPerMeter[0];
  const positions = new Float32Array(longitude.length * 2);
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let row = 0; row < longitude.length; row++) {
    const east = longitude[row];
    const north = latitude[row];
    if (
      typeof east !== 'number' ||
      typeof north !== 'number' ||
      !Number.isFinite(east) ||
      !Number.isFinite(north) ||
      Math.abs(east) > 180 ||
      Math.abs(north) > MAX_LATITUDE
    ) {
      positions.set([NaN, NaN], row * 2);
      continue;
    }
    const world = lngLatToWorld([east, north]);
    const point = [
      Math.fround((world[0] - commonOrigin[0]) / commonUnitsPerMeter),
      Math.fround((world[1] - commonOrigin[1]) / commonUnitsPerMeter)
    ];
    positions.set(point, row * 2);
    bounds[0] = Math.min(bounds[0], point[0]);
    bounds[1] = Math.min(bounds[1], point[1]);
    bounds[2] = Math.max(bounds[2], point[0]);
    bounds[3] = Math.max(bounds[3], point[1]);
  }
  return {positions, bounds, geographicBounds, commonOrigin, commonUnitsPerMeter};
}

/** Anchors cells in global Mercator space, then rebases them into the dataset's metric frame. */
export function getNativeAggregationTopology(
  projection: AggregationProjection,
  settings: Pick<NativeAggregationSettings, 'kind' | 'sizeMeters'>
): GPUKeplerAggregationTopology {
  const {bounds, commonOrigin, commonUnitsPerMeter} = projection;
  const size = settings.sizeMeters;
  if (settings.kind === 'grid') {
    // Deck snaps the dataset centroid to the global cell lattice, then rounds that common
    // origin to Float32. Include that deliberate rounding before selecting local bounds.
    const commonSize = size * commonUnitsPerMeter;
    const anchor = commonOrigin
      .slice(0, 2)
      .map(
        value =>
          (Math.fround(Math.floor(value / commonSize) * commonSize) - value) / commonUnitsPerMeter
      );
    const origin: [number, number] = [
      anchor[0] + Math.floor((bounds[0] - anchor[0]) / size) * size,
      anchor[1] + Math.floor((bounds[1] - anchor[1]) / size) * size
    ];
    return {
      kind: 'grid',
      origin,
      cellSize: [size, size],
      gridSize: [
        Math.floor((bounds[2] - origin[0]) / size) + 1,
        Math.floor((bounds[3] - origin[1]) / size) + 1
      ]
    };
  }
  const commonRadius = size * commonUnitsPerMeter;
  const centerCell = getGPUPointDensityD3HexagonCell(
    commonOrigin[0],
    commonOrigin[1],
    0,
    0,
    commonRadius
  );
  const center = getGPUPointDensityHexagonCenter(centerCell[0], centerCell[1], 0, 0, commonRadius);
  const anchor = center.map(
    (value, index) => (Math.fround(value) - commonOrigin[index]) / commonUnitsPerMeter
  );
  // Native hexbin rebases around the Float32 common-space centroid. Keep the first local
  // odd-r row even, preserving its phase; pad to retain cells overlapping source bounds.
  const row = Math.floor((bounds[1] - anchor[1]) / (1.5 * size) / 2) * 2 - 2;
  const column = Math.floor((bounds[0] - anchor[0]) / (Math.sqrt(3) * size)) - 1;
  return {
    kind: 'hexagon',
    radius: size,
    assignment: 'd3',
    bounds: [
      column * Math.sqrt(3) * size + anchor[0],
      row * 1.5 * size + anchor[1],
      bounds[2] + size * 2,
      bounds[3] + size * 2
    ]
  };
}

/** Per-layer reusable density and classification programs. The host owns queue submission. */
export class AggregationRuntime {
  readonly projection: AggregationProjection;
  readonly positions: GPUVector<'float32x2'>;
  readonly uploadBytes: number;
  private readonly programs = new Map<string, Program>();
  private readonly retiredPrograms = new Set<Program>();
  private readonly sourceReservations: KeplerMemoryReservation[] = [];
  private readonly retirements = new Set<Promise<void>>();
  private readonly reads = new Set<Promise<unknown>>();
  private destroyed = false;
  private released = false;
  private destruction?: Promise<void>;
  private readonly deviceLost: Promise<void>;
  private readonly maximumConcurrentReads: number;
  private readbackBytes = 0;
  private readonly widenedColumns = new Map<
    string,
    {
      pending?: Promise<void>;
      value?: ResidentChartWidenedMeasure;
      reason?: string;
    }
  >();
  constructor(
    private readonly props: {
      device: Device;
      longitude: ArrayLike<number | null | undefined>;
      latitude: ArrayLike<number | null | undefined>;
      /** Exact immutable publication batch lengths, including empty batches. */
      sourceChunkLengths?: readonly number[];
      maximumConcurrentReads?: number;
      selectionMask: GPUVector<'uint32'>;
      columns?: Readonly<
        Record<string, {values: GPUVector<GPUResidentScalarFormat>; validity?: GPUVector<'uint32'>}>
      >;
      coordinateColumns?: {lat: string; lng: string};
      maximumCells?: number;
      memoryRegistry?: KeplerDatasetRegistry;
      programMemoryBudgetBytes?: number;
      requestUpdate?: () => void;
      /** Dictionary labels of a categorical (dictionary-coded) column, indexed by code. */
      getCategoryLabels?: (column: string) => readonly string[] | undefined;
    }
  ) {
    this.deviceLost = props.device.lost.then(
      () => undefined,
      () => undefined
    );
    this.maximumConcurrentReads = props.maximumConcurrentReads ?? 2;
    if (!Number.isSafeInteger(this.maximumConcurrentReads) || this.maximumConcurrentReads < 1)
      throw new Error('Aggregation read concurrency must be a positive integer.');
    const chunkLengths = [
      ...(props.sourceChunkLengths ?? props.selectionMask.data.map(chunk => chunk.length))
    ];
    if (
      chunkLengths.some(length => !Number.isSafeInteger(length) || length < 0) ||
      chunkLengths.reduce((sum, length) => sum + length, 0) !== props.longitude.length
    )
      throw new Error('Aggregation source chunks must preserve the published row count.');
    if (
      props.selectionMask.length !== props.longitude.length ||
      props.selectionMask.data.some(chunk => chunk.buffer.device !== props.device)
    )
      throw new Error('Aggregation selection does not match source rows or device.');
    const buffers: Buffer[] = [];
    try {
      if (props.memoryRegistry) {
        this.sourceReservations.push(
          props.memoryRegistry.reserveMemory(props.longitude.length * 8, 'cpu', 'source')
        );
        this.sourceReservations.push(
          props.memoryRegistry.reserveMemory(
            chunkLengths.reduce((sum, length) => sum + Math.max(4, length * 8), 0),
            'gpu',
            'source'
          )
        );
      }
      this.projection = prepareAggregationProjection(props.longitude, props.latitude);
      let sourceRowOffset = 0;
      const chunks = chunkLengths.map((length, index) => {
        const values = this.projection.positions.subarray(
          sourceRowOffset * 2,
          (sourceRowOffset + length) * 2
        );
        sourceRowOffset += length;
        const buffer = props.device.createBuffer({
          id: `kepler-aggregate-projected-positions-${index}`,
          ...(length ? {data: values} : {byteLength: 4}),
          usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return new GPUData({format: 'float32x2', buffer, length, ownsBuffer: true});
      });
      this.positions = new GPUVector({
        type: 'data',
        name: 'aggregate-positions',
        format: 'float32x2',
        data: chunks,
        ownsData: true
      });
      this.uploadBytes = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      for (const reservation of this.sourceReservations) reservation.release();
      throw error;
    }
    void this.deviceLost.then(() => this.destroy(Promise.resolve())).catch(() => undefined);
  }
  /** Borrow a native layer's scoped source-row mask. Content updates reuse its program; replacing
   * the vector changes the graph bindings and retires the old program behind its completion fence.
   * The constructor mask remains the explicit default for callers without native layer scopes.
   */
  resolve(
    layer: NativeAggregationLayer,
    selectionMask: GPUVector<'uint32'> = this.props.selectionMask,
    selectionRevision?: number,
    /** Native bin-forming rows (CPU-filtered topology); orders categorical bins. */
    binOrderMask?: GPUVector<'uint32'>
  ): {binding?: ResidentAggregationBinding; reason?: string} {
    if (this.destroyed) return {reason: 'Aggregation runtime has been destroyed.'};
    try {
      if (
        selectionRevision !== undefined &&
        (!Number.isSafeInteger(selectionRevision) || selectionRevision < 0)
      )
        throw new Error('Aggregation selection revision must be a nonnegative integer.');
      const settings = getNativeAggregationSettings(layer);
      const weightFormat = settings.weightColumn
        ? this.props.columns?.[settings.weightColumn]?.values.format
        : undefined;
      if (!settings.advanced && (weightFormat === 'sint32' || weightFormat === 'uint32')) {
        // GPUGroupStatistics avoids the density fast path's fixed-point atomic sum for integers.
        // The shared widening below still certifies exact Float32 admission before any reduction.
        settings.advanced = getAggregationStylePlan(layer);
        settings.statistic = 'count';
        settings.weightColumn = undefined;
      }
      if (selectionMask.length !== this.positions.length)
        throw new Error('Aggregation layer selection does not match source rows.');
      const expected = this.props.coordinateColumns ?? {lat: 'latitude', lng: 'longitude'};
      if (
        layer.config.columns['lat']?.value !== expected.lat ||
        layer.config.columns['lng']?.value !== expected.lng
      )
        throw new Error('Selected aggregation coordinate columns do not match uploaded positions.');
      if (settings.weightColumn && !this.props.columns?.[settings.weightColumn])
        throw new Error(`No resident Float32 weight column is bound for ${settings.weightColumn}.`);
      const columns = this.getNumericColumns(settings);
      if (!columns) return {};
      // Dictionary growth (streamed batches) changes the first-appearance remap size.
      const advancedChannels = [settings.advanced?.color, settings.advanced?.elevation];
      const key = `${JSON.stringify(settings)}${advancedChannels
        .map(channel => this.getChannelCategoryLabels(channel)?.length ?? '')
        .join(',')}`;
      let program = this.programs.get(layer.id);
      if (
        program?.key !== key ||
        program.sourceSelectionMask !== selectionMask ||
        program.sourceBinOrderMask !== binOrderMask
      ) {
        const replacement = this.createProgram(
          layer.id,
          key,
          settings,
          selectionMask,
          binOrderMask,
          columns
        );
        if (program) this.retire(program);
        this.programs.set(layer.id, replacement);
        program = replacement;
        this.props.requestUpdate?.();
      }
      if (program.selectionRevision !== selectionRevision) {
        program.selectionRevision = selectionRevision;
        program.scaleLookupStale = true;
        program.countLabels.stale = true;
        program.readable = false;
        program.revision++;
        this.props.requestUpdate?.();
      }
      return {binding: program.binding};
    } catch (error) {
      this.remove(layer.id);
      return {reason: error instanceof Error ? error.message : String(error)};
    }
  }
  encode(encoder: CommandEncoder): void {
    if (this.destroyed) return;
    for (const program of this.programs.values()) {
      program.aggregation.encode(encoder);
      program.statistics?.encode(encoder);
      program.compiled.encode(encoder, {parameters: undefined});
      program.revision++;
      program.readable = true;
      if (program.scaleLookupStale && program.statistics?.scaleLookupKind) {
        program.scaleLookupStale = false;
        this.refreshScaleLookup(program, program.statistics);
      }
      if (program.countLabels.wanted && program.countLabels.stale) {
        program.countLabels.stale = false;
        this.refreshCountLabels(program);
      }
    }
  }
  /**
   * Derives the jenks thresholds or ordinal table exactly like native (CPU, over the aggregated
   * cell values) from one readback of the per-cell color measure. The readback is deferred a task
   * so the frame that encoded the statistics has been submitted first; stale results are dropped.
   */
  private refreshScaleLookup(program: Program, statistics: AggregationStatistics): void {
    const sequence = ++program.scaleLookupSequence;
    const pending = (async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
      if (this.destroyed || this.findProgramId(program) === '') return;
      // Categorical mode channels read the GPU-computed first-appearance indices (native order).
      const values = await statistics.readScaleInput();
      this.readbackBytes += values.byteLength;
      if (this.destroyed || sequence !== program.scaleLookupSequence) return;
      statistics.setScaleLookup(
        statistics.scaleLookupKind === 'jenks'
          ? getJenksBreaks(
              Array.from(values).filter(Number.isFinite),
              program.settings.palette.length
            )
          : getOrdinalLookup(values)
      );
      this.props.requestUpdate?.();
    })()
      .catch(() => {
        // A retired program or lost device simply leaves the scale hidden.
      })
      .finally(() => this.reads.delete(pending));
    this.reads.add(pending);
  }
  /** One bounded readback of cell counts (and visibility) per selection revision for count labels. */
  private refreshCountLabels(program: Program): void {
    const state = program.countLabels;
    const sequence = ++state.sequence;
    const {layout} = program.binding;
    const pending = (async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
      if (this.destroyed || this.findProgramId(program) === '') return;
      const visibleBuffer = program.statistics?.visible;
      const [countBytes, visibleBytes] = await Promise.all([
        program.aggregation.counts.data[0].buffer.readAsync(0, layout.cellCount * 4),
        visibleBuffer?.readAsync(0, layout.cellCount * 4)
      ]);
      this.readbackBytes += countBytes.byteLength + (visibleBytes?.byteLength ?? 0);
      if (this.destroyed || sequence !== state.sequence) return;
      state.labels = getAggregateCountLabels(
        layout,
        new Uint32Array(countBytes.slice().buffer),
        visibleBytes ? new Uint32Array(visibleBytes.slice().buffer) : undefined,
        this.projection.commonOrigin,
        this.projection.commonUnitsPerMeter
      );
      this.props.requestUpdate?.();
    })()
      .catch(() => {
        // A retired program or lost device simply leaves the labels as they were.
      })
      .finally(() => this.reads.delete(pending));
    this.reads.add(pending);
  }
  /** Labels of the color column when its aggregate is a categorical mode result. */
  private getCategoryLabels(program: Program): readonly string[] | undefined {
    return this.getChannelCategoryLabels(program.settings.advanced?.color);
  }
  private getChannelCategoryLabels(
    channel: {aggregation: string; weightColumn?: string} | undefined
  ): readonly string[] | undefined {
    return channel?.aggregation === 'mode' && channel.weightColumn
      ? this.props.getCategoryLabels?.(channel.weightColumn)
      : undefined;
  }
  private findProgramId(program: Program): string {
    for (const [id, candidate] of this.programs) if (candidate === program) return id;
    return '';
  }
  get accountedResidentByteLength(): number {
    return this.released
      ? 0
      : this.uploadBytes +
          [...this.widenedColumns.values()].reduce(
            (total, entry) =>
              total +
              (entry.value?.vector.data.reduce(
                (bytes, chunk) => bytes + chunk.buffer.byteLength,
                0
              ) ?? 0),
            0
          ) +
          [...this.programs.values(), ...this.retiredPrograms].reduce(
            (total, program) =>
              total +
              program.aggregation.accountedResidentByteLength +
              (program.statistics?.accountedResidentByteLength ?? 0) +
              program.buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) +
              program.compiled.stats.physicalTransientResourceBytes,
            0
          );
  }
  get readbackByteLength(): number {
    return this.readbackBytes;
  }
  /** Exactly one aggregate, not a source row. Stale/empty picks never publish values. */
  async readCell(layerId: string, cellId: number): Promise<AggregationCell | null> {
    const program = this.programs.get(layerId);
    if (
      this.destroyed ||
      !program ||
      !program.readable ||
      this.reads.size >= this.maximumConcurrentReads ||
      !Number.isInteger(cellId) ||
      cellId < 0 ||
      cellId >= program.aggregation.layout.cellCount
    )
      return null;
    const revision = program.revision;
    const pending = Promise.allSettled([
      program.aggregation.counts.data[0].buffer.readAsync(cellId * 4, 4),
      (program.statistics?.colorValues ?? program.aggregation.values.data[0].buffer).readAsync(
        cellId * 4,
        4
      ),
      program.aggregation.overflow.data[0].buffer.readAsync(0, 4)
    ]);
    this.reads.add(pending);
    this.readbackBytes += 12;
    try {
      const results = await pending;
      if (this.destroyed || this.programs.get(layerId) !== program || program.revision !== revision)
        return null;
      const failure = results.find(result => result.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
      const [counts, values, overflow] = results.map(result =>
        result.status === 'fulfilled' ? result.value : new Uint8Array()
      );
      if (new DataView(overflow.buffer, overflow.byteOffset, 4).getUint32(0, true))
        throw new Error('Aggregate lattice overflow; result is unavailable.');
      const count = new DataView(counts.buffer, counts.byteOffset, 4).getUint32(0, true);
      const value = new DataView(values.buffer, values.byteOffset, 4).getFloat32(0, true);
      return count
        ? {
            layerId,
            cellId,
            count,
            value,
            statistic: program.settings.advanced?.color.aggregation ?? program.settings.statistic,
            label: this.getCategoryLabels(program)?.[value],
            revision
          }
        : null;
    } finally {
      this.reads.delete(pending);
    }
  }
  remove(layerId: string): void {
    const program = this.programs.get(layerId);
    if (program) {
      this.programs.delete(layerId);
      this.retire(program);
    }
  }
  destroy(afterSubmittedWork?: Promise<unknown>): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    const fence = afterSubmittedWork ? undefined : this.props.device.createFence();
    this.destruction = Promise.race([
      Promise.allSettled([
        afterSubmittedWork ?? fence!.signaled,
        ...this.reads,
        ...this.retirements,
        ...[...this.widenedColumns.values()].flatMap(entry =>
          entry.pending ? [entry.pending] : []
        )
      ]),
      this.deviceLost
    ]).then(() => {
      fence?.destroy();
      for (const program of new Set([...this.programs.values(), ...this.retiredPrograms]))
        program.destroy();
      this.programs.clear();
      this.retiredPrograms.clear();
      this.positions.destroy();
      for (const entry of this.widenedColumns.values()) entry.value?.destroy();
      this.widenedColumns.clear();
      for (const reservation of this.sourceReservations) reservation.release();
      this.released = true;
    });
    return this.destruction;
  }
  private retire(program: Program): void {
    this.retiredPrograms.add(program);
    const fence = this.props.device.createFence();
    const pending = Promise.race([
      Promise.allSettled([fence.signaled, ...this.reads]),
      this.deviceLost
    ])
      .then(() => {
        program.destroy();
        this.retiredPrograms.delete(program);
        fence.destroy();
      })
      .finally(() => this.retirements.delete(pending));
    this.retirements.add(pending);
  }
  /** Reuse the certified, registry-accounted public GPU cast used by chart measures. */
  private getNumericColumns(settings: NativeAggregationSettings) {
    const columns = {...this.props.columns};
    const names = new Set([
      settings.weightColumn,
      ...[settings.advanced?.color, settings.advanced?.elevation].flatMap(channel =>
        channel && !['count', 'mode', 'countUnique'].includes(channel.aggregation)
          ? [channel.weightColumn]
          : []
      )
    ]);
    let pending = false;
    for (const name of names) {
      if (!name) continue;
      const column = columns[name];
      if (!column || (column.values.format !== 'sint32' && column.values.format !== 'uint32'))
        continue;
      let entry = this.widenedColumns.get(name);
      if (!entry) {
        if (!this.props.memoryRegistry)
          throw new Error(
            'Integer aggregation weights require a memory registry for exact widening.'
          );
        entry = {};
        this.widenedColumns.set(name, entry);
        const created = entry;
        created.pending = widenResidentChartMeasure({
          session: {device: this.props.device, registry: this.props.memoryRegistry},
          vector: column.values as GPUVector<'sint32'> | GPUVector<'uint32'>
        })
          .then(value => {
            if (this.destroyed) value.destroy();
            else created.value = value;
          })
          .catch(error => {
            created.reason = error instanceof Error ? error.message : String(error);
          })
          .finally(() => {
            created.pending = undefined;
            if (!this.destroyed) this.props.requestUpdate?.();
          });
      }
      if (entry.reason) throw new Error(`${name}: ${entry.reason}`);
      if (entry.value) columns[name] = {...column, values: entry.value.vector};
      else pending = true;
    }
    return pending ? undefined : columns;
  }
  private createProgram(
    id: string,
    key: string,
    settings: NativeAggregationSettings,
    sourceSelectionMask: GPUVector<'uint32'>,
    sourceBinOrderMask: GPUVector<'uint32'> | undefined,
    columns: NonNullable<AggregationRuntime['props']['columns']>
  ): Program {
    const topology = getNativeAggregationTopology(this.projection, settings);
    const layout = getKeplerAggregationLayout(topology);
    if (layout.cellCount > (this.props.maximumCells ?? 65536))
      throw new Error(
        'Requested aggregate size exceeds the configured cell budget. Increase cell size.'
      );
    const weight = settings.weightColumn ? columns[settings.weightColumn] : undefined;
    // The advanced statistics pipeline converts integer and dictionary-code columns to Float32,
    // so mode/countUnique can aggregate categorical (dictionary-coded) fields; sum/mean paths need Float32.
    if (
      weight &&
      weight.values.format !== 'float32' &&
      !(
        settings.advanced &&
        (weight.values.format === 'uint32' || weight.values.format === 'sint32')
      )
    )
      throw new Error('Resident aggregation currently requires Float32 weights.');
    // Conservative preflight includes contributor scratch and polygon/color output capacities.
    // Weighted advanced statistics sort every row by cell key, so reserve their working set.
    const weightedChannels = [settings.advanced?.color, settings.advanced?.elevation].filter(
      channel => channel && channel.aggregation !== 'count'
    ).length;
    const requiredBudget =
      this.positions.length * (32 + weightedChannels * 96) +
      layout.cellCount * (1024 + (settings.advanced ? 512 : 0)) +
      4 * 1024 * 1024;
    const budget = this.props.programMemoryBudgetBytes ?? requiredBudget;
    if (budget < requiredBudget)
      throw new Error(
        'Aggregation program memory budget is below its required preflight capacity.'
      );
    const reservation = this.props.memoryRegistry?.reserveMemory(budget, 'gpu', 'scratch');
    let aggregation: GPUKeplerAggregation;
    try {
      aggregation = new GPUKeplerAggregation({
        device: this.props.device,
        id,
        positions: this.positions,
        selectionMask: sourceSelectionMask,
        // Format was checked above; keep the generic narrowing at this adapter boundary.
        weights: weight?.values as GPUVector<'float32'> | undefined,
        validity: weight?.validity,
        topology,
        statistic: settings.statistic
      });
    } catch (error) {
      reservation?.release();
      throw error;
    }
    const buffers: Buffer[] = [];
    let compiled: CompiledGPUCommandGraph | undefined;
    let statistics: AggregationStatistics | undefined;
    try {
      const geometry = aggregation.getPolygonGeometry();
      const verticesPerCell = settings.kind === 'grid' ? 4 : 6;
      const vertices = layout.cellCount * verticesPerCell;
      const allocate = (
        name: string,
        length: number,
        values?: Float32Array | Uint32Array
      ): Buffer => {
        const buffer = this.props.device.createBuffer({
          id: `${id}-${name}`,
          ...(values ? {data: values} : {byteLength: Math.max(4, length * 4)}),
          usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return buffer;
      };
      const graph = new GPUCommandGraph(this.props.device, {id: `${id}-color`});
      const importBuffer = <Format extends 'uint32' | 'float32'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ): GraphDataView<Format> =>
        graph.createDataView(
          graph.importBuffer(
            {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          ),
          {format, length}
        );
      const advanced = settings.advanced;
      let cellColors: GraphDataView<'uint32'>;
      if (advanced) {
        aggregation.prepare();
        statistics = new AggregationStatistics({
          device: this.props.device,
          id,
          positions: this.positions,
          selectionMask: sourceSelectionMask,
          binOrderMask: sourceBinOrderMask,
          layout,
          counts: aggregation.counts.data[0],
          columns,
          plan: advanced,
          palette: settings.palette,
          categoryCounts: {
            color: this.getChannelCategoryLabels(advanced.color)?.length,
            elevation: this.getChannelCategoryLabels(advanced.elevation)?.length
          }
        });
        cellColors = importBuffer('cell-colors', statistics.colors, 'uint32', layout.cellCount);
      } else {
        const values = graph.importGPUData('values', aggregation.values.data[0]);
        const mask = graph.importGPUData('counts', aggregation.counts.data[0]);
        const domain = importBuffer(
          'domain',
          allocate('domain', settings.palette.length + 1),
          'float32',
          settings.palette.length + 1
        );
        const classCount = importBuffer('class-count', allocate('class-count', 1), 'uint32', 1);
        const method = settings.colorScale === 'quantile' ? 'quantile' : 'equal-interval';
        const breaksParameters = getGPUClassBreaksParameterValues(
          {method, classCount: settings.palette.length},
          settings.palette.length
        );
        graph.add(
          new GPUClassBreaks({
            id: `${id}-breaks`,
            values,
            mask,
            parameters: importBuffer(
              'breaks-parameters',
              allocate('breaks-parameters', breaksParameters.length, breaksParameters),
              'float32',
              breaksParameters.length
            ),
            maximumClassCount: settings.palette.length,
            methods: [method],
            output: {breaks: domain, classCount}
          })
        );
        const colorParameters = getGPUColorScaleParameterValues({
          scale: settings.colorScale,
          paletteCount: settings.palette.length,
          domainCount: settings.palette.length + 1,
          interpolation: settings.colorScale === 'linear' ? 'linear' : 'step',
          clamp: true
        });
        cellColors = importBuffer(
          'cell-colors',
          allocate('cell-colors', layout.cellCount),
          'uint32',
          layout.cellCount
        );
        graph.add(
          new GPUColorScale({
            id: `${id}-colors`,
            values,
            mask,
            domain,
            domainCount: classCount,
            palette: importBuffer(
              'palette',
              allocate('palette', settings.palette.length, Uint32Array.from(settings.palette)),
              'uint32',
              settings.palette.length
            ),
            parameters: importBuffer(
              'color-parameters',
              allocate('color-parameters', colorParameters.length, colorParameters),
              'float32',
              colorParameters.length
            ),
            maximumDomainCount: settings.palette.length + 1,
            maximumPaletteCount: settings.palette.length,
            output: {colors: cellColors}
          })
        );
      }
      const vertexColorBuffer = allocate('vertex-colors', vertices);
      const rowBuffer = geometry.rowIndices.data[0].buffer;
      if (!(rowBuffer instanceof Buffer))
        throw new Error('Aggregate geometry requires a fixed row buffer.');
      graph.add(
        new GPUGather({
          id: `${id}-expand-colors`,
          source: cellColors,
          indices: importBuffer('vertex-rows', rowBuffer, 'uint32', vertices),
          output: importBuffer('vertex-colors', vertexColorBuffer, 'uint32', vertices)
        })
      );
      compiled = graph.compile();
      const offsets = geometry.positions.data[0].valueOffsets!;
      const colorVector = new GPUVector<VertexList<'unorm8x4'>>({
        type: 'data',
        format: 'vertex-list<unorm8x4>',
        name: `${id}-polygon-colors`,
        ownsData: false,
        data: [
          new GPUData({
            buffer: vertexColorBuffer,
            format: 'vertex-list<unorm8x4>',
            length: layout.cellCount,
            valueLength: vertices,
            valueOffsets: offsets,
            ownsBuffer: false
          })
        ]
      });
      let positions = geometry.positions;
      if (settings.coverage !== 1) {
        const footprint = makeAggregateFootprints(layout, settings.coverage);
        const buffer = allocate('covered-positions', footprint.length, footprint);
        positions = new GPUVector({
          type: 'data',
          name: `${id}-covered-polygons`,
          format: 'vertex-list<float32x4>',
          ownsData: false,
          data: [
            new GPUData({
              buffer,
              format: 'vertex-list<float32x4>',
              length: layout.cellCount,
              valueLength: vertices,
              valueOffsets: offsets,
              ownsBuffer: false
            })
          ]
        });
      }
      const scale = this.projection.commonUnitsPerMeter;
      const extruded = Boolean(advanced?.extruded);
      // Percentile/cutoff-hidden cells leave the nonempty-cell mask; heights are meters, so the
      // model matrix scales Z like X and Y while extruded.
      const wrapCellVector = <Format extends 'uint32' | 'float32'>(
        name: string,
        buffer: Buffer,
        format: Format
      ): GPUVector<Format> =>
        new GPUVector({
          type: 'data',
          name: `${id}-${name}`,
          format,
          ownsData: false,
          data: [new GPUData({buffer, format, length: layout.cellCount, ownsBuffer: false})]
        });
      const binding: ResidentAggregationBinding = {
        ...geometry,
        ...(statistics
          ? {selectionMask: wrapCellVector('visible', statistics.visible, 'uint32')}
          : {}),
        positions,
        getFillColor: colorVector,
        ...(extruded && statistics?.elevations
          ? {
              getElevation: wrapCellVector('elevation', statistics.elevations, 'float32'),
              extrusionRings: makeAggregateRings(
                this.props.device,
                id,
                layout,
                settings.coverage,
                buffers
              )
            }
          : {}),
        coordinateSystem: 'cartesian',
        coordinateOrigin: this.projection.commonOrigin,
        modelMatrix: [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, extruded ? scale : 1, 0, 0, 0, 0, 1],
        layout,
        counts: aggregation.counts,
        values: aggregation.values,
        contract: 'mercator-centroid-meters-float32',
        extruded
      };
      aggregation.prepare();
      const residentBytes =
        aggregation.accountedResidentByteLength +
        (statistics?.accountedResidentByteLength ?? 0) +
        compiled.stats.physicalTransientResourceBytes +
        buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
      if (residentBytes > budget)
        throw new KeplerAllocationError('memory-budget', {
          message: 'Aggregation program exceeds its reserved memory budget.'
        });
      const ownedCompiled = compiled;
      const countLabels: CountLabelState = {wanted: false, stale: true, sequence: 0};
      binding.getCountLabels = () => {
        if (!countLabels.wanted) {
          countLabels.wanted = true;
          this.props.requestUpdate?.();
        }
        return countLabels.labels;
      };
      let released = false;
      return {
        key,
        sourceSelectionMask,
        sourceBinOrderMask,
        readable: false,
        aggregation,
        compiled,
        statistics,
        buffers,
        binding,
        settings,
        revision: 0,
        scaleLookupStale: true,
        scaleLookupSequence: 0,
        countLabels,
        destroy() {
          if (released) return;
          released = true;
          ownedCompiled.destroy();
          statistics?.destroy();
          aggregation.destroy();
          for (const buffer of buffers) buffer.destroy();
          reservation?.release();
        }
      };
    } catch (error) {
      compiled?.destroy();
      statistics?.destroy();
      aggregation.destroy();
      for (const buffer of buffers) buffer.destroy();
      reservation?.release();
      throw error;
    }
  }
}

/**
 * Closed per-cell rings (first vertex repeated) for native extrusion walls. Rings follow the
 * covered footprint, are counter-clockwise, and each ring's source row is its cell id.
 */
function makeAggregateRings(
  device: Device,
  id: string,
  layout: GPUKeplerAggregationLayout,
  coverage: number,
  buffers: Buffer[]
): NonNullable<ResidentAggregationBinding['extrusionRings']> {
  const vertexCount = layout.kind === 'grid' ? 4 : 6;
  const footprint = makeAggregateFootprints(layout, coverage);
  const ringVertexCount = vertexCount + 1;
  const rings = new Float32Array(layout.cellCount * ringVertexCount * 3);
  for (let cell = 0; cell < layout.cellCount; cell++)
    for (let vertex = 0; vertex < ringVertexCount; vertex++) {
      const source = (cell * vertexCount + (vertex % vertexCount)) * 4;
      rings.set(
        [footprint[source], footprint[source + 1], 0],
        (cell * ringVertexCount + vertex) * 3
      );
    }
  const sourceRows = Uint32Array.from({length: layout.cellCount}, (_, cell) => cell);
  const create = (name: string, data: Float32Array | Uint32Array): Buffer => {
    const buffer = device.createBuffer({
      id: `${id}-${name}`,
      data,
      usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return buffer;
  };
  return {
    positions: new GPUVector({
      type: 'data',
      name: `${id}-extrusion-rings`,
      format: 'vertex-list<float32x3>',
      ownsData: false,
      data: [
        new GPUData({
          buffer: create('extrusion-rings', rings),
          format: 'vertex-list<float32x3>',
          length: layout.cellCount,
          valueLength: layout.cellCount * ringVertexCount,
          valueOffsets: Int32Array.from(
            {length: layout.cellCount + 1},
            (_, cell) => cell * ringVertexCount
          ),
          ownsBuffer: false
        })
      ]
    }),
    sourceRowIds: new GPUVector({
      type: 'data',
      name: `${id}-extrusion-sources`,
      format: 'uint32',
      ownsData: false,
      data: [
        new GPUData({
          buffer: create('extrusion-sources', sourceRows),
          format: 'uint32',
          length: layout.cellCount,
          ownsBuffer: false
        })
      ]
    }),
    windingOrder: 'CCW'
  };
}

export function makeAggregateFootprints(
  layout: GPUKeplerAggregationLayout,
  coverage: number
): Float32Array {
  const vertexCount = layout.kind === 'grid' ? 4 : 6;
  const positions = new Float32Array(layout.cellCount * vertexCount * 4);
  for (let cell = 0; cell < layout.cellCount; cell++) {
    const column = cell % layout.gridSize[0];
    const row = Math.floor(cell / layout.gridSize[0]);
    const center =
      layout.kind === 'grid'
        ? [
            layout.bounds[0] + (column + 0.5) * layout.cellSize![0],
            layout.bounds[1] + (row + 0.5) * layout.cellSize![1]
          ]
        : getGPUPointDensityHexagonCenter(
            column,
            row,
            layout.bounds[0],
            layout.bounds[1],
            layout.radius!
          );
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const angle = (vertex * Math.PI) / 3 + Math.PI / 2;
      const offset =
        layout.kind === 'grid'
          ? [
              (vertex === 0 || vertex === 3 ? -0.5 : 0.5) * layout.cellSize![0],
              (vertex < 2 ? -0.5 : 0.5) * layout.cellSize![1]
            ]
          : [Math.cos(angle) * layout.radius!, Math.sin(angle) * layout.radius!];
      positions.set(
        [center[0] + offset[0] * coverage, center[1] + offset[1] * coverage, 0, 1],
        (cell * vertexCount + vertex) * 4
      );
    }
  }
  return positions;
}
function parseHexColor(value: string): number {
  if (!/^#[0-9a-f]{6}([0-9a-f]{2})?$/i.test(value))
    throw new Error('Resident aggregate palettes require #RRGGBB or #RRGGBBAA colors.');
  return packGPUColor(
    Number.parseInt(value.slice(1, 3), 16),
    Number.parseInt(value.slice(3, 5), 16),
    Number.parseInt(value.slice(5, 7), 16),
    value.length === 9 ? Number.parseInt(value.slice(7, 9), 16) : 255
  );
}
