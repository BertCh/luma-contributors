// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {FlowFieldAnalysisSnapshot} from './flow-field-grid';
import {buildFlowFieldGrid, type FlowFieldColumnMode, type FlowFieldGrid} from './flow-field-grid';
import {
  getFlowFieldGridKey,
  planFlowField,
  type FlowFieldUnitPolicy,
  type FlowFieldVisConfig
} from './flow-field-plan';
import {FlowFieldRuntime, getFlowFieldSeed} from './flow-field-runtime';
import {getFlowFieldExtent, type FlowFieldExtent} from './flow-field-gpu-reference';
import {FlowFieldGpuGrid, makeFlowFieldGpuColumns} from './flow-field-gpu';
import {getFlowFieldBinCounts} from './flow-field-grid';
import {FlowFieldCentroidCache} from './flow-field-centroids';
import {FlowFieldDebugGridCache} from './flow-field-debug-grid';
import type {Layer as DeckLayer} from '@deck.gl/core';
import {
  FlowFieldSelectionCache,
  getRemovedLayerIds,
  type FlowFieldSelectionContext
} from './flow-field-filter';

/** Structural subset of a native Flow Field layer configuration. */
export type FlowFieldLayerConfig = {
  columnMode?: string;
  columns: {[name: string]: {value?: string | null} | undefined};
  visConfig: FlowFieldVisConfig;
  color: readonly number[];
};

export type FlowFieldResolution = {
  runtime?: FlowFieldRuntime;
  bounds?: [number, number, number, number];
  /** Why nothing can render, or null. */
  reason: string | null;
  /** Native options rendered with reduced fidelity. */
  limits: string[];
  /** Debug grid overlay layers from the latest asynchronous snapshot (never awaits). */
  getDebugLayers?: (elevationMultiplier: number) => DeckLayer[];
};

const LEGACY_MODES: Record<string, FlowFieldColumnMode> = {
  GEOJSON_UV: 'UV',
  GEOJSON_SPEED_DIR: 'SPEED_DIR',
  GEOJSON_ELEVATION: 'ELEVATION'
};

export function normalizeFlowFieldMode(mode: string | undefined): FlowFieldColumnMode | null {
  if (!mode) return 'UV';
  if (mode === 'UV' || mode === 'SPEED_DIR' || mode === 'ELEVATION') return mode;
  return LEGACY_MODES[mode] ?? null;
}

/** Column roles required by each native mode, in sample order (first, second). */
export function getFlowFieldValueColumns(
  mode: FlowFieldColumnMode,
  columns: FlowFieldLayerConfig['columns']
): [string | undefined, string | undefined] {
  const name = (role: string) => columns[role]?.value ?? undefined;
  return mode === 'UV'
    ? [name('u'), name('v')]
    : mode === 'SPEED_DIR'
      ? [name('speed'), name('direction')]
      : [name('altitude'), undefined];
}

/** Native rule: lat/lng win when both are set; the GeoJSON column is the fallback position source. */
export function getFlowFieldPositionColumns(columns: FlowFieldLayerConfig['columns']): {
  longitude?: string;
  latitude?: string;
  geojson?: string;
} {
  const longitude = columns['lng']?.value ?? undefined;
  const latitude = columns['lat']?.value ?? undefined;
  if (longitude && latitude) return {longitude, latitude};
  const geojson = columns['geojson']?.value ?? undefined;
  return geojson ? {geojson} : {};
}

function readNumbers(table: arrow.Table, name: string): Float64Array | null {
  const vector = table.getChild(name);
  if (!vector) return null;
  const values = new Float64Array(table.numRows);
  for (let index = 0; index < values.length; index++) {
    const value: unknown = vector.get(index);
    values[index] =
      typeof value === 'number' ? value : typeof value === 'bigint' ? Number(value) : NaN;
  }
  return values;
}

const GEOJSON_POSITION_PREFIX = 'geojson:';

type CachedField = {
  key: string;
  table: arrow.Table;
  grid: FlowFieldGrid | null;
  selectionRevision: number | null;
  selectionMask?: GPUVector<'uint32'>;
};
type CpuFieldRequest = {
  key: string;
  config: FlowFieldLayerConfig;
  context: {
    device: Device;
    table: arrow.Table | undefined;
    selection?: FlowFieldSelectionContext;
  };
};
type CachedRuntime = {
  mode: FlowFieldColumnMode;
  key: string;
  device: Device;
  runtime: FlowFieldRuntime;
  table?: arrow.Table;
};
type CachedGpuField = {
  key: string;
  table: arrow.Table;
  extent: FlowFieldExtent | null;
  columns: number;
  rows: number;
};

/** Builds and caches one field grid and GPU particle runtime per Flow Field layer. */
export class FlowFieldBindings {
  unitPolicy: FlowFieldUnitPolicy = 'visual';
  /** `auto`: GPU-resident grid on WebGPU, CPU grid otherwise. `cpu`: always the CPU reference path. */
  gridMode: 'auto' | 'cpu' = 'auto';
  private readonly gpuFields = new Map<string, CachedGpuField>();
  private readonly gpuFailures = new Map<string, string>();
  private readonly fields = new Map<string, CachedField>();
  private readonly cpuRequests = new Map<string, CpuFieldRequest>();
  private readonly runtimes = new Map<string, CachedRuntime>();
  private readonly selections = new FlowFieldSelectionCache();
  private readonly centroids = new FlowFieldCentroidCache();
  readonly debugGrids = new FlowFieldDebugGridCache();

  resolve(
    layerId: string,
    config: FlowFieldLayerConfig,
    context: {
      device: Device;
      table: arrow.Table | undefined;
      /** Resident row mask of the layer; omitted when the table has no resident filter. */
      selection?: FlowFieldSelectionContext;
      /** Redraw callback for asynchronous debug grid snapshots. */
      requestUpdate?: () => void;
    }
  ): FlowFieldResolution {
    const resolution = this.resolveField(layerId, config, context);
    if (resolution.runtime) {
      resolution.runtime.onViewportAvailabilityChange = context.requestUpdate;
      const failure = resolution.runtime.stats.viewportFailure;
      if (failure) resolution.limits.push(failure);
    }
    if (resolution.runtime && context.requestUpdate) {
      const {runtime} = resolution;
      const requestUpdate = context.requestUpdate;
      resolution.getDebugLayers = elevationMultiplier =>
        this.debugGrids.getLayers(layerId, runtime, {requestUpdate, elevationMultiplier});
    }
    return resolution;
  }

  private resolveField(
    layerId: string,
    config: FlowFieldLayerConfig,
    context: {
      device: Device;
      table: arrow.Table | undefined;
      selection?: FlowFieldSelectionContext;
    }
  ): FlowFieldResolution {
    const none = (reason: string): FlowFieldResolution => ({reason, limits: []});
    const mode = normalizeFlowFieldMode(config.columnMode);
    if (!mode) return none('This flow field column mode has no resident binding.');
    if (!context.table)
      return none('No retained source table is bound to this flow field dataset.');
    if (!FlowFieldRuntime.isSupported(context.device))
      return none('GPU flow fields need a device that can render rgba32float targets.');
    const position = getFlowFieldPositionColumns(config.columns);
    // GeoJSON centroid positions are keyed as `geojson:<column>` in both position slots.
    const longitude = position.geojson ? `geojson:${position.geojson}` : position.longitude;
    const latitude = position.geojson ? longitude : position.latitude;
    if (!longitude || !latitude)
      return none(
        'Select latitude and longitude columns (or a GeoJSON column) for the flow field.'
      );
    const centroidLimit = position.geojson
      ? this.getCentroidLimit(layerId, context.table, position.geojson)
      : undefined;
    const [firstName, secondName] = getFlowFieldValueColumns(mode, config.columns);
    if (!firstName || (mode !== 'ELEVATION' && !secondName))
      return none('Select the vector field columns required by the flow field mode.');
    const visConfig = config.visConfig;
    // Native streamline Z: ELEVATION mode uses its altitude column, other modes an optional one.
    const altitudeName =
      mode === 'ELEVATION' ? undefined : (config.columns['altitude']?.value ?? undefined);
    let gpuLimit: string | undefined;
    if (this.gridMode === 'auto' && context.device.type === 'webgpu') {
      const failure = this.gpuFailures.get(layerId);
      if (!failure) {
        try {
          const resolution = this.resolveGpu(layerId, mode, visConfig, context, [
            longitude,
            latitude,
            firstName,
            secondName,
            altitudeName
          ]);
          if (resolution) {
            this.cpuRequests.delete(layerId);
            return resolution;
          }
        } catch (error) {
          this.releaseGpu(layerId);
          this.gpuFailures.set(layerId, error instanceof Error ? error.message : String(error));
        }
      }
      const reason = this.gpuFailures.get(layerId);
      if (reason) gpuLimit = `GPU field grid unavailable, using the CPU reference grid: ${reason}`;
    }
    const selection = this.selections.resolve(layerId, context.selection);
    const sourceKey = getFlowFieldGridKey(
      mode,
      [longitude, latitude, firstName, secondName, altitudeName],
      visConfig
    );
    this.cpuRequests.set(layerId, {key: sourceKey, config, context});
    const gridKey = `${sourceKey}|${selection.key}|mask:${selection.appliedGeneration}`;
    let field = this.fields.get(layerId);
    if (
      !field ||
      field.key !== gridKey ||
      field.table !== context.table ||
      field.selectionMask !== selection.appliedMask
    ) {
      const {longitudes, latitudes} = this.readPositions(
        layerId,
        context.table,
        longitude,
        latitude
      );
      const first = readNumbers(context.table, firstName);
      const second = secondName ? readNumbers(context.table, secondName) : undefined;
      const altitude = altitudeName ? readNumbers(context.table, altitudeName) : undefined;
      if (
        !longitudes ||
        !latitudes ||
        !first ||
        (secondName && !second) ||
        (altitudeName && !altitude)
      )
        return none('A flow field column is missing from the retained source.');
      field = {
        key: gridKey,
        table: context.table,
        selectionRevision: selection.appliedRevision,
        selectionMask: selection.appliedMask,
        grid: buildFlowFieldGrid(
          {
            longitudes,
            latitudes,
            first,
            second: second ?? undefined,
            altitude: altitude ?? undefined,
            selected: selection.words
          },
          {
            mode,
            gridResolution: visConfig.gridResolution ?? 64,
            smoothing: visConfig.smoothing ?? 4
          }
        )
      };
      this.fields.set(layerId, field);
    }
    if (!field.grid)
      return none(
        'The flow field needs at least four finite samples spanning a non-degenerate area within 85 degrees latitude.'
      );
    const plan = planFlowField(
      visConfig,
      {bounds: field.grid.bounds, maximumSpeed: field.grid.maximumSpeed},
      this.unitPolicy
    );
    const runtimeKey = `${gridKey}|${plan.particleCount}|${plan.lifetimeSeconds}|${plan.trailSeconds}|${plan.widthPixels}|${plan.opacity}|${plan.velocityScale}`;
    let cached = this.runtimes.get(layerId);
    if (
      !cached ||
      cached.key !== runtimeKey ||
      cached.device !== context.device ||
      cached.table !== context.table
    ) {
      cached?.runtime.destroy();
      cached = {
        mode,
        key: runtimeKey,
        device: context.device,
        table: context.table,
        runtime: new FlowFieldRuntime(
          context.device,
          layerId,
          field.grid,
          plan,
          getFlowFieldSeed(layerId),
          undefined,
          context.selection?.memoryRegistry
        )
      };
      this.runtimes.set(layerId, cached);
    }
    return {
      runtime: cached.runtime,
      bounds: field.grid.bounds,
      reason: null,
      limits: [
        ...plan.limits,
        ...(centroidLimit ? [centroidLimit] : []),
        ...(gpuLimit ? [gpuLimit] : []),
        ...(selection.failure
          ? [`Layer filters were not applied to the field: ${selection.failure}`]
          : [])
      ]
    };
  }

  /**
   * GPU-resident path: the extent comes from one CPU scan of the unfiltered table, then every
   * mask revision re-bins on the GPU into the same texture. Returns null for degenerate data
   * (the caller reports the same reason as the CPU path).
   */
  private resolveGpu(
    layerId: string,
    mode: FlowFieldColumnMode,
    visConfig: FlowFieldLayerConfig['visConfig'],
    context: {
      device: Device;
      table: arrow.Table | undefined;
      selection?: FlowFieldSelectionContext;
    },
    names: readonly (string | undefined)[]
  ): FlowFieldResolution | null {
    const table = context.table as arrow.Table;
    const [longitude, latitude, firstName, secondName, altitudeName] = names as [
      string,
      string,
      string,
      string?,
      string?
    ];
    this.selections.release(layerId);
    const centroidLimit = longitude.startsWith(GEOJSON_POSITION_PREFIX)
      ? this.getCentroidLimit(layerId, table, longitude.slice(GEOJSON_POSITION_PREFIX.length))
      : undefined;
    const fieldKey = getFlowFieldGridKey(mode, names, visConfig);
    let field = this.gpuFields.get(layerId);
    const readSamples = () => {
      const {longitudes, latitudes} = this.readPositions(layerId, table, longitude, latitude);
      const first = readNumbers(table, firstName);
      const second = secondName ? readNumbers(table, secondName) : undefined;
      const altitude = altitudeName ? readNumbers(table, altitudeName) : undefined;
      if (!longitudes || !latitudes || !first || (secondName && !second)) return null;
      if (altitudeName && !altitude) return null;
      return {
        longitudes,
        latitudes,
        first,
        second: second ?? undefined,
        altitude: altitude ?? undefined
      };
    };
    if (!field || field.key !== fieldKey || field.table !== table) {
      const samples = readSamples();
      if (!samples)
        return {reason: 'A flow field column is missing from the retained source.', limits: []};
      const extent = getFlowFieldExtent(samples);
      const counts = extent
        ? getFlowFieldBinCounts(...extent.bounds, visConfig.gridResolution ?? 64)
        : {columns: 0, rows: 0};
      field = {key: fieldKey, table, extent, ...counts};
      this.gpuFields.set(layerId, field);
      this.releaseRuntime(layerId);
    }
    if (!field.extent) return null;
    const plan = planFlowField(visConfig, {bounds: field.extent.bounds}, this.unitPolicy);
    const runtimeKey = `${fieldKey}|gpu|${plan.particleCount}|${plan.lifetimeSeconds}|${plan.trailSeconds}|${plan.widthPixels}|${plan.opacity}|${plan.velocityScaleNumerator}`;
    let cached = this.runtimes.get(layerId);
    if (
      !cached ||
      cached.key !== runtimeKey ||
      cached.device !== context.device ||
      cached.table !== table
    ) {
      cached?.runtime.destroy();
      this.runtimes.delete(layerId);
      const samples = readSamples();
      if (!samples) return null;
      const altitudeSource = mode === 'ELEVATION' ? 'first' : samples.altitude ? 'column' : 'none';
      const memoryReservation = context.selection?.memoryRegistry?.reserveMemory(
        FlowFieldGpuGrid.estimateByteLength(
          table.numRows,
          field.columns,
          field.rows,
          altitudeSource
        ),
        'gpu',
        'scratch'
      );
      const grid = new FlowFieldGpuGrid(
        context.device,
        layerId,
        makeFlowFieldGpuColumns(samples, field.extent),
        {
          mode,
          columns: field.columns,
          rows: field.rows,
          extent: field.extent,
          smoothing: visConfig.smoothing ?? 4,
          velocityScaleNumerator: plan.velocityScaleNumerator,
          altitude: altitudeSource,
          memoryReservation
        }
      );
      cached = {
        mode,
        key: runtimeKey,
        device: context.device,
        table,
        runtime: new FlowFieldRuntime(
          context.device,
          layerId,
          {columns: field.columns, rows: field.rows, bounds: field.extent.bounds},
          plan,
          getFlowFieldSeed(layerId),
          {grid, selection: context.selection}
        )
      };
      this.runtimes.set(layerId, cached);
    }
    // The latest mask drives the next frame's rebuild directly; nothing is read back.
    if (cached.runtime.gpu) cached.runtime.gpu.selection = context.selection;
    return {
      runtime: cached.runtime,
      bounds: field.extent.bounds,
      reason: null,
      limits: [...plan.limits, ...(centroidLimit ? [centroidLimit] : [])]
    };
  }

  private readPositions(
    layerId: string,
    table: arrow.Table,
    longitude: string,
    latitude: string
  ): {longitudes: Float64Array | null; latitudes: Float64Array | null} {
    if (longitude.startsWith(GEOJSON_POSITION_PREFIX)) {
      const {longitudes, latitudes} = this.centroids.get(
        layerId,
        table,
        longitude.slice(GEOJSON_POSITION_PREFIX.length)
      );
      return {longitudes, latitudes};
    }
    return {longitudes: readNumbers(table, longitude), latitudes: readNumbers(table, latitude)};
  }

  private getCentroidLimit(
    layerId: string,
    table: arrow.Table,
    columnName: string
  ): string | undefined {
    const {truncatedCount} = this.centroids.get(layerId, table, columnName);
    return truncatedCount
      ? `GeoJSON centroids were read for the first ${table.numRows - truncatedCount} rows only; ${truncatedCount} rows are left out of the field.`
      : undefined;
  }

  private releaseRuntime(layerId: string): void {
    this.runtimes.get(layerId)?.runtime.destroy();
    this.runtimes.delete(layerId);
  }

  private releaseGpu(layerId: string): void {
    this.releaseRuntime(layerId);
    this.gpuFields.delete(layerId);
  }

  /** Test and diagnostics snapshot of live particle runtimes. */
  getStats(): Record<
    string,
    FlowFieldRuntime['stats'] & {columns: number; rows: number; mode: FlowFieldColumnMode}
  > {
    return Object.fromEntries(
      [...this.runtimes].map(([layerId, {runtime, mode}]) => {
        const request = this.cpuRequests.get(layerId);
        const field = this.fields.get(layerId);
        const selection = request?.context.selection;
        const gridReady = request
          ? Boolean(
              field?.grid &&
                field.selectionRevision === (selection?.getRevision() ?? null) &&
                field.selectionMask === selection?.mask
            )
          : runtime.isGridReady;
        return [
          layerId,
          {
            ...runtime.stats,
            mode,
            gridReady,
            columns: runtime.grid.columns,
            rows: runtime.grid.rows
          }
        ];
      })
    );
  }

  /** Explicit analysis snapshot. Reads the physical field before GPU visual scaling. */
  async readGridForAnalysis(layerId: string): Promise<FlowFieldAnalysisSnapshot> {
    const runtime = this.runtimes.get(layerId)?.runtime;
    if (!runtime) throw new Error('The flow layer has no active velocity grid');
    if (runtime.grid.texels) {
      const request = this.cpuRequests.get(layerId);
      if (!request) throw new Error('The CPU flow layer has no source for analysis');
      await this.selections.waitForCurrent(layerId, request.context.selection);
      const latest = this.cpuRequests.get(layerId);
      if (
        !latest ||
        latest.key !== request.key ||
        latest.context.table !== request.context.table ||
        latest.context.device !== request.context.device ||
        latest.context.selection?.mask !== request.context.selection?.mask
      )
        throw new Error('Flow layer changed while awaiting its analysis mask');
      // The normal renderer may not have run since readback completed. Build from the applied
      // current words now, through the same binding and grid builder used for rendering.
      const resolution = this.resolveField(layerId, latest.config, latest.context);
      const field = this.fields.get(layerId);
      if (!resolution.runtime?.grid.texels || !field?.grid)
        throw new Error(resolution.reason ?? 'The filtered CPU flow grid is unavailable');
      const selection = latest.context.selection;
      if (
        field.selectionRevision !== (selection?.getRevision() ?? null) ||
        field.selectionMask !== selection?.mask
      )
        throw new Error('Flow selection changed while preparing its analysis grid');
      return {
        ...field.grid,
        mode: this.runtimes.get(layerId)!.mode,
        bounds: [...field.grid.bounds],
        texels: field.grid.texels.slice()
      };
    }
    await runtime.waitForGridReady();
    if (this.runtimes.get(layerId)?.runtime !== runtime)
      throw new Error('Flow layer changed while awaiting its analysis grid');
    if (!runtime.gpu) throw new Error('The flow layer has no GPU velocity grid');
    const bounds = [...runtime.grid.bounds] as [number, number, number, number];
    const revision = runtime.stats.selectionRevision;
    const buildCount = runtime.stats.gridBuildCount;
    const texels = await runtime.gpu.grid.readVelocitiesForAnalysis();
    if (
      this.runtimes.get(layerId)?.runtime !== runtime ||
      !runtime.isGridReady ||
      revision !== runtime.stats.selectionRevision ||
      buildCount !== runtime.stats.gridBuildCount ||
      (runtime.gpu.selection && runtime.gpu.selection.getRevision() !== revision) ||
      bounds.some((value, index) => value !== runtime.grid.bounds[index])
    )
      throw new Error('Flow grid changed during snapshot; run again');
    return {
      ...runtime.grid,
      mode: this.runtimes.get(layerId)!.mode,
      bounds,
      texels,
      minimumSpeed: 0,
      maximumSpeed: 0,
      sampleCount: 0,
      skippedCount: 0
    };
  }

  /** Test-only: reads back the GPU velocity texels (never used while rendering). */
  async readGridForDebug(layerId: string): Promise<Float32Array | undefined> {
    return this.runtimes.get(layerId)?.runtime.gpu?.grid.readTexelsForDebug();
  }

  /** Releases the layer's GPU state (layer removal or device loss). */
  release(layerId: string): void {
    this.releaseGpu(layerId);
    this.gpuFailures.delete(layerId);
    this.fields.delete(layerId);
    this.cpuRequests.delete(layerId);
    this.selections.release(layerId);
    this.centroids.release(layerId);
    this.debugGrids.release(layerId);
  }

  /** Releases layers that were removed from the Kepler state (their GPU runtimes and filter words). */
  retainLayers(activeLayerIds: ReadonlySet<string>): void {
    const cached = new Set([
      ...this.runtimes.keys(),
      ...this.fields.keys(),
      ...this.gpuFields.keys()
    ]);
    for (const layerId of getRemovedLayerIds(cached, activeLayerIds)) this.release(layerId);
  }

  destroy(): void {
    for (const layerId of [...this.runtimes.keys()]) this.release(layerId);
  }
}

/** Shared native flow workflow state, independent of layer registration and host publication. */
export const flowFieldBindings = new FlowFieldBindings();
