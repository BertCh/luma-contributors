// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer as DeckLayer} from '@deck.gl/core';
import {PathLayer, PolygonLayer} from '@deck.gl/layers';
import type {FlowFieldRuntime} from './flow-field-runtime';

/** Row 0 is south. `altitude` NaN (or omitted) means no elevation for the cell. */
export type FlowFieldDebugGridSnapshot = {
  columns: number;
  rows: number;
  bounds: [number, number, number, number];
  filled: Uint8Array;
  altitude?: Float32Array;
};

export type FlowFieldDebugGridGeometry = {
  gridLines: {path: [number, number, number][]}[];
  filledCells: {polygon: [number, number, number][]}[];
};

function getAxis(start: number, end: number, count: number): number[] {
  return Array.from({length: count}, (_, index) =>
    count > 1 ? start + (index * (end - start)) / (count - 1) : start
  );
}

/** Midpoint cell edges around axis node `index` (half-way to its neighbors), as native. */
function getAxisCellEdges(values: number[], index: number): [number, number] {
  const last = values.length - 1;
  return [
    index === 0 ? values[0] : (values[index - 1] + values[index]) / 2,
    index === last ? values[last] : (values[index] + values[index + 1]) / 2
  ];
}

/** Mirrors the native `buildDebugGridGeometry`; z is altitude times the elevation multiplier. */
export function buildFlowFieldDebugGridGeometry(
  snapshot: FlowFieldDebugGridSnapshot,
  elevationMultiplier: number
): FlowFieldDebugGridGeometry {
  const {columns, rows, bounds, filled, altitude} = snapshot;
  const longitudes = getAxis(bounds[0], bounds[2], columns);
  const latitudes = getAxis(bounds[1], bounds[3], rows);
  const getZ = (column: number, row: number): number => {
    if (!altitude) return 0;
    const value = altitude[row * columns + column];
    return Number.isFinite(value) ? value * elevationMultiplier : 0;
  };
  const gridLines: FlowFieldDebugGridGeometry['gridLines'] = [];
  for (let column = 0; column < columns; column++) {
    const path: [number, number, number][] = [];
    for (let row = 0; row < rows; row++)
      path.push([longitudes[column], latitudes[row], getZ(column, row)]);
    if (path.length >= 2) gridLines.push({path});
  }
  for (let row = 0; row < rows; row++) {
    const path: [number, number, number][] = [];
    for (let column = 0; column < columns; column++)
      path.push([longitudes[column], latitudes[row], getZ(column, row)]);
    if (path.length >= 2) gridLines.push({path});
  }
  const filledCells: FlowFieldDebugGridGeometry['filledCells'] = [];
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      if (!filled[row * columns + column]) continue;
      const [west, east] = getAxisCellEdges(longitudes, column);
      const [south, north] = getAxisCellEdges(latitudes, row);
      const z = getZ(column, row);
      filledCells.push({
        polygon: [
          [west, south, z],
          [east, south, z],
          [east, north, z],
          [west, north, z],
          [west, south, z]
        ]
      });
    }
  return {gridLines, filledCells};
}

export type FlowFieldDebugGridLayerOptions = {
  layerId: string;
  elevationMultiplier: number;
  parameters?: Record<string, unknown>;
};

/** Native debug overlay: grid lines as a PathLayer and filled cells as a PolygonLayer. */
export function buildFlowFieldDebugGridLayers(
  snapshot: FlowFieldDebugGridSnapshot,
  options: FlowFieldDebugGridLayerOptions
): DeckLayer[] {
  const {gridLines, filledCells} = buildFlowFieldDebugGridGeometry(
    snapshot,
    options.elevationMultiplier
  );
  const parameters = {depthTest: false, ...options.parameters};
  return [
    new PathLayer({
      id: `${options.layerId}-debug-grid`,
      data: gridLines,
      getPath: (cell: FlowFieldDebugGridGeometry['gridLines'][number]) => cell.path,
      getColor: [120, 140, 160, 120],
      getWidth: 1,
      widthUnits: 'pixels',
      pickable: false,
      wrapLongitude: false,
      parameters
    } as never),
    new PolygonLayer({
      id: `${options.layerId}-debug-filled`,
      data: filledCells,
      getPolygon: (cell: FlowFieldDebugGridGeometry['filledCells'][number]) => cell.polygon,
      getFillColor: [80, 220, 140, 90],
      getLineColor: [40, 180, 110, 200],
      lineWidthMinPixels: 1,
      stroked: true,
      filled: true,
      pickable: false,
      wrapLongitude: false,
      parameters
    } as never)
  ];
}

/**
 * Debug-only readback of the current grid (one GPU readback for a GPU grid). Never called while
 * rendering and never counted in `gridReadbackCount`.
 */
export async function readFlowFieldDebugGrid(
  runtime: FlowFieldRuntime
): Promise<FlowFieldDebugGridSnapshot | undefined> {
  const {columns, rows, bounds} = runtime.grid;
  const gpuGrid = runtime.gpu?.grid as
    | (NonNullable<FlowFieldRuntime['gpu']>['grid'] & {
        readAltitudeForDebug?: () => Promise<Float32Array | undefined>;
      })
    | undefined;
  const texels = gpuGrid ? await gpuGrid.readTexelsForDebug() : runtime.grid.texels;
  if (!texels) return undefined;
  const filled = new Uint8Array(columns * rows);
  for (let index = 0; index < filled.length; index++)
    filled[index] = texels[index * 4 + 2] > 0.5 ? 1 : 0;
  const altitude = gpuGrid
    ? await gpuGrid.readAltitudeForDebug?.()
    : (runtime.grid as {altitude?: Float32Array}).altitude;
  return {
    columns,
    rows,
    bounds: [...bounds] as [number, number, number, number],
    filled,
    ...(altitude ? {altitude} : {})
  };
}

type DebugGridEntry = {
  runtime: FlowFieldRuntime;
  revision: number | null;
  boundsKey: string;
  snapshot?: FlowFieldDebugGridSnapshot;
  pending: boolean;
  layers?: {elevationMultiplier: number; snapshot: FlowFieldDebugGridSnapshot; value: DeckLayer[]};
};

/**
 * Per-layer asynchronous debug snapshots. Rendering reads the latest snapshot and never awaits; a
 * new readback starts only when the runtime, selection revision or bounds change.
 */
export class FlowFieldDebugGridCache {
  /** Debug readbacks started; separate from the runtime's `gridReadbackCount`. */
  debugReadCount = 0;
  private readonly entries = new Map<string, DebugGridEntry>();

  getLayers(
    layerId: string,
    runtime: FlowFieldRuntime,
    options: {
      requestUpdate: () => void;
      elevationMultiplier: number;
      parameters?: Record<string, unknown>;
    }
  ): DeckLayer[] {
    const revision = runtime.stats.selectionRevision;
    const boundsKey = runtime.grid.bounds.join(',');
    let entry = this.entries.get(layerId);
    if (!entry || entry.runtime !== runtime) {
      entry = {runtime, revision: null, boundsKey: '', pending: false};
      this.entries.set(layerId, entry);
    }
    const stale = entry.revision !== revision || entry.boundsKey !== boundsKey || !entry.snapshot;
    if (stale && !entry.pending) {
      const target = entry;
      target.pending = true;
      target.revision = revision;
      target.boundsKey = boundsKey;
      this.debugReadCount++;
      void readFlowFieldDebugGrid(runtime)
        .then(snapshot => {
          if (this.entries.get(layerId) !== target) return;
          if (snapshot) target.snapshot = snapshot;
        })
        .catch(() => undefined)
        .finally(() => {
          target.pending = false;
          if (this.entries.get(layerId) === target) options.requestUpdate();
        });
    }
    const snapshot = entry.snapshot;
    if (!snapshot) return [];
    if (
      !entry.layers ||
      entry.layers.snapshot !== snapshot ||
      entry.layers.elevationMultiplier !== options.elevationMultiplier
    )
      entry.layers = {
        snapshot,
        elevationMultiplier: options.elevationMultiplier,
        value: buildFlowFieldDebugGridLayers(snapshot, {
          layerId,
          elevationMultiplier: options.elevationMultiplier,
          parameters: options.parameters
        })
      };
    return entry.layers.value;
  }

  release(layerId: string): void {
    this.entries.delete(layerId);
  }
}
