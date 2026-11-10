// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Zone rasters for the change-of-support runs: Voronoi zones of jittered sites over a run grid,
// plus a one-line GPU gather that expands a per-zone table to a per-cell raster for drawing.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {
  getGPUPolygonRasterizationParameterValues,
  GPUPolygonRasterization
} from '@luma-contributors/analysis/raster-analysis';
import type {TractsInput} from '../../../inputs';
import {findPolygonAt} from '../../../pick';
import {getPackedPolygons, uploadPackedPolygons} from './geometry';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {RunGrid} from './grid';

export type ZoneRaster = {
  zoneCount: number;
  /** Zone id per cell, row-major, row 0 at the top. */
  zones: Uint32Array;
  /** Site (x, y) of each zone. */
  sites: Float32Array;
};

/** Nearest-site zones of `zoneCount` jittered lattice sites. */
export function createVoronoiZones(
  random: () => number,
  grid: RunGrid,
  world: {width: number; height: number},
  zoneCount: number
): ZoneRaster {
  const sites = new Float32Array(zoneCount * 2);
  const columns = Math.max(1, Math.round(Math.sqrt(zoneCount * world.width)));
  const rows = Math.ceil(zoneCount / columns);
  for (let zone = 0; zone < zoneCount; zone++) {
    sites[zone * 2] = ((zone % columns) + 0.5 + (random() - 0.5) * 0.7) * (world.width / columns);
    sites[zone * 2 + 1] =
      (Math.floor(zone / columns) + 0.5 + (random() - 0.5) * 0.7) * (world.height / rows);
  }
  const zones = new Uint32Array(grid.columns * grid.rows);
  for (let cell = 0; cell < zones.length; cell++) {
    const x = grid.centers[cell * 2];
    const y = grid.centers[cell * 2 + 1];
    let best = 0;
    let bestDistance = Infinity;
    for (let zone = 0; zone < zoneCount; zone++) {
      const distance = Math.hypot(x - sites[zone * 2], y - sites[zone * 2 + 1]);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = zone;
      }
    }
    zones[cell] = best;
  }
  return {zoneCount, zones, sites};
}

/** Burn Census polygons into the operation's north-up grid. Empty cells retain the no-zone sentinel. */
export function addTractZones(builder: RunBuilder, grid: RunGrid, tracts: TractsInput) {
  const cells = grid.rows * grid.columns;
  const zones = builder.allocate('tract-zones', 'uint32', cells);
  builder.graph.add(
    new GPUPolygonRasterization({
      ...uploadPackedPolygons(builder, getPackedPolygons(tracts)),
      grid: grid.grid,
      parameters: builder.upload(
        'extent',
        'float32',
        getGPUPolygonRasterizationParameterValues(grid.grid)
      ).view,
      crossingCapacity: cells * 4,
      output: {
        zones: zones.view,
        boundary: builder.allocate('boundary', 'uint32', cells).view,
        status: {overflow: builder.allocate('tract-zone-overflow', 'uint32', 1).view}
      }
    })
  );
  return zones;
}

/** Adds a node writing `table[zones[cell]]` for every cell (NaN for cells without a zone), so a zone result can be drawn as a raster. */
export function addZoneGather(
  builder: RunBuilder,
  zones: RunBuffer<'uint32'>,
  table: RunBuffer<'float32'>,
  id: string,
  /** Optional per-zone divisors (cells per zone): the raster then holds the value per cell. */
  divisors?: RunBuffer<'float32'>
): RunBuffer<'float32'> {
  const raster = builder.allocate(id, 'float32', zones.length);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'zone-gather',
        bindings: [
          {name: 'zoneIds', view: zones.view, type: 'u32', access: 'read'},
          {name: 'zoneTable', view: table.view, type: 'f32', access: 'read'},
          ...(divisors
            ? [
                {
                  name: 'zoneDivisors',
                  view: divisors.view,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'zoneRaster', view: raster.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: zones.length,
        body: `let zone = zoneIds[zoneIdsOffset + index];
let safeZone = min(zone, ${table.length - 1}u);
let zoneValue = zoneTable[zoneTableOffset + safeZone]${divisors ? ' / max(zoneDivisors[zoneDivisorsOffset + safeZone], 1.0)' : ''};
zoneRaster[zoneRasterOffset + index] = select(bitcast<f32>(zone | 0x7fc00000u), zoneValue, zone < ${table.length}u);`
      })
    ]
  });
  return raster;
}

/** Grid cells whose center lies in each tract (CPU, for turning zone totals into per-cell values). */
export function countTractCells(grid: RunGrid, tracts: TractsInput): Float32Array {
  const counts = new Float32Array(tracts.count);
  for (let cell = 0; cell < grid.columns * grid.rows; cell++) {
    const tract = findPolygonAt(
      tracts.vertices,
      tracts.triangleIndices,
      tracts.triangleOwners,
      grid.centers[cell * 2],
      grid.centers[cell * 2 + 1]
    );
    if (tract >= 0) counts[tract]++;
  }
  return counts;
}

/** Grid cells in each of `zoneCount` zones of a zone raster. */
export function countZoneCells(zones: Uint32Array, zoneCount: number): Float32Array {
  const counts = new Float32Array(zoneCount);
  for (const zone of zones) if (zone < zoneCount) counts[zone]++;
  return counts;
}
