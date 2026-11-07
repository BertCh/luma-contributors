// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Zone rasters for the change-of-support runs: Voronoi zones of jittered sites over a run grid,
// plus a one-line GPU gather that expands a per-zone table to a per-cell raster for drawing.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
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

/** Cell count per zone. */
export function countZoneCells(raster: ZoneRaster): Float32Array {
  const counts = new Float32Array(raster.zoneCount);
  for (const zone of raster.zones) counts[zone]++;
  return counts;
}

/** Adds a node writing `table[zones[cell]]` for every cell (NaN for cells without a zone), so a zone result can be drawn as a raster. */
export function addZoneGather(
  builder: RunBuilder,
  zones: RunBuffer<'uint32'>,
  table: RunBuffer<'float32'>,
  id: string
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
          {name: 'zoneRaster', view: raster.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: zones.length,
        body: `let zone = zoneIds[zoneIdsOffset + index];
zoneRaster[zoneRasterOffset + index] = select(
  bitcast<f32>(zone | 0x7fc00000u), zoneTable[zoneTableOffset + min(zone, ${table.length - 1}u)], zone < ${table.length}u);`
      })
    ]
  });
  return raster;
}
