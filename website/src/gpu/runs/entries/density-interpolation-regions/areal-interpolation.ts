// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUArealInterpolation: source zones (muted) re-aggregated onto different target zones. The
// operation writes source-target overlap weights; GPUSpatialLag applies them to the source values,
// and the per-target result is expanded to cells for drawing. The service areas are clipped to the
// tract footprint, so both sides of the curtain cover the same land, and their boundaries are drawn
// over the tracts.

import {
  GPUArealInterpolation,
  GPUSpatialLag
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './areal-interpolation.parameters';
import {createRunGrid, type RunGrid} from './support/grid';
import {makeRandom} from '../../../demos/kit';
import {findPolygonAt} from '../../pick';
import {createTracts} from '../../inputs';
import {
  addTractZones,
  addZoneGather,
  countTractCells,
  countZoneCells,
  createVoronoiZones
} from './support/zones';

const NO_ZONE = 0xffffffff;
/** Service areas end this far across the world, so part of the city has no target zone. */
const COVERAGE_WEST = 0.25;

/** Boundary segments (x0, y0, x1, y1) between cells of different zones, neither side empty. */
function getZoneBoundaries(grid: RunGrid, zones: Uint32Array): Float32Array {
  const segments: number[] = [];
  const {columns, rows, cellSize, centers} = grid;
  const half = cellSize / 2;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      const zone = zones[cell];
      if (zone === NO_ZONE) continue;
      const x = centers[cell * 2];
      const y = centers[cell * 2 + 1];
      const right = column + 1 < columns ? zones[cell + 1] : NO_ZONE;
      const below = row + 1 < rows ? zones[cell + columns] : NO_ZONE;
      if (right !== NO_ZONE && right !== zone)
        segments.push(x + half, y - half, x + half, y + half);
      if (below !== NO_ZONE && below !== zone)
        segments.push(x - half, y - half, x + half, y - half);
    }
  }
  return Float32Array.from(segments);
}

export default defineOperationRun({
  parameters,
  caption:
    'Chicago tract populations or poverty shares transferred to illustrative service areas by overlap.',
  createInputs: ({random, world}) => {
    const grid = createRunGrid(world, 64);
    const tracts = createTracts(random, world, 8);
    // Cells inside any tract: the land both zone systems share.
    const covered = new Uint8Array(grid.columns * grid.rows);
    for (let cell = 0; cell < covered.length; cell++) {
      const x = grid.centers[cell * 2];
      const y = grid.centers[cell * 2 + 1];
      covered[cell] =
        findPolygonAt(tracts.vertices, tracts.triangleIndices, tracts.triangleOwners, x, y) >= 0
          ? 1
          : 0;
    }
    return {grid, tracts, world, covered, tractCells: countTractCells(grid, tracts)};
  },
  build(builder, {grid, tracts, world, covered, tractCells}, current) {
    const target = createVoronoiZones(
      makeRandom(100 + current.targetZoneCount),
      grid,
      world,
      current.targetZoneCount
    );
    // Service areas cover the city only, and stop partway across it from the west, so part of
    // the tract population has no target zone (partial coverage).
    for (let cell = 0; cell < target.zones.length; cell++) {
      if (!covered[cell] || grid.centers[cell * 2] < world.width * COVERAGE_WEST) {
        target.zones[cell] = NO_ZONE;
      }
    }
    const capacity = Math.min(tracts.count * target.zoneCount, grid.columns * grid.rows);
    const sourceZones = addTractZones(builder, grid, tracts);
    const targetZones = builder.upload('target-zones', 'uint32', target.zones);
    const sourceValues = builder.upload(
      'source-values',
      'float32',
      current.mode === 0 ? tracts.population : tracts.povertyShare
    );
    const weights = {
      offsets: builder.allocate('offsets', 'uint32', target.zoneCount + 1),
      neighbors: builder.allocate('neighbors', 'uint32', capacity),
      weights: builder.allocate('weights', 'float32', capacity)
    };
    const targetValues = builder.allocate('target-values', 'float32', target.zoneCount);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUArealInterpolation({
        grid: grid.grid,
        sourceZones: sourceZones.view,
        targetZones: targetZones.view,
        sourceZoneCount: tracts.count,
        targetZoneCount: target.zoneCount,
        mode: current.mode === 0 ? 'extensive' : 'intensive',
        denominator: current.denominator === 0 ? 'zone' : 'overlap',
        output: {
          weights: {
            offsets: weights.offsets.view,
            neighbors: weights.neighbors.view,
            weights: weights.weights.view
          },
          status: {overflow: overflow.view}
        }
      })
    );
    builder.graph.add(
      new GPUSpatialLag({
        weights: {
          offsets: weights.offsets.view,
          neighbors: weights.neighbors.view,
          weights: weights.weights.view,
          status: {overflow: overflow.view}
        },
        values: sourceValues.view,
        output: targetValues.view
      })
    );
    // Extensive totals are compared per cell, so a big zone does not look brighter than a small one;
    // both rasters then share one range, up to the densest source tract.
    const isExtensive = current.mode === 0;
    const sourceValueArray = isExtensive ? tracts.population : tracts.povertyShare;
    const sourceDivisors = builder.upload('source-cells', 'float32', tractCells);
    const targetDivisors = builder.upload(
      'target-cells',
      'float32',
      countZoneCells(target.zones, target.zoneCount)
    );
    let rangeMaximum = 0;
    sourceValueArray.forEach((value, tract) => {
      rangeMaximum = Math.max(
        rangeMaximum,
        isExtensive ? value / Math.max(1, tractCells[tract]) : value
      );
    });
    return {
      outputs: {
        sourceRaster: addZoneGather(
          builder,
          sourceZones,
          sourceValues,
          'source-raster',
          isExtensive ? sourceDivisors : undefined
        ),
        resultRaster: addZoneGather(
          builder,
          targetZones,
          targetValues,
          'result-raster',
          isExtensive ? targetDivisors : undefined
        ),
        rangeMaximum,
        targetBoundaries: builder.upload(
          'target-boundaries',
          'float32x4',
          getZoneBoundaries(grid, target.zones)
        )
      }
    };
  },
  // A curtain: the tract values on the left, the service-area result of the same operation on
  // the right, on one range.
  interaction: {
    kind: 'swipe',
    hint: 'swipe tracts and areas',
    rest: [0.5, 0.5],
    resolve: (pointer, {values}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `tracts | ${parameters[0].options?.[values.mode]} over ${values.targetZoneCount} zones`
    })
  },
  draw: ({inputs, outputs, values, interaction, world}) => {
    const curtain = values.show === 1 ? (interaction.curtain ?? world.width * 0.5) : world.width;
    const range = [0, outputs.rangeMaximum * 0.5] as const;
    return [
      {
        layer: 'raster' as const,
        values: outputs.sourceRaster,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows,
        range,
        clip: {side: 'left' as const, x: curtain}
      },
      ...(values.show === 1
        ? [
            {
              layer: 'raster' as const,
              values: outputs.resultRaster,
              columns: inputs.grid.columns,
              rows: inputs.grid.rows,
              range,
              clip: {side: 'right' as const, x: curtain}
            },
            {
              layer: 'lines' as const,
              segments: outputs.targetBoundaries,
              color: [1, 1, 1, 1] as const,
              alpha: 0.55
            }
          ]
        : [])
    ];
  }
});
