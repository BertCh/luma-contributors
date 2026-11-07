// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUArealInterpolation: source zones (muted) re-aggregated onto different target zones. The
// operation writes source-target overlap weights; GPUSpatialLag applies them to the source values,
// and the per-target result is expanded to cells for drawing.

import {
  GPUArealInterpolation,
  GPUSpatialLag
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './areal-interpolation.parameters';
import {createRunGrid} from './support/grid';
import {makeRandom} from '../../../demos/kit';
import {addZoneGather, createVoronoiZones} from './support/zones';

const SOURCE_ZONE_COUNT = 18;

export default defineOperationRun({
  parameters,
  caption: 'Source-zone values moved onto the target zones by overlap area.',
  createInputs: ({random, world}) => {
    const grid = createRunGrid(world, 36);
    const source = createVoronoiZones(random, grid, world, SOURCE_ZONE_COUNT);
    const values = Float32Array.from({length: SOURCE_ZONE_COUNT}, (_, zone) => {
      const x = source.sites[zone * 2] / world.width;
      const y = source.sites[zone * 2 + 1];
      return 60 + 140 * (0.5 + 0.5 * Math.sin(x * 5) * Math.cos(y * 4)) + random() * 20;
    });
    return {grid, source, values, world};
  },
  build(builder, {grid, source, values, world}, current) {
    const target = createVoronoiZones(
      makeRandom(100 + current.targetZoneCount),
      grid,
      world,
      current.targetZoneCount
    );
    // The target excludes a disc (a lake): cells there belong to no target zone.
    grid.centers.forEach((_, index) => {
      if (index % 2 === 1) return;
      const cell = index / 2;
      const distance = Math.hypot(
        grid.centers[index] - 0.5 * world.width,
        grid.centers[index + 1] - 0.5
      );
      if (distance < 0.22) target.zones[cell] = 0xffffffff;
    });
    const capacity = Math.min(source.zoneCount * target.zoneCount, grid.columns * grid.rows);
    const sourceZones = builder.upload('source-zones', 'uint32', source.zones);
    const targetZones = builder.upload('target-zones', 'uint32', target.zones);
    const sourceValues = builder.upload('source-values', 'float32', values);
    const weights = {
      offsets: builder.allocate('offsets', 'uint32', target.zoneCount + 1),
      neighbors: builder.allocate('neighbors', 'uint32', capacity),
      weights: builder.allocate('weights', 'float32', capacity)
    };
    const targetValues = builder.allocate('target-values', 'float32', target.zoneCount);
    builder.graph.add(
      new GPUArealInterpolation({
        grid: grid.grid,
        sourceZones: sourceZones.view,
        targetZones: targetZones.view,
        sourceZoneCount: source.zoneCount,
        targetZoneCount: target.zoneCount,
        mode: current.mode === 0 ? 'extensive' : 'intensive',
        denominator: current.denominator === 0 ? 'zone' : 'overlap',
        output: {
          weights: {
            offsets: weights.offsets.view,
            neighbors: weights.neighbors.view,
            weights: weights.weights.view
          },
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    builder.graph.add(
      new GPUSpatialLag({
        weights: {
          offsets: weights.offsets.view,
          neighbors: weights.neighbors.view,
          weights: weights.weights.view
        },
        values: sourceValues.view,
        output: targetValues.view
      })
    );
    // The source as a raster is only an input picture; the result is gathered on the GPU.
    const sourceRaster = Float32Array.from(source.zones, zone => values[zone]);
    return {
      outputs: {
        sourceRaster: builder.upload('source-raster', 'float32', sourceRaster),
        resultRaster: addZoneGather(builder, targetZones, targetValues, 'result-raster')
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.sourceRaster,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows
    },
    ...(values.show === 1
      ? [
          {
            layer: 'raster' as const,
            values: outputs.resultRaster,
            columns: inputs.grid.columns,
            rows: inputs.grid.rows
          }
        ]
      : [])
  ]
});
