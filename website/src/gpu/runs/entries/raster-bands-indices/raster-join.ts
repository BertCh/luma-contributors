// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterJoin behind GPUPolygonRasterization: every point looks up the zone of its raster cell
// (no point-in-polygon test), then counts and sums are reduced per zone. Coarse grids misassign
// points near polygon edges; `boundaryCounts` shows how many points are at risk.

import {GPURasterJoin} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './raster-join.parameters';
import {addZoneRaster, createZoneInputs, drawZones, uploadZoneMesh} from './support/zones';

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'Points in; counts per polygon out, via the rasterized zone grid.',
  createInputs: ({random, world}) => ({
    ...createZoneInputs(random, world),
    points: createUniformPoints(random, world, 2500)
  }),
  build(builder, inputs, values) {
    const {features, points} = inputs;
    const zoneRaster = addZoneRaster(builder, inputs, values.gridWidth);
    const pointPositions = builder.upload('points', 'float32x2', points.positions);
    const pointValues = builder.upload('point-values', 'float32', points.values);
    const zoneCount = features.featureCount;
    const output = {
      counts: builder.allocate('counts', 'uint32', zoneCount),
      sums: builder.allocate('sums', 'float32', zoneCount),
      boundaryCounts: builder.allocate('boundary-counts', 'uint32', zoneCount),
      pointZones: builder.allocate('point-zones', 'uint32', points.count)
    };
    builder.graph.add(
      new GPURasterJoin({
        grid: zoneRaster.grid,
        parameters: zoneRaster.extent.view,
        points: pointPositions.view,
        values: pointValues.view,
        zones: zoneRaster.zones.view,
        boundary: zoneRaster.boundary.view,
        zoneCount,
        output: {
          counts: output.counts.view,
          sums: output.sums.view,
          boundaryCounts: output.boundaryCounts.view,
          pointZones: output.pointZones.view
        }
      })
    );
    return {outputs: {...output, pointPositions, mesh: uploadZoneMesh(builder, features)}};
  },
  draw: ({inputs, outputs, theme}) => [
    drawZones(inputs.features, outputs.mesh, outputs.sums, {ramp: 'sequential', alpha: 0.8}),
    {layer: 'dots', positions: outputs.pointPositions, values: outputs.pointZones, radius: 2},
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.7]}
  ]
});
