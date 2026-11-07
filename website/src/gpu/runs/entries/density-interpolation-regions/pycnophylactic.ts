// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPycnophylactic turns zone totals into a smooth surface whose cells still sum to each zone's
// total. The muted raster is the zone-uniform density; the output relaxes it across the borders.

import {GPUPycnophylactic} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {parameters} from './pycnophylactic.parameters';
import {createRunGrid} from './support/grid';
import {countZoneCells, createVoronoiZones} from './support/zones';

export default defineOperationRun({
  parameters,
  caption: 'Zone totals relaxed into a surface that still sums to each total.',
  createInputs: ({world}) => ({grid: createRunGrid(world, 40), world}),
  build(builder, {grid, world}, values) {
    const raster = createVoronoiZones(
      makeRandom(11 + values.zoneCount),
      grid,
      world,
      values.zoneCount
    );
    const cellCounts = countZoneCells(raster);
    const totals = Float32Array.from({length: raster.zoneCount}, (_, zone) => {
      const x = raster.sites[zone * 2] / world.width;
      const y = raster.sites[zone * 2 + 1];
      return cellCounts[zone] * (0.2 + 1.6 * Math.max(0, Math.sin(x * 4.5 + y * 2) ** 2));
    });
    const surface = builder.allocate('surface', 'float32', grid.columns * grid.rows);
    builder.graph.add(
      new GPUPycnophylactic({
        grid: grid.grid,
        zones: builder.upload('zones', 'uint32', raster.zones).view,
        zoneCount: raster.zoneCount,
        totals: builder.upload('totals', 'float32', totals).view,
        maximumIterations: values.maximumIterations,
        kernel: values.kernel === 0 ? 'rook' : 'box',
        output: {mass: surface.view}
      })
    );
    const uniform = Float32Array.from(raster.zones, zone => totals[zone] / cellCounts[zone]);
    return {outputs: {surface, uniform: builder.upload('uniform', 'float32', uniform)}};
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.uniform,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows
    },
    {
      layer: 'raster',
      values: outputs.surface,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      smoothing: true
    }
  ]
});
