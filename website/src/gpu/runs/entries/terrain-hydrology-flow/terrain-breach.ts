// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainBreach on the raw DEM: from each pit (strict local minimum) it cuts the cheapest
// bounded channel to lower ground. The picture is the depth it cut; pits are the dots. A pit whose
// outlet is beyond maximumSteps or needs more than maximumCutDepth stays closed.

import {GPUTerrainBreach} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-breach.parameters';
import {createCraterTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const PIT_LIMIT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Cut depth along the channel from each pit (dots) to its outlet.',
  createInputs: context => {
    const terrain = createCraterTerrainInputs(context);
    // Spread the pits over the whole map, then place their dots in drawing coordinates.
    const stride = Math.max(1, Math.floor(terrain.pitCells.length / PIT_LIMIT));
    const pitCells = terrain.pitCells
      .filter((_, index) => index % stride === 0)
      .slice(0, PIT_LIMIT);
    const pitPositions = new Float32Array(pitCells.length * 2);
    pitCells.forEach((cell, index) => {
      pitPositions[index * 2] = ((cell % terrain.columns) + 0.5) * terrain.cellSize;
      pitPositions[index * 2 + 1] =
        context.world.height - (Math.floor(cell / terrain.columns) + 0.5) * terrain.cellSize;
    });
    return {...terrain, pitCells, pitPositions};
  },
  build(builder, terrain, values) {
    const pitCount = terrain.pitCells.length;
    const maximumSteps = Math.round(values.maximumSteps);
    const elevation = builder.upload('dem', 'float32', terrain.elevation);
    const pitCells = builder.upload('pit-cells', 'uint32', terrain.pitCells);
    const breached = builder.allocate('breached', 'float32', terrain.cellCount);
    const cutDepth = builder.allocate('cut-depth', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainBreach({
        grid: terrain.grid,
        elevation: elevation.view,
        pitCells: pitCells.view,
        maximumSteps,
        maximumCutDepth: values.maximumCutDepth,
        elevationDecrement: 1e-5,
        output: {
          elevation: breached.view,
          cutDepth: cutDepth.view,
          modificationMask: builder.allocate('mask', 'uint32', terrain.cellCount).view,
          pathCells: builder.allocate('path-cells', 'uint32', pitCount * (maximumSteps + 1)).view,
          pathLengths: builder.allocate('path-lengths', 'uint32', pitCount).view,
          pitStatus: builder.allocate('pit-status', 'uint32', pitCount).view,
          unresolvedCount: builder.allocate('unresolved-count', 'uint32', 1).view
        }
      })
    );
    const pits = builder.upload('pits', 'float32x2', terrain.pitPositions);
    return {outputs: {elevation, cutDepth, pits}};
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.elevation),
    {
      layer: 'raster',
      values: outputs.cutDepth,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 0.02],
      fade: true
    },
    {layer: 'dots', context: true, positions: outputs.pits, radius: 4, color: [1, 1, 1, 0.9]}
  ]
});
