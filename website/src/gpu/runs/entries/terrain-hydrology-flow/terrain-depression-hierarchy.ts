// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainDepressionHierarchy on a cratered DEM: every cell is assigned to the depression it
// belongs to (colors), and the area and volume curves of all depressions at the chosen elevation
// stages are written as bars. With too small a nodeCapacity the tree is rejected.

import {GPUTerrainDepressionHierarchy} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun, type RunInputContext} from '../../define-run';
import {parameters} from './terrain-depression-hierarchy.parameters';
import {createCraterTerrainInputs, getTerrainContextLayer} from './shared/terrain';

function createDepressionInputs(context: RunInputContext) {
  const terrain = createCraterTerrainInputs(context, 32, 5, 3, 0.05, 0);
  // Water leaves the map across the whole border.
  const borderMask = new Uint32Array(terrain.cellCount);
  for (let cell = 0; cell < terrain.cellCount; cell++) {
    const column = cell % terrain.columns;
    const row = Math.floor(cell / terrain.columns);
    const onBorder =
      column === 0 || row === 0 || row === terrain.rows - 1 || column === terrain.columns - 1;
    borderMask[cell] = onBorder ? 1 : 0;
  }
  return {...terrain, borderMask};
}

export default defineOperationRun({
  parameters,
  caption: 'Cells colored by the depression that owns them; bars are area and volume per stage.',
  createInputs: context => createDepressionInputs(context),
  build(builder, terrain, values) {
    const stageCount = Math.round(values.stages);
    const nodeCapacity = Math.round(values.nodeCapacity);
    const [low, high] = terrain.elevationRange;
    const stageLevels = Float32Array.from(
      {length: stageCount},
      (_, index) => low + ((index + 1) / stageCount) * (high - low)
    );
    const elevation = builder.upload('dem', 'float32', terrain.elevation);
    const outlets = builder.upload('outlets', 'uint32', terrain.borderMask);
    const cellNodes = builder.allocate('cell-nodes', 'uint32', terrain.cellCount);
    const curves = builder.allocate('curves', 'float32', nodeCapacity * stageCount * 2);
    builder.graph.add(
      new GPUTerrainDepressionHierarchy({
        grid: terrain.grid,
        elevation: elevation.view,
        outlets: outlets.view,
        stages: builder.upload('stages', 'float32', stageLevels).view,
        nodeCapacity,
        output: {
          nodes: builder.allocate('nodes', 'uint32', nodeCapacity * 4).view,
          levels: builder.allocate('levels', 'float32', nodeCapacity * 2).view,
          cellNodes: cellNodes.view,
          curves: curves.view,
          status: builder.allocate('status', 'uint32', 3).view
        }
      })
    );
    return {
      outputs: {elevation, cellNodes, curves}
    };
  },
  rebuildOn: ['nodeCapacity', 'stages'],
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.elevation),
    {
      layer: 'raster',
      values: outputs.cellNodes,
      columns: inputs.columns,
      rows: inputs.rows,
      alpha: 0.7
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.curves,
      ramp: 'sequential',
      rectangle: [0.04, 0.04, 0.96, 0.3]
    }
  ]
});
