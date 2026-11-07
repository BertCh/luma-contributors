// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainWatersheds on the D8 directions of GPUTerrainFlow: every cell is labelled with the
// pour point its flow path reaches. The slider activates more of the pour points in the view;
// unused rows hold the "none" marker and cells that reach no pour point stay unlabelled.

import {GPU_TERRAIN_FLOW_NONE, GPUTerrainWatersheds} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-watersheds.parameters';
import {addTerrainFlow, createTerrainInputs, findValleyCell} from './shared/terrain';

const POUR_POINT_CAPACITY = 12;

export default defineOperationRun({
  parameters,
  caption: 'Each color is the area draining to one pour point.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    // Pour points down the trunk valley, so each takes the area between it and the next.
    const pourPoints = new Uint32Array(POUR_POINT_CAPACITY);
    for (let index = 0; index < POUR_POINT_CAPACITY; index++) {
      pourPoints[index] = findValleyCell(terrain, 0.97 - 0.075 * index);
    }
    return {...terrain, pourPoints};
  },
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    flow.writeStreamThreshold(25);
    const pourPoints = builder.allocate('pour-points', 'uint32', POUR_POINT_CAPACITY);
    const labels = builder.allocate('labels', 'uint32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainWatersheds({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        pourPoints: pourPoints.view,
        output: {labels: labels.view}
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, labels},
      update: values => {
        const active = new Uint32Array(POUR_POINT_CAPACITY).fill(GPU_TERRAIN_FLOW_NONE);
        active.set(terrain.pourPoints.subarray(0, Math.round(values.pourPointCount)));
        pourPoints.write(active);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.filledElevation,
      columns: inputs.columns,
      rows: inputs.rows,
      hillshade: 0.8,
      elevation: 30
    },
    {
      layer: 'raster',
      values: outputs.labels,
      columns: inputs.columns,
      rows: inputs.rows,
      alpha: 0.7
    }
  ]
});
