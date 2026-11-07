// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCatchmentSummary over the basins GPUTerrainWatersheds delineates from pour points along
// the valley: per basin elevation statistics and a hypsometric histogram (the bars, one group of
// histogramBins per basin, colored by cell count). The basins are drawn faintly behind.

import {
  GPU_TERRAIN_FLOW_NONE,
  GPUTerrainCatchmentSummary,
  GPUTerrainWatersheds
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-catchment-summary.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  findValleyCell,
  getTerrainContextLayer
} from './shared/terrain';

const BASIN_CAPACITY = 8;

export default defineOperationRun({
  parameters,
  caption: 'Hypsometric histogram of every basin: how much of its area lies at each elevation.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    const pourPoints = new Uint32Array(BASIN_CAPACITY);
    for (let index = 0; index < BASIN_CAPACITY; index++) {
      pourPoints[index] = findValleyCell(terrain, 0.97 - 0.1 * index);
    }
    return {...terrain, pourPoints};
  },
  build(builder, terrain, values) {
    const flow = addTerrainFlow(builder, terrain);
    flow.writeStreamThreshold(25);
    const histogramBins = Math.round(values.histogramBins);
    const pourPoints = builder.allocate('pour-points', 'uint32', BASIN_CAPACITY);
    const basins = builder.allocate('basins', 'uint32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainWatersheds({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        pourPoints: pourPoints.view,
        output: {labels: basins.view}
      })
    );
    const hypsometry = builder.allocate('hypsometry', 'float32', BASIN_CAPACITY * histogramBins);
    builder.graph.add(
      new GPUTerrainCatchmentSummary({
        grid: terrain.grid,
        elevation: flow.filledElevation.view,
        basinIds: basins.view,
        basinCount: BASIN_CAPACITY,
        streams: flow.streams.view,
        flowDirections: flow.flowDirections.view,
        histogramBins,
        output: {
          statistics: builder.allocate('statistics', 'float32', BASIN_CAPACITY * 6).view,
          hypsometry: hypsometry.view,
          profiles: builder.allocate('profiles', 'float32', terrain.cellCount * 3).view,
          profileStatus: builder.allocate('profile-status', 'uint32', terrain.cellCount).view
        }
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, basins, hypsometry},
      update: current => {
        const active = new Uint32Array(BASIN_CAPACITY).fill(GPU_TERRAIN_FLOW_NONE);
        active.set(terrain.pourPoints.subarray(0, Math.round(current.pourPointCount)));
        pourPoints.write(active);
      }
    };
  },
  rebuildOn: ['histogramBins'],
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      context: true,
      values: outputs.basins,
      columns: inputs.columns,
      rows: inputs.rows,
      alpha: 0.35
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.hypsometry,
      ramp: 'categorical',
      rectangle: [0.05, 0.06, 0.95, 0.5]
    }
  ]
});
