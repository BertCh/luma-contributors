// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointPatternIndices: quadrat counts (the grid behind the dots) and each point's distance to its
// nearest neighbor (dot color, blue is close). A clustered pattern has empty quadrats beside full
// ones and short nearest-neighbor distances.

import {
  GPUPointPatternIndices,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  getGPUPointPatternIndicesParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {createPatternSource, getPatternPositions, hashToUnit} from './_lib/helpers';
import {parameters} from './point-pattern-indices.parameters';

const POINT_COUNT = 700;

export default defineOperationRun({
  parameters,
  caption: 'Quadrat counts (grid) and nearest-neighbor distance (dots).',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    world
  }),
  build(builder, inputs, values) {
    const grid = values.quadratGrid;
    const {world} = inputs;
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const mask = builder.allocate('mask', 'uint32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
    );
    const nearest = builder.allocate('nearest', 'float32', POINT_COUNT);
    const quadratCounts = builder.allocate('quadrat-counts', 'uint32', grid * grid);
    builder.graph.add(
      new GPUPointPatternIndices({
        positions: positions.view,
        mask: mask.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        quadratGrid: [grid, grid],
        output: {nearestNeighborDistances: nearest.view, quadratCounts: quadratCounts.view}
      })
    );
    const cells = addFloatCopy(builder, 'cells', quadratCounts, grid * grid, {flipColumns: grid});
    return {
      outputs: {positions, nearest, cells},
      update: current => {
        positions.write(getPatternPositions(inputs.source, world.width, 1, current.clustering));
        mask.write(
          Uint32Array.from({length: POINT_COUNT}, (_, row) =>
            hashToUnit(row) * 100 < current.mask ? 1 : 0
          )
        );
        parameterBuffer.write(
          getGPUPointPatternIndicesParameterValues({
            bounds: [0, 0, world.width, 1],
            maximumDistance: 0.05
          })
        );
      }
    };
  },
  rebuildOn: ['quadratGrid'],
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.cells,
      columns: values.quadratGrid,
      rows: values.quadratGrid,
      range: [0, (POINT_COUNT / (values.quadratGrid * values.quadratGrid)) * 2.2]
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.nearest,
      ramp: 'diverging',
      range: [-0.04, 0.04],
      radius: 3.4
    }
  ]
});
