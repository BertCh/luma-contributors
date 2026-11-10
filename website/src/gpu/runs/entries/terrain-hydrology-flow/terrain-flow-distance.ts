// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainFlowDistance on the D8 directions of GPUTerrainFlow: how far each cell's flow path runs
// downstream, either to the first stream cell or to the outlet.

import {GPUTerrainFlowDistance} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-flow-distance.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  getCellPolyline,
  getTerrainCell,
  getTerrainCellCenter,
  getTerrainContextLayer,
  traceDownstream
} from './shared/terrain';

/** Index 0 is not a stream; every other value is drawn in the same blue. */
const STREAM_COLORS = Array.from({length: 16}, (_, index) =>
  index === 0 ? ([0, 0, 0, 0] as const) : ([0.35, 0.7, 1, 1] as const)
);

export default defineOperationRun({
  parameters,
  caption: 'Downstream path length from every cell to the channel (or the outlet).',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain, values) {
    const flow = addTerrainFlow(builder, terrain);
    const distance = builder.allocate('distance', 'float32', terrain.cellCount);
    const validity = builder.allocate('validity', 'uint32', terrain.cellCount);
    const terminalCells = builder.allocate('terminal-cells', 'uint32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainFlowDistance({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        streams: values.streams === 1 ? flow.streams.view : undefined,
        maximumSteps: terrain.cellCount,
        output: {
          distance: distance.view,
          validity: validity.view,
          terminalCells: terminalCells.view
        }
      })
    );
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        distance,
        flowDirections: flow.flowDirections,
        streams: flow.streams
      },
      update: current => flow.writeStreamThreshold(current.streamThreshold)
    };
  },
  rebuildOn: ['streams'],
  // The pointer reads one cell: the D8 path its water follows is drawn down to the first stream
  // cell (or the outlet), and the distance along it is the value of the cell.
  interaction: {
    kind: 'probe',
    hint: 'hover a slope',
    rest: [0.3, 0.42],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const directions = peek(outputs.flowDirections);
      const streams = peek(outputs.streams);
      const distances = peek(outputs.distance);
      if (!directions || !streams || !distances) return {};
      const cell = getTerrainCell(inputs, world, pointer.x, pointer.y);
      const path = traceDownstream(
        inputs,
        directions,
        cell.index,
        values.streams === 1 ? streams : null
      );
      const end = getTerrainCellCenter(inputs, world, path[path.length - 1]);
      const distance = distances[cell.index];
      return {
        glyphs: [
          {shape: 'polyline', points: getCellPolyline(inputs, world, path)},
          {shape: 'dot', at: end}
        ],
        readout: `flow distance ${Number.isFinite(distance) ? distance.toFixed(2) : 'none'} · ${path.length - 1} cells to ${values.streams === 1 ? 'stream' : 'outlet'}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.distance,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 0.25],
      alpha: 0.9,
      smoothing: true
    },
    ...(values.streams === 1
      ? [
          {
            layer: 'raster',
            values: outputs.streams,
            columns: inputs.columns,
            rows: inputs.rows,
            categories: STREAM_COLORS,
            alpha: 0.9
          } as const
        ]
      : [])
  ]
});
