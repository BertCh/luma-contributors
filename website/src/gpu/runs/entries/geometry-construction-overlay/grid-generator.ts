// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGridGenerator writes the cell rings and centers of a square, hexagon or triangle grid from
// four numbers (origin and cell size). Each cell ring has its own color; dots are the centers.

import {
  getGPUGridCellCount,
  getGPUGridGeneratorParameterValues,
  getGPUGridVerticesPerCell,
  GPUGridGenerator
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getRainStations} from '../../../data/point-observations';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments} from './shared/lines';
import {parameters} from './grid-generator.parameters';

const GRID_TYPES = ['square', 'hex', 'triangle'] as const;
const COLUMNS = 20;
const ROWS = 12;

export default defineOperationRun({
  parameters,
  caption:
    'Sampling grid over Carolina rain gauges. Cell spacing changes while station locations stay fixed.',
  createInputs: ({random, world}) => {
    const {positions, rain} = getRainStations(random, world, 180, true);
    const east = positions.filter((_, index) => index % 2 === 0);
    const north = positions.filter((_, index) => index % 2 === 1);
    return {
      positions,
      rain,
      minX: Math.min(...east),
      minY: Math.min(...north),
      shift: [0, 0] as [number, number]
    };
  },
  build(builder, inputs, values) {
    const gridType = GRID_TYPES[values.gridType];
    const cellCount = getGPUGridCellCount(gridType, COLUMNS, ROWS);
    const verticesPerCell = getGPUGridVerticesPerCell(gridType);
    const positions = builder.allocate('cell-positions', 'float32x2', cellCount * verticesPerCell);
    const centers = builder.allocate('cell-centers', 'float32x2', cellCount);
    const parameterBuffer = builder.allocate('grid-parameters', 'float32', 4);
    builder.graph.add(
      new GPUGridGenerator({
        gridType,
        columns: COLUMNS,
        rows: ROWS,
        hexOrientation:
          gridType === 'hex' ? (values.hexOrientation === 0 ? 'pointy' : 'flat') : undefined,
        parameters: parameterBuffer.view,
        output: {positions: positions.view, centers: centers.view}
      })
    );
    const ringOffsets = builder.upload(
      'ring-offsets',
      'uint32',
      Uint32Array.from({length: cellCount + 1}, (_, cell) => cell * verticesPerCell)
    );
    const drawing = addPackedSegments(builder, 'cells', positions, ringOffsets, {closed: true});
    return {
      outputs: {
        centers,
        ...drawing,
        stations: builder.upload('stations', 'float32x2', inputs.positions),
        rainfall: builder.upload('rainfall-mm', 'float32', inputs.rain)
      },
      update: current => {
        const isHex = gridType === 'hex';
        parameterBuffer.write(
          getGPUGridGeneratorParameterValues({
            minX: inputs.minX + inputs.shift[0],
            minY: inputs.minY + inputs.shift[1],
            cellWidth: current.cellWidth,
            cellHeight: isHex ? current.cellWidth * 0.8660254 : current.cellWidth
          })
        );
      }
    };
  },
  rebuildOn: ['gridType', 'hexOrientation'],
  // The pointer is the grid origin (minX, minY), wrapped into one cell: sliding it slides the
  // lattice under the stations, and the first cell is outlined at the origin.
  interaction: {
    kind: 'drag',
    hint: 'slide the grid origin',
    rest: ({inputs, toUnit}) => toUnit([inputs.minX, inputs.minY]),
    resolve(pointer, {inputs, values}) {
      const cellWidth = values.cellWidth;
      const cellHeight = values.gridType === 1 ? cellWidth * 0.8660254 : cellWidth;
      const wrap = (value: number, size: number) => ((value % size) + size) % size;
      inputs.shift = [
        wrap(pointer.x - inputs.minX, cellWidth),
        wrap(pointer.y - inputs.minY, cellHeight)
      ];
      const origin: [number, number] = [
        inputs.minX + inputs.shift[0],
        inputs.minY + inputs.shift[1]
      ];
      return {
        rerun: true,
        glyphs: [
          {shape: 'box', minimum: origin, maximum: [origin[0] + cellWidth, origin[1] + cellHeight]},
          {shape: 'crosshair', at: origin}
        ],
        readout: `origin ${origin[0].toFixed(2)}, ${origin[1].toFixed(2)} · cellWidth ${cellWidth.toFixed(3)}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      color: [0.4, 0.53, 0.65, 0.65],
      radius: 1.4
    },
    {layer: 'dots', positions: outputs.centers, color: [0.5, 0.6, 0.7, 0.65], radius: 2},
    {
      layer: 'dots',
      positions: outputs.stations,
      values: outputs.rainfall,
      context: true,
      ramp: 'sequential',
      radius: 5.5
    }
  ]
});
