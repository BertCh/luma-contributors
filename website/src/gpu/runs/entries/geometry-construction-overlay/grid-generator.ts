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
import {defineOperationRun} from '../../define-run';
import {addPackedSegments, OUTPUT_COLOR} from './shared/lines';
import {parameters} from './grid-generator.parameters';

const GRID_TYPES = ['square', 'hex', 'triangle'] as const;
const COLUMNS = 20;
const ROWS = 12;

export default defineOperationRun({
  parameters,
  caption: 'Grid cells generated from an origin and a cell size; dots are the cell centers.',
  createInputs: () => ({}),
  build(builder, _inputs, values) {
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
      outputs: {centers, ...drawing},
      update: current => {
        const isHex = gridType === 'hex';
        parameterBuffer.write(
          getGPUGridGeneratorParameterValues({
            minX: 0.04,
            minY: 0.04,
            cellWidth: current.cellWidth,
            cellHeight: isHex ? current.cellWidth * 0.8660254 : current.cellWidth
          })
        );
      }
    };
  },
  rebuildOn: ['gridType', 'hexOrientation'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 1.6
    },
    {layer: 'dots', positions: outputs.centers, color: OUTPUT_COLOR, radius: 3}
  ]
});
