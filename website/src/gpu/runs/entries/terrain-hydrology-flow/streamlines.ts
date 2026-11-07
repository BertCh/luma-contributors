// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUStreamlines on a seeded swirl field: Jobard-Lefer seeding with pruning gives evenly spaced
// lines that follow the field. The polylines it wrote are drawn point by point.

import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues,
  GPUStreamlines
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './streamlines.parameters';
import {createSwirlField} from './shared/vector-field';

const SEED_COLUMNS = 40;
const SEED_ROWS = 22;
const STEPS_PER_DIRECTION = 48;
const LINE_CAPACITY = SEED_COLUMNS * SEED_ROWS;
const POINT_CAPACITY = LINE_CAPACITY * (2 * STEPS_PER_DIRECTION + 1);

export default defineOperationRun({
  parameters,
  caption: 'Evenly spaced lines that follow the vector field.',
  createInputs: context => createSwirlField(context),
  build(builder, field, current) {
    const worldWidth = field.columns * field.cellSize;
    const velocities = builder.upload('velocities', 'float32x2', field.velocities);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const wordBuffer = builder.allocate('words', 'uint32', 4);
    const speeds = builder.upload('speeds', 'float32', field.speeds);
    const positions = builder.allocate('positions', 'float32x2', POINT_CAPACITY);
    const gridWidth = SEED_COLUMNS * 2;
    const gridHeight = SEED_ROWS * 2;
    builder.graph.add(
      new GPUStreamlines({
        velocities: velocities.view,
        fieldWidth: field.columns,
        fieldHeight: field.rows,
        gridWidth,
        gridHeight,
        seedColumns: SEED_COLUMNS,
        seedRows: SEED_ROWS,
        stepsPerDirection: STEPS_PER_DIRECTION,
        parameters: parameterBuffer.view,
        uint32Parameters: wordBuffer.view,
        output: {
          lines: {
            ids: builder.allocate('line-ids', 'uint32', LINE_CAPACITY).view,
            status: {
              count: builder.allocate('line-count', 'uint32', 1).view,
              overflow: builder.allocate('overflow', 'uint32', 1).view
            }
          },
          lineOffsets: builder.allocate('line-offsets', 'uint32', LINE_CAPACITY + 1).view,
          positions: positions.view,
          positionCount: builder.allocate('position-count', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {speeds, positions},
      update: values => {
        const settings = {
          fieldExtent: [0, 0, field.cellSize, field.cellSize] as const,
          gridExtent: [0, 0, worldWidth / gridWidth, 1 / gridHeight] as const,
          stepLength: values.stepLength,
          minimumSpeed: values.minimumSpeed,
          seed: Math.round(values.seed),
          minimumPoints: 8
        };
        parameterBuffer.write(getGPUStreamlinesParameterValues(settings));
        wordBuffer.write(getGPUStreamlinesUint32ParameterValues(settings));
        // Rows past the position count are never written; keep them out of the picture.
        positions.write(new Float32Array(POINT_CAPACITY * 2).fill(Number.NaN));
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.speeds,
      columns: inputs.columns,
      rows: inputs.rows,
      smoothing: true
    },
    {layer: 'dots', positions: outputs.positions, radius: 1.7, color: [0.92, 0.97, 1, 0.95]}
  ]
});
