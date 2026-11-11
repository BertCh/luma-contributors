// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeKFunction of GOES lightning flashes. The panel on the right is Diggle's D(u, v) =
// K(u, v) - K_s(u) K_t(v): rows are distance classes (near at the top), columns time-lag classes
// (short at the left). D above zero (red) means flashes close in space are closer in time than
// space and time alone predict: space-time clustering. The pointer is a pad on the two extents of
// the grid; on the map, the ring of radius u is drawn at the busiest flash and the flashes within
// lag v of it are bright.

import {
  getGPUSpaceTimeKFunctionParameterValues,
  GPUSpaceTimeKFunction,
  GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createLightningFlashes} from './_lib/events';
import {createCellTable, getCellBox, getTableCell, uploadCellTable} from './_lib/table';
import {OBSERVATION_HOURS} from './_lib/time-window';
import {parameters} from './space-time-k-function.parameters';

const FLASH_COUNT = 1500;
const DISTANCE_CLASSES = 12;
const TIME_CLASSES = 10;
/** Radius, in km, used to find the busiest flash (the most neighbors). */
const BUSY_RADIUS = 60;

function createInputs({world}: {world: {width: number; height: number}}) {
  const flashes = createLightningFlashes([0, 0, world.width * 0.6, world.height], FLASH_COUNT);
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (let row = 0; row < flashes.count; row++) {
    const x = flashes.positions[row * 2];
    const y = flashes.positions[row * 2 + 1];
    bounds[0] = Math.min(bounds[0], x);
    bounds[1] = Math.min(bounds[1], y);
    bounds[2] = Math.max(bounds[2], x);
    bounds[3] = Math.max(bounds[3], y);
  }
  // The busiest flash: the most other flashes within BUSY_RADIUS km.
  const radius = BUSY_RADIUS / flashes.kilometersPerUnit;
  let busiest = 0;
  let busiestCount = -1;
  for (let row = 0; row < flashes.count; row += 3) {
    let count = 0;
    for (let other = 0; other < flashes.count; other++) {
      const dx = flashes.positions[row * 2] - flashes.positions[other * 2];
      const dy = flashes.positions[row * 2 + 1] - flashes.positions[other * 2 + 1];
      if (dx * dx + dy * dy <= radius * radius) count++;
    }
    if (count > busiestCount) {
      busiestCount = count;
      busiest = row;
    }
  }
  return {
    ...flashes,
    world,
    busiest,
    bounds: bounds as [number, number, number, number],
    table: createCellTable(TIME_CLASSES, DISTANCE_CLASSES, [
      world.width * 0.64,
      0.12,
      world.width * 0.96,
      0.88
    ])
  };
}

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  d: RunBuffer;
  positions: RunBuffer;
  nearInTime: RunBuffer;
  tablePositions: RunBuffer;
  tableIndices: RunBuffer;
  tableOwners: RunBuffer;
};

/** Three significant digits: D spans several decades (units km² h). */
function formatSignificant(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) ? '…' : value.toPrecision(3);
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'GOES lightning, May 21–22 2024 (1,500 of 4,096 sampled flashes): Diggle D(u, v) per distance class (rows, near at top) and time lag (columns); red is space-time clustering.',
  createInputs,
  build(builder, flashes) {
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH
    );
    const classCount = DISTANCE_CLASSES * TIME_CLASSES;
    const d = builder.allocate('d', 'float32', classCount);
    const positions = builder.upload('positions', 'float32x2', flashes.positions);
    builder.graph.add(
      new GPUSpaceTimeKFunction({
        positions: positions.view,
        times: builder.upload('times', 'float32', flashes.hours).view,
        parameters: parameterBuffer.view,
        gridSize: [32, 20],
        distanceClassCount: DISTANCE_CLASSES,
        timeClassCount: TIME_CLASSES,
        output: {k: builder.allocate('k', 'float32', classCount).view, d: d.view}
      })
    );
    const nearInTime = builder.allocate('near-in-time', 'float32', flashes.count);
    const table = uploadCellTable(builder, flashes.table);
    return {
      outputs: {
        d,
        positions,
        nearInTime,
        tablePositions: table.positions,
        tableIndices: table.indices,
        tableOwners: table.owners
      },
      update: current => {
        const time = flashes.hours[flashes.busiest];
        nearInTime.write(
          Float32Array.from(flashes.hours, hour =>
            Math.abs(hour - time) <= current.maximumTimeLag ? 1 : 0.15
          )
        );
        parameterBuffer.write(
          getGPUSpaceTimeKFunctionParameterValues({
            bounds: flashes.bounds,
            maximumDistance: current.maximumDistance / flashes.kilometersPerUnit,
            maximumTimeLag: current.maximumTimeLag,
            timeWindow: [0, OBSERVATION_HOURS]
          })
        );
      }
    };
  },
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'lag across, distance up',
    x: 'maximumTimeLag',
    y: 'maximumDistance',
    decorate: (values, pointer, {inputs, outputs, peek}) => {
      const cell = getTableCell(inputs.table, pointer.x, pointer.y);
      const at = [
        inputs.positions[inputs.busiest * 2],
        inputs.positions[inputs.busiest * 2 + 1]
      ] as const;
      const distance = (values.maximumDistance * (cell.row + 1)) / DISTANCE_CLASSES;
      const lag = (values.maximumTimeLag * (cell.column + 1)) / TIME_CLASSES;
      return {
        glyphs: [
          {shape: 'box', ...getCellBox(inputs.table, cell.row, cell.column)},
          {shape: 'ring', at, radius: values.maximumDistance / inputs.kilometersPerUnit},
          {shape: 'crosshair', at, size: 10}
        ],
        readout: `u ${values.maximumDistance} km · v ${values.maximumTimeLag} h · D(${distance.toFixed(
          0
        )} km, ${lag.toFixed(2)} h) ${formatSignificant(peek(outputs.d)?.[cell.index])}`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.nearInTime,
      range: [0, 1],
      ramp: 'grey',
      radius: 2
    },
    {
      layer: 'polygons',
      positions: outputs.tablePositions,
      indices: outputs.tableIndices,
      owners: outputs.tableOwners,
      values: outputs.d,
      triangleCount: inputs.table.triangleCount,
      ramp: 'diverging',
      range: 'auto'
    }
  ]
});
