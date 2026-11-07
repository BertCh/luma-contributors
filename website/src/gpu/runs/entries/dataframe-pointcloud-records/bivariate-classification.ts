// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBivariateClassification of two record columns on equal-interval edges. Each record takes the
// joint class (yClass * classCountX + xClass). The map colors each record with a two-variable
// palette (x in blue, y in orange); the colored matrix is its key and the matrix below it counts
// the rows in each joint class.

import {
  GPUBivariateClassification,
  getGPUBivariateClassificationParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {addIntegerToFloat} from './_lib/convert';
import {parameters} from './bivariate-classification.parameters';

const ROW_COUNT = 2000;
const MAXIMUM_CLASS_COUNT = 4;
const CELL_COUNT = MAXIMUM_CLASS_COUNT ** 2;

function clampUnit(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function getEdges(classCount: number): Float32Array {
  const edges = new Float32Array(MAXIMUM_CLASS_COUNT + 1).fill(2);
  for (let index = 0; index <= classCount; index++)
    edges[index] = index / classCount + (index === classCount ? 0.001 : 0);
  return edges;
}

/** Bivariate palette indexed by joint class (yClass * classCountX + xClass): blue by x, orange by y. */
function getBivariateColors(classCountX: number, classCountY: number) {
  const colors: (readonly [number, number, number])[] = [];
  for (let yClass = 0; yClass < classCountY; yClass++) {
    for (let xClass = 0; xClass < classCountX; xClass++) {
      const x = xClass / Math.max(classCountX - 1, 1);
      const y = yClass / Math.max(classCountY - 1, 1);
      const blue = [0.9 - 0.8 * x, 0.92 - 0.55 * x, 0.97 - 0.15 * x];
      const orange = [0.98 - 0.05 * y, 0.9 - 0.55 * y, 0.85 - 0.8 * y];
      colors.push([blue[0] * orange[0], blue[1] * orange[1], blue[2] * orange[2]]);
    }
  }
  return colors;
}

/** One square per joint class, laid out row by row; unused slots collapse to a point. */
function getMatrixCenters(
  originX: number,
  originY: number,
  cellSize: number,
  classCountX: number,
  classCountY: number
): Float32Array {
  const centers = new Float32Array(CELL_COUNT * 2).fill(-10);
  for (let cell = 0; cell < classCountX * classCountY; cell++) {
    centers[cell * 2] = originX + ((cell % classCountX) + 0.46) * cellSize;
    centers[cell * 2 + 1] = originY + (Math.floor(cell / classCountX) + 0.46) * cellSize;
  }
  return centers;
}

function getMatrixPositions(
  originX: number,
  originY: number,
  cellSize: number,
  classCountX: number,
  classCountY: number
): Float32Array {
  const positions = new Float32Array(CELL_COUNT * 4 * 2);
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    const isUsed = cell < classCountX * classCountY;
    const size = isUsed ? cellSize * 0.92 : 0;
    const x = originX + (cell % classCountX) * cellSize;
    const y = originY + Math.floor(cell / classCountX) * cellSize;
    positions.set([x, y, x + size, y, x + size, y + size, x, y + size], cell * 8);
  }
  return positions;
}

export default defineOperationRun({
  parameters,
  caption: 'Joint class of two variables per record; the grid counts rows per class.',
  createInputs: ({random, world}) => {
    // The map fills the left 72 percent; the key and the counts sit on the right.
    const mapWorld = {width: world.width * 0.72, height: world.height};
    const points = createUniformPoints(random, mapWorld, ROW_COUNT, 0.25);
    const valuesX = new Float32Array(ROW_COUNT);
    const valuesY = new Float32Array(ROW_COUNT);
    for (let row = 0; row < ROW_COUNT; row++) {
      // x grows to the right, y grows upward, each with smooth and random variation.
      valuesX[row] = clampUnit(
        0.6 * (points.positions[row * 2] / mapWorld.width) + 0.4 * points.values[row]
      );
      valuesY[row] = clampUnit(0.6 * points.positions[row * 2 + 1] + 0.4 * random());
    }
    const cellSize = Math.min(world.width * 0.05, 0.1);
    return {world, positions: points.positions, valuesX, valuesY, cellSize};
  },
  build(builder, inputs) {
    const {world, cellSize} = inputs;
    const valuesX = builder.upload('values-x', 'float32', inputs.valuesX);
    const valuesY = builder.upload('values-y', 'float32', inputs.valuesY);
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const matrixX = world.width * 0.76;
    const keyPositions = builder.upload(
      'key-positions',
      'float32x2',
      getMatrixPositions(matrixX, 0.58, cellSize, MAXIMUM_CLASS_COUNT, MAXIMUM_CLASS_COUNT)
    );
    const countPositions = builder.upload(
      'count-positions',
      'float32x2',
      getMatrixCenters(matrixX, 0.12, cellSize, MAXIMUM_CLASS_COUNT, MAXIMUM_CLASS_COUNT)
    );
    const cellIndices = builder.upload(
      'cell-indices',
      'uint32',
      Uint32Array.from({length: CELL_COUNT * 6}, (_, index) => {
        const corner = [0, 1, 2, 0, 2, 3][index % 6];
        return Math.floor(index / 6) * 4 + corner;
      })
    );
    const cellOwners = builder.upload(
      'cell-owners',
      'uint32',
      Uint32Array.from({length: CELL_COUNT * 2}, (_, index) => Math.floor(index / 2))
    );
    const cellClasses = builder.upload(
      'cell-classes',
      'float32',
      Float32Array.from({length: CELL_COUNT}, (_, index) => index)
    );
    const breaksX = builder.allocate('breaks-x', 'float32', MAXIMUM_CLASS_COUNT + 1);
    const breaksY = builder.allocate('breaks-y', 'float32', MAXIMUM_CLASS_COUNT + 1);
    const palette = builder.allocate('palette', 'uint32', MAXIMUM_CLASS_COUNT ** 2);
    const mask = builder.allocate('mask', 'uint32', ROW_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const classIds = builder.allocate('class-ids', 'uint32', ROW_COUNT);
    const classCounts = builder.allocate('class-counts', 'uint32', MAXIMUM_CLASS_COUNT ** 2);
    builder.graph.add(
      new GPUBivariateClassification({
        valuesX: valuesX.view,
        valuesY: valuesY.view,
        mask: mask.view,
        breaksX: breaksX.view,
        breaksY: breaksY.view,
        palette: palette.view,
        parameters: parameterBuffer.view,
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        output: {classIds: classIds.view, classCounts: classCounts.view}
      })
    );
    const jointClass = addIntegerToFloat(builder, 'joint-class', classIds);
    const gridCounts = addIntegerToFloat(builder, 'grid-counts', classCounts);
    return {
      outputs: {
        positions,
        jointClass,
        gridCounts,
        keyPositions,
        countPositions,
        cellIndices,
        cellOwners,
        cellClasses
      },
      update: current => {
        const shape = [matrixX, cellSize, current.classCountX, current.classCountY] as const;
        keyPositions.write(getMatrixPositions(shape[0], 0.58, shape[1], shape[2], shape[3]));
        countPositions.write(getMatrixCenters(shape[0], 0.12, shape[1], shape[2], shape[3]));
        palette.write(new Uint32Array(MAXIMUM_CLASS_COUNT ** 2).fill(0xffffffff));
        breaksX.write(getEdges(current.classCountX));
        breaksY.write(getEdges(current.classCountY));
        const kept = Math.floor((ROW_COUNT * current.maskShare) / 100);
        mask.write(Uint32Array.from({length: ROW_COUNT}, (_, row) => (row < kept ? 1 : 0)));
        parameterBuffer.write(
          getGPUBivariateClassificationParameterValues({
            classCountX: current.classCountX,
            classCountY: current.classCountY
          })
        );
      }
    };
  },
  rebuildOn: [],
  draw: ({outputs, values}) => {
    const categories = getBivariateColors(values.classCountX, values.classCountY);
    return [
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.jointClass,
        ramp: 'categorical',
        categories,
        radius: 4
      },
      {
        layer: 'polygons',
        positions: outputs.keyPositions,
        indices: outputs.cellIndices,
        owners: outputs.cellOwners,
        values: outputs.cellClasses,
        triangleCount: CELL_COUNT * 2,
        ramp: 'categorical',
        categories
      },
      {
        layer: 'dots',
        positions: outputs.countPositions,
        values: outputs.gridCounts,
        ramp: 'sequential',
        range: [0, ROW_COUNT / 6],
        radius: 17
      }
    ];
  }
});
