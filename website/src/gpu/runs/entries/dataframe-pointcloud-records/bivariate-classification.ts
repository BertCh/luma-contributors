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
import {sampleFlashes} from '../../../data/lightning';
import {chooseSticky, createPointIndex} from '../../pick';
import {addIntegerToFloat} from './_lib/convert';
import {createBoxMask, getBoxMaskBounds} from './_lib/helpers';
import {parameters} from './bivariate-classification.parameters';

const ROW_COUNT = 2000;
const MAXIMUM_CLASS_COUNT = 4;
const CELL_COUNT = MAXIMUM_CLASS_COUNT ** 2;

function clampUnit(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

/** Class edges at the quantiles of `sorted`, so every class holds about the same number of rows. */
function getEdges(sorted: Float32Array, classCount: number): Float32Array {
  const edges = new Float32Array(MAXIMUM_CLASS_COUNT + 1).fill(2);
  edges[0] = 0;
  for (let index = 1; index < classCount; index++) {
    edges[index] = sorted[Math.floor((index * sorted.length) / classCount)];
  }
  edges[classCount] = 1.001;
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

/** The key matrix: one square per joint class, laid out row by row. */
/**
 * Each square is centered in its slot; its side grows with the rows counted in that class, so the
 * matrix is both the color key and the count.
 */
function getMatrixPositions(
  originX: number,
  originY: number,
  cellSize: number,
  classCountX: number,
  classCountY: number,
  counts?: ArrayLike<number> | null
): Float32Array {
  const positions = new Float32Array(CELL_COUNT * 4 * 2);
  let largest = 1;
  for (let cell = 0; cell < classCountX * classCountY; cell++)
    largest = Math.max(largest, counts?.[cell] ?? 0);
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    const isUsed = cell < classCountX * classCountY;
    const share = counts ? Math.sqrt((counts[cell] ?? 0) / largest) : 1;
    const size = isUsed ? cellSize * 0.92 * (0.18 + 0.82 * share) : 0;
    const slot = cellSize * 0.46;
    const x = originX + (cell % classCountX) * cellSize + slot - size / 2;
    const y = originY + Math.floor(cell / classCountX) * cellSize + slot - size / 2;
    positions.set([x, y, x + size, y, x + size, y + size, x, y + size], cell * 8);
  }
  return positions;
}

export default defineOperationRun({
  parameters,
  caption:
    'Joint classes of log flash energy and footprint area; the key squares at right grow with their row count.',
  createInputs: ({world}) => {
    // The map fills the left 72 percent; the key and the counts sit on the right.
    const mapWorld = {width: world.width * 0.66, height: world.height};
    const points = sampleFlashes(mapWorld, ROW_COUNT);
    const valuesX = Float32Array.from(points.energy, value => clampUnit(Math.log1p(value) / 8));
    const valuesY = Float32Array.from(points.area, value => clampUnit((Math.log1p(value) - 4) / 4));
    const cellSize = Math.min(world.width * 0.07, 0.13);
    return {
      world,
      mapWorld,
      positions: points.positions,
      index: createPointIndex(points.positions),
      focus: {row: -1},
      valuesX,
      valuesY,
      sortedX: Float32Array.from(valuesX).sort(),
      sortedY: Float32Array.from(valuesY).sort(),
      cellSize,
      matrixX: world.width * 0.7,
      matrixY: 0.5 - 2 * cellSize
    };
  },
  build(builder, inputs) {
    const {cellSize, matrixX, matrixY} = inputs;
    const valuesX = builder.upload('values-x', 'float32', inputs.valuesX);
    const valuesY = builder.upload('values-y', 'float32', inputs.valuesY);
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const keyPositions = builder.upload(
      'key-positions',
      'float32x2',
      getMatrixPositions(matrixX, matrixY, cellSize, MAXIMUM_CLASS_COUNT, MAXIMUM_CLASS_COUNT)
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
    const highlight = builder.allocate('highlight', 'float32x2', ROW_COUNT);
    return {
      outputs: {
        positions,
        jointClass,
        classCounts,
        highlight,
        keyPositions,
        cellIndices,
        cellOwners,
        cellClasses
      },
      update: current => {
        palette.write(new Uint32Array(MAXIMUM_CLASS_COUNT ** 2).fill(0xffffffff));
        breaksX.write(getEdges(inputs.sortedX, current.classCountX));
        breaksY.write(getEdges(inputs.sortedY, current.classCountY));
        mask.write(
          createBoxMask(
            inputs.positions,
            getBoxMaskBounds(inputs.mapWorld.width, inputs.mapWorld.height, current.maskShare)
          )
        );
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
  // The pointer picks a joint class, by hovering a flash on the map or a square of the key. Its
  // flashes are drawn full size, and its square shows its count.
  interaction: {
    kind: 'probe',
    hint: 'hover a class',
    rest: ({inputs, values}) => {
      const column = Math.min(1, values.classCountX - 1);
      const row = Math.min(1, values.classCountY - 1);
      const x = inputs.matrixX + (column + 0.5) * inputs.cellSize;
      const y = inputs.matrixY + (row + 0.5) * inputs.cellSize;
      return [x / inputs.world.width, 1 - y / inputs.world.height];
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const classCountX = values.classCountX;
      const classCountY = values.classCountY;
      const counts = peek(outputs.classCounts);
      const classes = peek(outputs.jointClass);
      outputs.keyPositions.write(
        getMatrixPositions(
          inputs.matrixX,
          inputs.matrixY,
          inputs.cellSize,
          classCountX,
          classCountY,
          counts
        )
      );
      const column = Math.floor((pointer.x - inputs.matrixX) / inputs.cellSize);
      const row = Math.floor((pointer.y - inputs.matrixY) / inputs.cellSize);
      let jointClass = -1;
      if (column >= 0 && column < classCountX && row >= 0 && row < classCountY) {
        jointClass = row * classCountX + column;
      } else if (classes) {
        inputs.focus.row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
        const value = classes[inputs.focus.row];
        if (inputs.focus.row >= 0 && value < CELL_COUNT) jointClass = value;
      }
      if (jointClass < 0 || !classes) return {focus: -1, readout: 'hover a flash or the key'};
      const rows = new Float32Array(ROW_COUNT * 2).fill(-10);
      let found = 0;
      for (let index = 0; index < ROW_COUNT; index++) {
        if (classes[index] !== jointClass) continue;
        rows.set([inputs.positions[index * 2], inputs.positions[index * 2 + 1]], found * 2);
        found++;
      }
      outputs.highlight.write(rows);
      const xClass = jointClass % classCountX;
      const yClass = Math.floor(jointClass / classCountX);
      const x = inputs.matrixX + (xClass + 0.46) * inputs.cellSize;
      const y = inputs.matrixY + (yClass + 0.46) * inputs.cellSize;
      const half = inputs.cellSize * 0.5;
      return {
        focus: [jointClass, found],
        spotlight: null,
        glyphs: [{shape: 'box', minimum: [x - half, y - half], maximum: [x + half, y + half]}],
        readout: `class (x ${xClass + 1}, y ${yClass + 1}) · ${found} flashes`
      };
    }
  },
  draw: ({outputs, values, interaction}) => {
    const categories = getBivariateColors(values.classCountX, values.classCountY);
    const focus = interaction.focus;
    const hasFocus = Array.isArray(focus) && focus[0] >= 0;
    const hovered = Array.isArray(focus) ? focus : [-1, 0];
    const color = categories[hovered[0]] ?? [1, 1, 1];
    return [
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.jointClass,
        ramp: 'categorical',
        categories,
        alpha: hasFocus ? 0.3 : 1,
        radius: 4
      },
      ...(hasFocus
        ? [
            {
              layer: 'dots' as const,
              positions: outputs.highlight,
              count: Math.max(1, hovered[1]),
              color: [color[0], color[1], color[2], 1] as const,
              radius: 5.5
            }
          ]
        : []),
      {
        layer: 'polygons',
        positions: outputs.keyPositions,
        indices: outputs.cellIndices,
        owners: outputs.cellOwners,
        values: outputs.cellClasses,
        triangleCount: CELL_COUNT * 2,
        ramp: 'categorical',
        categories
      }
    ];
  }
});
