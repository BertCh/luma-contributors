// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTransitionMatrix counts how often a place moves from one class to another between periods.
// Left: the class of each tract. Right: the row-normalized transition probabilities (row = from
// class, column = to class) and, underneath, the steady-state distribution of the chain.

import {
  GPUClassAssignment,
  GPUTransitionMatrix
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createChartAxes, createMatrixTicks, getChartRectangle, mergeText} from '../../text';
import {parameters} from './transition-matrix.parameters';
import {addSlice} from './support/gather';
import {
  createBreaks,
  createDynamics,
  createMatrixMesh,
  createMoveLabels,
  getMatrixCell,
  getMatrixCellBox,
  getTractMove,
  getPeriodOwners,
  PERIOD_COUNT,
  uploadMatrixMesh
} from './support/dynamics';
import {createTractGlyph, findTractAt, uploadTractMesh} from './support/geometry';
import {allocateRegionOutline, drawRegionOutline, writeRegionOutline} from './support/regions';

const MATRIX = (width: number) => [width * 0.56, 0.3, width * 0.97, 0.92] as const;

export default defineOperationRun({
  parameters,
  caption: 'Class transitions between periods as a probability matrix.',
  createInputs: ({random, world}) => ({
    ...createDynamics(random, world),
    world
  }),
  build(builder, {tracts, values, world}, current) {
    const classCount = current.classCount;
    const classes = builder.allocate('classes', 'uint32', values.length);
    builder.graph.add(
      new GPUClassAssignment({
        values: builder.upload('values', 'float32', values).view,
        breaks: builder.upload('breaks', 'float32', createBreaks(values, classCount, true)).view,
        classCount: builder.upload('class-count', 'uint32', Uint32Array.of(classCount)).view,
        output: classes.view
      })
    );
    const probabilities = builder.allocate('probabilities', 'float32', classCount * classCount);
    const steadyStates = builder.allocate('steady-states', 'float32', classCount);
    const stateSparsity = builder.allocate('state-sparsity', 'uint32', 2);
    builder.graph.add(
      new GPUTransitionMatrix({
        classes: classes.view,
        periodCount: PERIOD_COUNT,
        classCount,
        periodLag: current.periodLag,
        minimumTransitions: current.minimumTransitions,
        output: {
          counts: builder.allocate('counts', 'uint32', classCount * classCount).view,
          probabilities: probabilities.view,
          rowTotals: builder.allocate('row-totals', 'uint32', classCount).view,
          steadyStates: steadyStates.view,
          stateSparsity: stateSparsity.view
        }
      })
    );
    const matrix = createMatrixMesh(classCount, classCount, MATRIX(world.width));
    const mesh = uploadTractMesh(builder, tracts);
    return {
      update: values =>
        mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, values.period)),
      outputs: {
        highlight: allocateRegionOutline(builder, {tracts}),
        classes,
        probabilities,
        steadyStates,
        sparsity: addSlice(builder, stateSparsity, 0, 2, 'sparsity-float'),
        matrixMesh: uploadMatrixMesh(builder, matrix, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', matrix.outline),
        matrixOutlineCount: matrix.outlineCount,
        ...mesh
      }
    };
  },
  rebuildOn: ['classCount', 'periodLag', 'minimumTransitions'],
  // The hovered matrix cell (from class, to class) outlines the tracts that made that move.
  interaction: {
    kind: 'probe',
    hint: 'hover a matrix cell',
    rest: ({inputs, values, toUnit}) => {
      const middle = Math.floor(values.classCount / 2);
      const box = getMatrixCellBox(
        MATRIX(inputs.world.width),
        values.classCount,
        values.classCount,
        {
          row: middle,
          column: middle
        }
      );
      return toUnit([(box.minimum[0] + box.maximum[0]) / 2, (box.minimum[1] + box.maximum[1]) / 2]);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const rectangle = MATRIX(inputs.world.width);
      const classes = peek(outputs.classes);
      if (!classes) return {};
      // Over the matrix the cell is the pointer's; over the map it is the move of the nearest tract.
      const tract = findTractAt(inputs.tracts, pointer.x, pointer.y);
      const hovered = getMatrixCell(
        rectangle,
        values.classCount,
        values.classCount,
        pointer.x,
        pointer.y
      );
      const cell =
        hovered ??
        getTractMove(classes, inputs.tracts.count, values.period, values.periodLag, tract);
      const {labels, moveCount} = createMoveLabels(
        classes,
        inputs.tracts.count,
        values.period,
        values.periodLag,
        cell.row,
        cell.column
      );
      writeRegionOutline(inputs, labels, 1, outputs.highlight);
      const probability = peek(outputs.probabilities)?.[cell.row * values.classCount + cell.column];
      return {
        focus: cell.row * values.classCount + cell.column,
        glyphs: [
          ...(hovered ? [] : [createTractGlyph(inputs.tracts, tract)]),
          {shape: 'box', ...getMatrixCellBox(rectangle, values.classCount, values.classCount, cell)}
        ],
        readout: `class ${cell.row + 1} to ${cell.column + 1} · ${moveCount} moves · p ${probability?.toFixed(2) ?? '...'}`
      };
    }
  },
  draw: ({inputs, outputs, values, world}) => {
    const classNames = Array.from({length: values.classCount}, (_, index) => `${index + 1}`);
    const steadyRectangle = [0.56, 0.05, 0.84, 0.22] as const;
    const steadyAxes = createChartAxes({
      rectangle: getChartRectangle(world, steadyRectangle),
      x: {categories: classNames}
    });
    return [
      {
        layer: 'polygons',
        positions: outputs.vertices,
        indices: outputs.indices,
        owners: outputs.owners,
        values: outputs.classes,
        triangleCount: inputs.tracts.triangleCount,
        ramp: 'sequential',
        range: [0, values.classCount - 1]
      },
      drawRegionOutline(inputs, outputs.highlight),
      {
        layer: 'polygons',
        positions: outputs.matrixMesh.vertices,
        indices: outputs.matrixMesh.indices,
        owners: outputs.matrixMesh.owners,
        values: outputs.probabilities,
        triangleCount: outputs.matrixMesh.triangleCount,
        range: [0, 1]
      },
      {
        layer: 'lines',
        context: true,
        segments: outputs.matrixOutline,
        count: outputs.matrixOutlineCount,
        alpha: 0.5
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.steadyStates,
        range: [0, 0.6],
        rectangle: [0.56, 0.05, 0.84, 0.22],
        ramp: 'sequential'
      },
      {
        // Sparse and empty states (fewer than minimumTransitions moves).
        layer: 'chart',
        kind: 'bars',
        values: outputs.sparsity,
        range: [0, values.classCount],
        rectangle: [0.87, 0.05, 0.97, 0.22],
        ramp: 'sequential'
      },
      {
        layer: 'text',
        ...mergeText(
          createMatrixTicks({
            rectangle: [...MATRIX(world.width)],
            rows: classNames,
            columns: classNames,
            rowTitle: 'from class',
            columnTitle: 'to class'
          }),
          steadyAxes,
          {
            labels: [
              {
                text: 'steady state',
                at: [0.56 * world.width, 0.22],
                anchor: 'bottom-left',
                offset: [0, 2],
                role: 'label'
              },
              {
                text: 'sparse',
                at: [0.87 * world.width, 0.22],
                anchor: 'bottom-left',
                offset: [0, 2],
                role: 'label'
              }
            ]
          }
        )
      }
    ];
  }
});
