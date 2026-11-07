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
import {parameters} from './transition-matrix.parameters';
import {addSlice} from './support/gather';
import {
  createBreaks,
  createDynamics,
  createMatrixMesh,
  getPeriodOwners,
  PERIOD_COUNT,
  uploadMatrixMesh
} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

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
    const matrix = createMatrixMesh(classCount, classCount, [
      world.width * 0.56,
      0.3,
      world.width * 0.97,
      0.92
    ]);
    const mesh = uploadTractMesh(builder, tracts);
    mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, current.period));
    return {
      outputs: {
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
  draw: ({inputs, outputs, values}) => [
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
    }
  ]
});
