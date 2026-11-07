// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTransitionDiagnostics reads transition counts (left: west and east halves, the input) and finds the chain's
// steady state (right, bars) and its homogeneity test statistics. Rows with fewer than
// `minimumTransitions` moves are treated as sparse.

import {
  GPUClassAssignment,
  GPUTransitionDiagnostics,
  GPUTransitionMatrix
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './transition-diagnostics.parameters';
import {addSlice} from './support/gather';
import {
  createBreaks,
  createDynamics,
  createMatrixGrid,
  PERIOD_COUNT,
  uploadMatrixMesh
} from './support/dynamics';

export default defineOperationRun({
  parameters,
  caption:
    'Transition counts (left) and the steady states they imply (bars: pooled and per condition).',
  createInputs: ({random, world}) => ({...createDynamics(random, world, 14, 1), world}),
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
    // Two conditions (west and east half) so the homogeneity test can compare their chains.
    const conditions = Uint32Array.from({length: values.length}, (_, index) =>
      tracts.centers[(index % tracts.count) * 2] < world.width * 0.25 ? 0 : 1
    );
    const counts = builder.allocate('counts', 'uint32', 2 * classCount * classCount);
    builder.graph.add(
      new GPUTransitionMatrix({
        classes: classes.view,
        periodCount: PERIOD_COUNT,
        classCount,
        periodLag: current.periodLag,
        conditionClasses: builder.upload('conditions', 'uint32', conditions).view,
        conditionCount: 2,
        output: {counts: counts.view}
      })
    );
    const steadyStates = builder.allocate('steady-states', 'float32', 3 * classCount);
    const stateSparsity = builder.allocate('state-sparsity', 'uint32', 2);
    builder.graph.add(
      new GPUTransitionDiagnostics({
        counts: counts.view,
        classCount,
        conditionCount: 2,
        minimumTransitions: current.minimumTransitions,
        output: {
          homogeneity: builder.allocate('homogeneity', 'float32', 5).view,
          steadyStates: steadyStates.view,
          stateSparsity: stateSparsity.view,
          status: {converged: builder.allocate('converged', 'uint32', 1).view}
        }
      })
    );
    const matrix = createMatrixGrid(2, classCount, [
      0.03 * world.width,
      0.3,
      world.width * 0.52,
      0.7
    ]);
    return {
      outputs: {
        counts: addSlice(builder, counts, 0, counts.length, 'counts-float'),
        steadyStates,
        sparsity: addSlice(builder, stateSparsity, 0, 2, 'sparsity-float'),
        stateCount: 2 * classCount,
        matrixMesh: uploadMatrixMesh(builder, matrix, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', matrix.outline),
        matrixOutlineCount: matrix.outlineCount
      }
    };
  },
  draw: ({outputs, world}) => [
    {
      layer: 'polygons',
      positions: outputs.matrixMesh.vertices,
      indices: outputs.matrixMesh.indices,
      owners: outputs.matrixMesh.owners,
      values: outputs.counts,
      triangleCount: outputs.matrixMesh.triangleCount,
      ramp: 'sequential',
      range: [0, 80]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.matrixOutline,
      count: outputs.matrixOutlineCount,
      alpha: 0.4
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.steadyStates,
      range: [0, 0.5],
      rectangle: [0.58, 0.1, 0.97, 0.9],
      ramp: 'sequential'
    },
    {
      // Sparse and empty (condition, from) states: bars grow with minimumTransitions.
      layer: 'chart',
      kind: 'bars',
      values: outputs.sparsity,
      range: [0, outputs.stateCount],
      rectangle: [0.03, 0.08, 0.52, 0.24],
      ramp: 'sequential'
    }
  ]
});
