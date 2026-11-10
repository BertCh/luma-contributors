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
import {addFloatSlice, addSlice} from './support/gather';

const MATRICES = (width: number) => [0.03 * width, 0.3, 0.52 * width, 0.7] as const;

// The thresholds the pointer can settle on; the default is the middle one.
const THRESHOLDS = [1, 30, 60, 90, 120] as const;
import {
  createBreaks,
  createDynamics,
  createMatrixGrid,
  getMatrixGridRectangles,
  PERIOD_COUNT,
  uploadMatrixMesh
} from './support/dynamics';

export default defineOperationRun({
  parameters,
  caption:
    'Transition counts (left; rows below minimumTransitions struck through) and the steady states they imply (bars: pooled, west, east).',
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
      tracts.centers[(index % tracts.count) * 2] < world.width / 2 ? 0 : 1
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
    const matrix = createMatrixGrid(2, classCount, MATRICES(world.width));
    return {
      outputs: {
        rawCounts: counts,
        counts: addSlice(builder, counts, 0, counts.length, 'counts-float'),
        steadyStates,
        pooled: addFloatSlice(builder, steadyStates, classCount, 'steady-pooled', 0),
        west: addFloatSlice(builder, steadyStates, classCount, 'steady-west', classCount),
        east: addFloatSlice(builder, steadyStates, classCount, 'steady-east', 2 * classCount),
        sparsity: addSlice(builder, stateSparsity, 0, 2, 'sparsity-float'),
        stateCount: 2 * classCount,
        matrixMesh: uploadMatrixMesh(builder, matrix, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', matrix.outline),
        matrixOutlineCount: matrix.outlineCount
      }
    };
  },
  // Rows with fewer moves than minimumTransitions are struck through in the count matrices.
  async afterRun({outputs, inputs, values, upload}) {
    const bytes = await outputs.rawCounts.buffer.readAsync();
    const counts = new Uint32Array(bytes.buffer, bytes.byteOffset, 2 * values.classCount ** 2);
    const struck: number[] = [];
    getMatrixGridRectangles(2, MATRICES(inputs.world.width)).forEach((rectangle, matrix) => {
      const rowHeight = (rectangle[3] - rectangle[1]) / values.classCount;
      for (let row = 0; row < values.classCount; row++) {
        let total = 0;
        for (let column = 0; column < values.classCount; column++) {
          total += counts[(matrix * values.classCount + row) * values.classCount + column];
        }
        if (total >= values.minimumTransitions) continue;
        const y = rectangle[3] - (row + 0.5) * rowHeight;
        struck.push(rectangle[0], y, rectangle[2], y);
      }
    });
    return {
      struck: upload(
        'struck-rows',
        'float32x4',
        struck.length ? Float32Array.from(struck) : Float32Array.of(Number.NaN, 0, 0, 0)
      ),
      struckCount: Math.max(1, struck.length / 4)
    };
  },
  buildCache: 5,
  rebuildOn: ['classCount', 'minimumTransitions', 'periodLag'],
  // x raises minimumTransitions: rows with fewer moves than that are sparse. The operation is
  // rebuilt once the pointer rests on one of five thresholds.
  interaction: {
    kind: 'level',
    hint: 'raise the threshold',
    rest: [0.5, 0.5],
    resolve(pointer, {outputs, peek, world}) {
      const index = Math.min(THRESHOLDS.length - 1, Math.floor(pointer.unitX * THRESHOLDS.length));
      const sparsity = peek(outputs.sparsity);
      return {
        settle: {minimumTransitions: THRESHOLDS[index]},
        glyphs: THRESHOLDS.map((_, stop) => ({
          shape: stop === index ? ('crosshair' as const) : ('dot' as const),
          at: [(0.03 + ((stop + 0.5) / THRESHOLDS.length) * 0.49) * world.width, 0.04] as const
        })),
        readout: `minimumTransitions ${THRESHOLDS[index]}${sparsity ? ` · ${sparsity[0]} sparse rows` : ''}`
      };
    }
  },
  draw: ({outputs, extra}) => [
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
      layer: 'lines',
      segments: extra.struck,
      count: extra.struckCount,
      radius: 7,
      color: [1, 0.9, 0.35, 0.55]
    },
    ...(
      [
        [outputs.pooled, 0.67, 0.9, 'pooled'],
        [outputs.west, 0.38, 0.61, 'west'],
        [outputs.east, 0.09, 0.32, 'east']
      ] as const
    ).map(([values, bottom, top]) => ({
      layer: 'chart' as const,
      kind: 'bars' as const,
      values,
      range: [0, 0.5] as const,
      rectangle: [0.58, bottom, 0.97, top] as const,
      ramp: 'sequential' as const
    }))
  ]
});
