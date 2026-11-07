// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialMarkov conditions the class transitions on the class of each tract's spatial lag.
// Left: the lag class of each tract. Right: one transition matrix per lag class (low to high).

import {getGPUClassBreaksParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPUClassAssignment,
  GPUSpatialMarkov
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './spatial-markov.parameters';
import {
  createBreaks,
  createDynamics,
  createMatrixGrid,
  getPeriodOwners,
  PERIOD_COUNT,
  rowStandardize,
  uploadMatrixMesh
} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Transition matrices for low to high lag classes; the map shows each tract lag class.',
  createInputs: ({random, world}) => ({...createDynamics(random, world), world}),
  build(builder, {tracts, values, world}, current) {
    const classCount = current.classCount;
    const lagClassCount = current.maximumLagClassCount;
    const input = builder.upload('values', 'float32', values);
    const classes = builder.allocate('classes', 'uint32', values.length);
    builder.graph.add(
      new GPUClassAssignment({
        values: input.view,
        breaks: builder.upload('breaks', 'float32', createBreaks(values, classCount, true)).view,
        classCount: builder.upload('class-count', 'uint32', Uint32Array.of(classCount)).view,
        output: classes.view
      })
    );
    const probabilities = builder.allocate(
      'probabilities',
      'float32',
      lagClassCount * classCount * classCount
    );
    const lagClasses = builder.allocate('lag-classes', 'uint32', values.length);
    builder.graph.add(
      new GPUSpatialMarkov({
        values: input.view,
        classes: classes.view,
        classCount,
        weights: uploadSpatialWeights(builder, rowStandardize(tracts.contiguity)),
        periodCount: PERIOD_COUNT,
        normalize: current.normalize === 1,
        maximumLagClassCount: lagClassCount,
        lagParameters: builder.upload(
          'lag-parameters',
          'float32',
          getGPUClassBreaksParameterValues(
            {method: 'quantile', classCount: lagClassCount},
            lagClassCount
          )
        ).view,
        output: {
          counts: builder.allocate('counts', 'uint32', lagClassCount * classCount * classCount)
            .view,
          probabilities: probabilities.view,
          lagBreaks: builder.allocate('lag-breaks', 'float32', lagClassCount + 1).view,
          lagClassCount: builder.allocate('lag-class-count', 'uint32', 1).view,
          lagClasses: lagClasses.view
        }
      })
    );
    const grid = createMatrixGrid(lagClassCount, classCount, [
      world.width * 0.56,
      0.08,
      world.width * 0.97,
      0.92
    ]);
    const mesh = uploadTractMesh(builder, tracts);
    mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, current.period));
    return {
      outputs: {
        lagClasses,
        probabilities,
        matrixMesh: uploadMatrixMesh(builder, grid, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', grid.outline),
        matrixOutlineCount: grid.outlineCount,
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
      values: outputs.lagClasses,
      triangleCount: inputs.tracts.triangleCount,
      ramp: 'sequential',
      range: [0, values.maximumLagClassCount - 1]
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
      alpha: 0.4
    }
  ]
});
