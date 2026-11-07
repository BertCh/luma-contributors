// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULISAMarkov classifies each tract in each period by its local Moran quadrant (0 = not
// significant, then HH, LH, LL, HL) and counts the moves between quadrants. Left: the quadrant
// map of one period. Right: the transition probabilities.

import {
  getGPUSpatialAutocorrelationParameterValues,
  GPULISAMarkov,
  GPU_LISA_MARKOV_STATE_COUNT
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './lisa-markov.parameters';
import {
  createDynamics,
  createMatrixMesh,
  getPeriodOwners,
  PERIOD_COUNT,
  rowStandardize,
  uploadMatrixMesh
} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Local Moran quadrants per tract (map) and how places move between them (matrix).',
  createInputs: ({random, world}) => ({...createDynamics(random, world, 12), world}),
  build(builder, {tracts, values, world}, current) {
    const stateCount = GPU_LISA_MARKOV_STATE_COUNT;
    const quadrants = builder.allocate('quadrants', 'uint32', values.length);
    const probabilities = builder.allocate('probabilities', 'float32', stateCount * stateCount);
    builder.graph.add(
      new GPULISAMarkov({
        values: builder.upload('values', 'float32', values).view,
        weights: uploadSpatialWeights(builder, rowStandardize(tracts.contiguity)),
        periodCount: PERIOD_COUNT,
        periodLag: current.periodLag,
        parameters: builder.upload(
          'parameters',
          'float32',
          getGPUSpatialAutocorrelationParameterValues({
            significanceLevel: current.significanceLevel
          })
        ).view,
        gating: current.gating === 0 ? 'significance' : 'none',
        output: {
          counts: builder.allocate('counts', 'uint32', stateCount * stateCount).view,
          probabilities: probabilities.view,
          quadrants: quadrants.view
        }
      })
    );
    const matrix = createMatrixMesh(stateCount, stateCount, [
      world.width * 0.58,
      0.14,
      world.width * 0.97,
      0.86
    ]);
    const mesh = uploadTractMesh(builder, tracts);
    mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, current.period));
    return {
      outputs: {
        quadrants,
        probabilities,
        matrixMesh: uploadMatrixMesh(builder, matrix, 'matrix'),
        matrixOutline: builder.upload('matrix-outline', 'float32x4', matrix.outline),
        matrixOutlineCount: matrix.outlineCount,
        ...mesh
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.quadrants,
      triangleCount: inputs.tracts.triangleCount
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
