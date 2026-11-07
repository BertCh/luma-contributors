// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUInequality over household incomes grouped by tract: one inequality index per zone, written by
// the GPU and drawn as the tract fill. `output` picks which index is shown.

import {
  GPUInequality,
  GPU_INEQUALITY_PARAMETER_LENGTH,
  getGPUInequalityParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {drawNormal, uploadTractGeometry} from './_lib/helpers';
import {parameters} from './inequality.parameters';

const HOUSEHOLDS_PER_TRACT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Inequality index of the incomes inside each tract.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world, 14, 0.1);
    const rowCount = tracts.count * HOUSEHOLDS_PER_TRACT;
    return {
      tracts,
      normals: Float32Array.from({length: rowCount}, () => drawNormal(random)),
      zoneIds: Uint32Array.from({length: rowCount}, (_, row) =>
        Math.floor(row / HOUSEHOLDS_PER_TRACT)
      )
    };
  },
  build(builder, inputs) {
    const {tracts} = inputs;
    const rowCount = inputs.normals.length;
    const incomes = builder.allocate('incomes', 'float32', rowCount);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_INEQUALITY_PARAMETER_LENGTH
    );
    const gini = builder.allocate('gini', 'float32', tracts.count);
    const theilT = builder.allocate('theil-t', 'float32', tracts.count);
    const atkinson = builder.allocate('atkinson', 'float32', tracts.count);
    const hoover = builder.allocate('hoover', 'float32', tracts.count);
    builder.graph.add(
      new GPUInequality({
        values: incomes.view,
        zoneIds: builder.upload('zone-ids', 'uint32', inputs.zoneIds).view,
        zoneCount: tracts.count,
        parameters: parameterBuffer.view,
        output: {gini: gini.view, theilT: theilT.view, atkinson: atkinson.view, hoover: hoover.view}
      })
    );
    return {
      outputs: {...uploadTractGeometry(builder, tracts), gini, theilT, atkinson, hoover},
      update: current => {
        incomes.write(
          Float32Array.from(inputs.normals, (normal, row) => {
            const zone = Math.floor(row / HOUSEHOLDS_PER_TRACT);
            const sigma = (0.15 + 0.7 * tracts.values[zone]) * current.spread;
            return Math.exp(10 + normal * sigma);
          })
        );
        parameterBuffer.write(getGPUInequalityParameterValues({epsilon: current.epsilon}));
      }
    };
  },
  rebuildOn: [],
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: [outputs.gini, outputs.theilT, outputs.atkinson, outputs.hoover][values.output],
      triangleCount: inputs.tracts.triangleCount,
      range: [0, [0.5, 0.6, 0.35, 0.32][values.output]]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
  ]
});
