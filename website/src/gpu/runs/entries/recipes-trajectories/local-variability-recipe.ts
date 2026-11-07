// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addLocalVariabilityRecipe: local spatial heteroscedasticity (LOSH) fills each tract by how much
// its neighbors' values scatter; dots are the local Moran quadrants (HH, LL, LH, HL) next to it.

import {
  addLocalVariabilityRecipe,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {getTractFill, uploadTractMesh} from './lib/tracts';
import {parameters} from './local-variability-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local dispersion (fill) beside local Moran clusters (dots).',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {
      tracts,
      draws: Float32Array.from({length: tracts.count}, () => random() * 2 - 1),
      world
    };
  },
  build(builder, {tracts, draws, world}, values) {
    const valueBuffer = builder.allocate('values', 'float32', tracts.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const losh = builder.allocate('losh', 'float32', tracts.count);
    const quadrants = builder.allocate('quadrants', 'uint32', tracts.count);
    addLocalVariabilityRecipe(builder.graph, {
      weights: uploadSpatialWeights(builder, tracts.contiguity),
      values: valueBuffer.view,
      parameters: parameterBuffer.view,
      residualPower: values.residualPower === 1 ? 1 : 2,
      outputs: {losh: losh.view, quadrants: quadrants.view}
    });
    return {
      outputs: {losh, quadrants, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        parameterBuffer.write(getGPUSpatialAutocorrelationParameterValues());
        valueBuffer.write(
          Float32Array.from(tracts.values, (value, index) => {
            // Scatter grows from left to right, so dispersion differs across the map.
            const share = tracts.centers[index * 2] / world.width;
            return value + draws[index] * current.noise * 0.3 * share;
          })
        );
      }
    };
  },
  rebuildOn: ['residualPower'],
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.losh, {ramp: 'sequential'}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    },
    {layer: 'dots', positions: outputs.mesh.centers, values: outputs.quadrants, radius: 4}
  ]
});
