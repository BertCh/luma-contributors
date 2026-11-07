// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSimilarLocations: ranks every tract by its distance to a reference tract in a space of four
// attributes (standardized). The colors are those distances (bright = most different); the
// reference tract is the large dot. The map is smooth in attribute space but not in geography,
// so similar tracts are often far apart.

import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  GPUSimilarLocations
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField, createTracts} from '../../inputs';
import {uploadTractMesh} from './shared/tract-rates';
import {parameters} from './similar-locations.parameters';

const COLUMN_COUNT = 4;
const RESULT_CAPACITY = 40;
const STANDARDIZATIONS = ['zscore', 'rank'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Distance of every tract from the reference tract in attribute space.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    const fields = Array.from({length: COLUMN_COUNT}, () => createSmoothField(random, 3));
    const attributes = new Float32Array(tracts.count * COLUMN_COUNT);
    for (let tract = 0; tract < tracts.count; tract++) {
      const x = tracts.centers[tract * 2] / world.width;
      const y = tracts.centers[tract * 2 + 1] / world.height;
      fields.forEach((field, column) => {
        attributes[tract * COLUMN_COUNT + column] = field(x, y) * 100;
      });
    }
    return {tracts, attributes};
  },
  build(builder, {tracts, attributes}, values) {
    const attributeBuffer = builder.upload('attributes', 'float32', attributes);
    const referenceMask = builder.allocate('reference-mask', 'uint32', tracts.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUSimilarLocationsParameterLength(COLUMN_COUNT)
    );
    const ranks = builder.allocate('ranks', 'uint32', tracts.count);
    const distances = builder.allocate('distances', 'float32', tracts.count);
    const topIds = builder.allocate('top-ids', 'uint32', RESULT_CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    builder.graph.add(
      new GPUSimilarLocations({
        values: attributeBuffer.view,
        columnCount: COLUMN_COUNT,
        referenceMask: referenceMask.view,
        parameters: parameterBuffer.view,
        standardization: STANDARDIZATIONS[values.standardization],
        resultCapacity: RESULT_CAPACITY,
        output: {
          ranks: ranks.view,
          distances: distances.view,
          topIds: topIds.view,
          status: {count: count.view}
        }
      })
    );
    const reference = new Uint32Array(tracts.count);
    return {
      outputs: {
        distances,
        referenceMask,
        ...uploadTractMesh(builder, tracts)
      },
      update: current => {
        const tract = Math.min(Math.round(current.referenceTract), tracts.count - 1);
        reference.fill(0);
        reference[tract] = 1;
        referenceMask.write(reference);
        parameterBuffer.write(
          getGPUSimilarLocationsParameterValues(
            {
              resultCount: 12,
              weights: [current.firstWeight, current.secondWeight, 1, 1]
            },
            COLUMN_COUNT
          )
        );
      }
    };
  },
  rebuildOn: ['standardization'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.distances,
      triangleCount: inputs.tracts.triangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.referenceMask,
      ramp: 'sequential',
      range: [0, 1],
      radius: 3.5
    }
  ]
});
