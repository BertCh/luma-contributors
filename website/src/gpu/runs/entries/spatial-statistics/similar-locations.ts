// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSimilarLocations: ranks every tract by its distance to a reference tract in a space of four
// attributes (standardized). The colors are the ranks of that distance (bright = most different,
// the same scale whatever the weights); the reference tract and its 12 most similar tracts are
// outlined. The map is smooth in attribute space but not in geography, so similar tracts are often
// far apart. The pointer picks the reference tract.

import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  GPUSimilarLocations
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {createTractOutline, pickTract, uploadTractMesh} from './shared/tract-rates';
import {parameters} from './similar-locations.parameters';

const COLUMN_COUNT = 4;
const RESULT_CAPACITY = 40;
const DEFAULT_TRACT = 40;
const SIMILAR_COUNT = 12;
const STANDARDIZATIONS = ['z-score', 'rank'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Distance of every tract from the reference tract in attribute space.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    const attributes = new Float32Array(tracts.count * COLUMN_COUNT);
    for (let tract = 0; tract < tracts.count; tract++) {
      attributes.set(
        [
          tracts.povertyShare[tract],
          tracts.unemploymentShare[tract],
          tracts.age65[tract] / tracts.population[tract],
          tracts.population[tract]
        ],
        tract * COLUMN_COUNT
      );
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
        ranks,
        distances,
        topIds,
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
              resultCount: SIMILAR_COUNT,
              weights: [current.firstWeight, current.secondWeight, 1, 1]
            },
            COLUMN_COUNT
          )
        );
      }
    };
  },
  rebuildOn: ['standardization'],
  interaction: {
    kind: 'place',
    hint: 'pick the reference tract',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.tracts.centers[DEFAULT_TRACT * 2],
        inputs.tracts.centers[DEFAULT_TRACT * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const picked = pickTract(inputs.tracts, pointer, values.referenceTract);
      const topIds = peek(outputs.topIds);
      const distances = peek(outputs.distances);
      const glyphs = [createTractOutline(inputs.tracts, picked)];
      let farthest = NaN;
      if (topIds) {
        for (let rank = 0; rank < SIMILAR_COUNT; rank++) {
          const tract = topIds[rank];
          if (tract >= inputs.tracts.count || tract === picked) continue;
          glyphs.push(createTractOutline(inputs.tracts, tract));
          if (distances) farthest = distances[tract];
        }
      }
      return {
        values: {referenceTract: picked},
        glyphs,
        readout: `reference tract ${picked} · ${SIMILAR_COUNT} most similar${
          Number.isFinite(farthest) ? ` within ${farthest.toFixed(1)}` : ''
        } · weights ${values.firstWeight}, ${values.secondWeight}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.meshIndices,
      owners: outputs.owners,
      values: outputs.ranks,
      ramp: 'sequential',
      range: [0, inputs.tracts.count],
      triangleCount: inputs.tracts.triangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
  ]
});
