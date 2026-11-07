// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegregation: diversity and segregation indices over population groups per tract. Each tract
// is colored by its local entropy (how mixed its population is): dark where one group dominates,
// bright where the groups mix. Raising the separation slider sorts the groups into their own
// regions, and the indices computed on the GPU fall accordingly.

import {GPUSegregation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {uploadTractMesh} from './shared/tract-rates';
import {parameters} from './segregation-indices.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local entropy per tract: dark is one dominant group, bright is mixed.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {
      tracts,
      populations: Float32Array.from({length: tracts.count}, () => 200 + Math.floor(random() * 800))
    };
  },
  build(builder, {tracts, populations}, values) {
    const groupCount = values.groupCount;
    const isSpatial = values.scale === 1;
    const groupCounts = builder.allocate('group-counts', 'float32', tracts.count * groupCount);
    const scaleCount = 1;
    const indices = builder.allocate(
      'segregation-indices',
      'float32',
      3 + 3 * groupCount + groupCount * groupCount
    );
    const environment = builder.allocate(
      'environment',
      'float32',
      scaleCount * tracts.count * groupCount
    );
    const localEntropy = builder.allocate('local-entropy', 'float32', scaleCount * tracts.count);
    builder.graph.add(
      new GPUSegregation({
        groupCount,
        groupCounts: groupCounts.view,
        scales: [isSpatial ? uploadSpatialWeights(builder, tracts.contiguity) : null],
        output: {
          indices: indices.view,
          localEnvironment: environment.view,
          localEntropy: localEntropy.view
        }
      })
    );
    return {
      outputs: {localEntropy, ...uploadTractMesh(builder, tracts)},
      update: current => {
        // Group g prefers the part of the field around (g + 0.5) / groupCount.
        const counts = new Float32Array(tracts.count * groupCount);
        for (let tract = 0; tract < tracts.count; tract++) {
          const shares = Array.from({length: groupCount}, (_, group) => {
            const distance = tracts.values[tract] - (group + 0.5) / groupCount;
            return Math.exp(-current.separation * 14 * groupCount * distance * distance);
          });
          const total = shares.reduce((sum, share) => sum + share, 0);
          shares.forEach((share, group) => {
            counts[tract * groupCount + group] = Math.round((populations[tract] * share) / total);
          });
        }
        groupCounts.write(counts);
      }
    };
  },
  rebuildOn: ['groupCount', 'scale'],
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.localEntropy,
      triangleCount: inputs.tracts.triangleCount,
      range: [0, Math.log(values.groupCount)]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
  ]
});
