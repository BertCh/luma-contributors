// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegregation: diversity and segregation indices over population groups per tract. Each tract
// is colored by its local entropy (how mixed its population is): dark where one group dominates,
// bright where the groups mix. Raising the separation slider sorts the groups into their own
// regions, and the indices computed on the GPU fall accordingly. Hovering a tract outlines it and
// its rook neighbors and shows the group shares of its local environment as bars.

import {GPUSegregation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {
  createTractOutline,
  getTractNeighbors,
  pickTract,
  uploadTractMesh
} from './shared/tract-rates';
import {parameters} from './segregation-indices.parameters';

export default defineOperationRun({
  parameters,
  caption:
    'Modeled group mixtures across Chicago tracts, with local entropy computed from the group counts.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {
      tracts,
      populations: tracts.population,
      focus: {row: -1}
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
      outputs: {
        localEntropy,
        environment,
        globalIndices: indices,
        groupCount,
        // Render-only: the hovered tract's group shares.
        shares: builder.allocate('shares', 'float32', groupCount),
        ...uploadTractMesh(builder, tracts)
      },
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
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const row = pickTract(inputs.tracts, pointer, inputs.focus.row);
      inputs.focus.row = row;
      const environment = peek(outputs.environment);
      const entropy = peek(outputs.localEntropy)?.[row];
      const indices = peek(outputs.globalIndices);
      if (row < 0 || !environment) return {};
      outputs.shares.write(
        Float32Array.from(
          {length: outputs.groupCount},
          (_, group) => environment[row * outputs.groupCount + group]
        )
      );
      const glyphs = [
        createTractOutline(inputs.tracts, row),
        ...Array.from(getTractNeighbors(inputs.tracts, row), neighbor =>
          createTractOutline(inputs.tracts, neighbor)
        )
      ];
      return {
        glyphs,
        readout: `entropy ${entropy === undefined ? '…' : entropy.toFixed(2)} of ${Math.log(
          outputs.groupCount
        ).toFixed(2)} · H ${indices ? indices[0].toFixed(2) : '…'} · D ${
          indices ? indices[1].toFixed(2) : '…'
        }`
      };
    }
  },
  draw: ({inputs, outputs, values, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.meshIndices,
      owners: outputs.owners,
      values: outputs.localEntropy,
      triangleCount: inputs.tracts.triangleCount,
      range: [0, Math.log(values.groupCount)]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    // Group shares of the hovered tract's local environment.
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.shares,
      range: [0, 1],
      rectangle: [0.03, 0.04, 0.2, 0.22],
      color: [...theme.categories[0], 0.95]
    }
  ]
});
