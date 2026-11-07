// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsMinimumSpanningTree connects neighboring tracts into the spanning tree of
// least attribute dissimilarity (the tree SKATER cuts). The faint lines are every adjacency, the
// bright ones are the tree edges the operation selected, tracts are shaded by the first attribute.

import {GPUSpatialWeightsMinimumSpanningTree} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './spatial-weights-minimum-spanning-tree.parameters';
import {asPairs, createAdjacencyEdges, createRegionScene} from './support/regions';
import {uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Minimum spanning tree over tract adjacency; edge cost is attribute dissimilarity.',
  createInputs: ({random, world}) => ({random, world}),
  build(builder, {world}, current) {
    const scene = createRegionScene(makeRandom(7), world, current.tractRows, 2);
    const {tracts} = scene;
    const columnCount = current.columnCount;
    const attributes =
      columnCount === 2
        ? scene.attributes
        : Float32Array.from({length: tracts.count}, (_, row) => scene.attributes[row * 2]);
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const endpoints = builder.allocate('endpoints', 'uint32', 2 * tracts.count);
    builder.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        weights,
        values: builder.upload('attributes', 'float32', attributes).view,
        columnCount,
        standardize: current.standardize === 1,
        output: {
          treeEdgeFlags: builder.allocate('flags', 'uint32', tracts.contiguity.neighbors.length)
            .view,
          componentLabels: builder.allocate('components', 'uint32', tracts.count).view,
          standardizedValues:
            current.standardize === 1
              ? builder.allocate('standardized', 'float32', tracts.count * columnCount).view
              : undefined,
          edges: {
            ids: builder.allocate('edge-ids', 'uint32', tracts.count).view,
            status: {
              count: builder.allocate('edge-count', 'uint32', 1).view,
              overflow: builder.allocate('edge-overflow', 'uint32', 1).view
            }
          },
          edgeEndpoints: endpoints.view
        }
      })
    );
    return {
      outputs: {
        tracts,
        treeEdges: asPairs(endpoints),
        adjacency: builder.upload('adjacency', 'uint32x2', createAdjacencyEdges(tracts)),
        attribute: builder.upload(
          'attribute',
          'float32',
          Float32Array.from({length: tracts.count}, (_, row) => attributes[row * columnCount])
        ),
        ...uploadTractMesh(builder, tracts)
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.attribute,
      triangleCount: outputs.tracts.triangleCount,
      range: [0, 1]
    },
    {layer: 'segments', context: true, nodes: outputs.centers, edges: outputs.adjacency},
    {
      layer: 'segments',
      nodes: outputs.centers,
      edges: outputs.treeEdges,
      color: [1, 1, 1, 1],
      radius: 2.5
    }
  ]
});
