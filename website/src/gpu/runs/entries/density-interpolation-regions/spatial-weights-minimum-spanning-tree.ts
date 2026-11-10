// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsMinimumSpanningTree connects neighboring tracts into the spanning tree of
// least attribute dissimilarity (the tree SKATER cuts). The faint lines are every adjacency, the
// bright ones are the tree edges the operation selected, tracts are shaded by the first attribute.

import {GPUSpatialWeightsMinimumSpanningTree} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './spatial-weights-minimum-spanning-tree.parameters';
import {asPairs, createAdjacencyEdges, createRegionScene} from './support/regions';
import {createTractGlyph, findMiddleTract, findTractAt, uploadTractMesh} from './support/geometry';
import type {RunGlyph} from '../../interaction';

export default defineOperationRun({
  parameters,
  caption: 'Minimum spanning tree over tract adjacency; edge cost is attribute dissimilarity.',
  createInputs: ({random, world}) => {
    // Only the default window, to place the poster's probe on a middle tract.
    const tracts = createTracts(random, world, 7);
    return {
      random,
      world,
      focus: {tract: findMiddleTract(tracts, world)},
      restCenter: [
        tracts.centers[findMiddleTract(tracts, world) * 2],
        tracts.centers[findMiddleTract(tracts, world) * 2 + 1]
      ] as const
    };
  },
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
        endpoints,
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
  // The hovered tract shows which of its adjacencies the tree kept (solid) and which it skipped
  // (dashed): the tree joins the most similar neighbors only.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: ({inputs, toUnit}) => toUnit(inputs.restCenter),
    resolve(pointer, {inputs, outputs, peek}) {
      const {tracts} = outputs;
      const tract = findTractAt(tracts, pointer.x, pointer.y);
      inputs.focus.tract = tract;
      const endpoints = peek(outputs.endpoints);
      const kept = new Set<number>();
      if (endpoints) {
        for (let edge = 0; edge < endpoints.length; edge += 2) {
          const [first, second] = [endpoints[edge], endpoints[edge + 1]];
          if (first === second) continue;
          if (first === tract) kept.add(second);
          if (second === tract) kept.add(first);
        }
      }
      const from: [number, number] = [tracts.centers[tract * 2], tracts.centers[tract * 2 + 1]];
      const glyphs: RunGlyph[] = [createTractGlyph(tracts, tract)];
      const {offsets, neighbors} = tracts.contiguity;
      for (let slot = offsets[tract]; slot < offsets[tract + 1]; slot++) {
        const to: [number, number] = [
          tracts.centers[neighbors[slot] * 2],
          tracts.centers[neighbors[slot] * 2 + 1]
        ];
        glyphs.push({shape: 'line', from, to, dashed: !kept.has(neighbors[slot])});
      }
      return {
        focus: tract,
        glyphs,
        readout: `tract ${tract} · ${kept.size} tree edges of ${offsets[tract + 1] - offsets[tract]} neighbors`
      };
    }
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
