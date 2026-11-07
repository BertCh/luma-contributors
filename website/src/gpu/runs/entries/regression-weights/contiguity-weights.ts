// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUContiguityWeights over a jittered tile map. Tiles are colored by how many neighbors the
// operation found, and each CSR link is drawn from tile center to neighbor center. `snapTolerance`
// joins corners that digitizing left a sliver apart.

import {GPUContiguityWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './contiguity-weights.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  createQuadTiles,
  createSeededRandom,
  drawLinks,
  uploadTileMesh
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Tiles colored by neighbor count; links are the weights the operation wrote.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, {seed, world}, values) {
    const tiles = createQuadTiles(createSeededRandom(seed), world, 9, 6, values.gap, 0.08);
    const mesh = uploadTileMesh(builder, tiles);
    const weights = allocateWeights(builder, 'contiguity', tiles.count, tiles.count * 9);
    builder.graph.add(
      new GPUContiguityWeights({
        criterion: values.criterion === 0 ? 'queen' : 'rook',
        polygons: {
          kind: 'polygons',
          positions: builder.upload('positions', 'float32x2', tiles.positions).view,
          ringOffsets: builder.upload('ring-offsets', 'uint32', tiles.ringOffsets).view,
          polygonOffsets: builder.upload('polygon-offsets', 'uint32', tiles.polygonOffsets).view,
          featureOffsets: builder.upload('feature-offsets', 'uint32', tiles.featureOffsets).view
        },
        snapTolerance: values.snapTolerance,
        output: weights.port
      })
    );
    const cardinality = addCardinality(builder, 'contiguity', weights);
    const links = addLinkSegments(builder, 'contiguity', weights, mesh.centers);
    return {outputs: {tiles, mesh, cardinality, links}};
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.mesh.vertices,
      indices: outputs.mesh.indices,
      owners: outputs.mesh.owners,
      values: outputs.cardinality,
      triangleCount: outputs.tiles.triangleIndices.length / 3,
      range: [0, 8],
      alpha: 0.85
    },
    {layer: 'lines', context: true, segments: outputs.mesh.outline},
    drawLinks(outputs.links),
    {
      layer: 'dots',
      context: true,
      positions: outputs.mesh.centers,
      radius: 3
    }
  ],
  rebuildOn: ['criterion', 'snapTolerance', 'gap']
});
