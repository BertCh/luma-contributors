// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFuzzyContiguity over tiles whose boundaries do not touch. Every link the operation wrote is
// drawn; tiles are colored by neighbor count. Raising `tolerance` past the sliver width links tiles
// that exact contiguity would miss.

import {GPUFuzzyContiguity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './fuzzy-contiguity.parameters';
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
  caption: 'Tiles separated by a sliver link once the tolerance exceeds the gap.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, {seed, world}, values) {
    const tiles = createQuadTiles(createSeededRandom(seed), world, 9, 6, values.gap, 0.08);
    const mesh = uploadTileMesh(builder, tiles);
    const weights = allocateWeights(builder, 'fuzzy', tiles.count, tiles.count * 12);
    const featureStatus = builder.allocate('feature-status', 'uint32', tiles.count);
    const added = builder.allocate('added', 'uint32', weights.capacity);
    builder.graph.add(
      new GPUFuzzyContiguity({
        polygons: {
          kind: 'polygons',
          positions: builder.upload('positions', 'float32x2', tiles.positions).view,
          ringOffsets: builder.upload('ring-offsets', 'uint32', tiles.ringOffsets).view,
          polygonOffsets: builder.upload('polygon-offsets', 'uint32', tiles.polygonOffsets).view,
          featureOffsets: builder.upload('feature-offsets', 'uint32', tiles.featureOffsets).view
        },
        tolerance: values.tolerance,
        output: {
          ...weights.port,
          featureStatus: featureStatus.view,
          added: added.view
        }
      })
    );
    const cardinality = addCardinality(builder, 'fuzzy', weights);
    const links = addLinkSegments(builder, 'fuzzy', weights, mesh.centers);
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
  rebuildOn: ['tolerance', 'gap']
});
