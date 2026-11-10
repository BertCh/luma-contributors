// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUContiguityWeights over a jittered tile map. Tiles are colored by how many neighbors the
// operation found, and each CSR link is drawn from tile center to neighbor center. `snapTolerance`
// joins corners that digitizing left a sliver apart.

import {GPUContiguityWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findPolygonAt} from '../../pick';
import {parameters} from './contiguity-weights.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  createTractTiles,
  createSeededRandom,
  drawLinks,
  getRingGlyph,
  getRowLinkGlyphs,
  getRowPositions,
  peekRow,
  uploadTileMesh
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption:
    'Chicago tract adjacency under rook or queen contact with adjustable boundary tolerance.',
  createInputs: ({seed, world}) => ({seed, world}),
  // The pointer picks a tract; the CSR row the operation wrote for it is outlined and linked.
  build(builder, {seed, world}, values) {
    const tiles = createTractTiles(createSeededRandom(seed), world, values.gap);
    const mesh = uploadTileMesh(builder, tiles);
    const weights = allocateWeights(
      builder,
      'contiguity',
      tiles.count,
      tiles.count * (tiles.count - 1)
    );
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
    return {outputs: {tiles, mesh, weights, cardinality, links}};
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.mesh.vertices,
      indices: outputs.mesh.indices,
      owners: outputs.mesh.owners,
      values: outputs.cardinality,
      triangleCount: outputs.tiles.triangleIndices.length / 3,
      range: [0, 10],
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
  rebuildOn: ['criterion', 'snapTolerance', 'gap'],
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.5, 0.5],
    resolve(pointer, {outputs, values, peek}) {
      const {tiles} = outputs;
      const tract = findPolygonAt(
        tiles.meshVertices,
        tiles.triangleIndices,
        tiles.triangleOwners,
        pointer.x,
        pointer.y
      );
      if (tract < 0) return {};
      const neighbors = peekRow(peek, outputs.weights, tract);
      if (!neighbors) return {};
      const center = [tiles.centers[tract * 2], tiles.centers[tract * 2 + 1]] as const;
      return {
        focus: tract,
        glyphs: [
          getRingGlyph(tiles.positions, tiles.ringOffsets, tract),
          ...Array.from(neighbors, row =>
            getRingGlyph(tiles.positions, tiles.ringOffsets, row, true)
          ),
          ...getRowLinkGlyphs(center, getRowPositions(tiles.centers, neighbors))
        ],
        readout: `tract ${tract} · ${neighbors.length} neighbors · snapTolerance ${values.snapTolerance}`
      };
    }
  }
});
