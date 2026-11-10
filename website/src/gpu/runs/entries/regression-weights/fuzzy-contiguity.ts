// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFuzzyContiguity over tiles whose boundaries do not touch. Every link the operation wrote is
// drawn; tiles are colored by neighbor count. Raising `tolerance` past the sliver width links tiles
// that exact contiguity would miss.

import {GPUFuzzyContiguity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findPolygonAt} from '../../pick';
import {parameters} from './fuzzy-contiguity.parameters';
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
  caption: 'Chicago tract neighbors linked across modeled boundary gaps.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, {seed, world}, values) {
    const tiles = createTractTiles(createSeededRandom(seed), world, values.gap);
    const mesh = uploadTileMesh(builder, tiles);
    const weights = allocateWeights(builder, 'fuzzy', tiles.count, tiles.count * (tiles.count - 1));
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
    const addedLinks = addLinkSegments(builder, 'added', weights, mesh.centers, {onlyWhere: added});
    return {outputs: {tiles, mesh, weights, added, cardinality, links, addedLinks}};
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
    drawLinks(outputs.links, [0.6, 0.65, 0.85, 0.45]),
    // The links the operation added: contact that exact contiguity would miss across the gaps.
    {layer: 'lines', segments: outputs.addedLinks, color: [1, 0.5, 0.15, 0.95], radius: 1.8},
    {
      layer: 'dots',
      context: true,
      positions: outputs.mesh.centers,
      radius: 3
    }
  ],
  rebuildOn: ['tolerance', 'gap'],
  // The pointer picks a tract; its fuzzy neighbors are outlined and linked.
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
      const neighbors = tract < 0 ? null : peekRow(peek, outputs.weights, tract);
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
        readout: `tract ${tract} · ${neighbors.length} neighbors · tolerance ${values.tolerance}`
      };
    }
  }
});
