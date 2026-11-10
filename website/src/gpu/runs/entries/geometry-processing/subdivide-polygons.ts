// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSubdividePolygons splits star-shaped polygons into convex pieces of at most
// `maximumVertices`. Each piece is drawn in its own color from the output positions, using the
// per-piece vertex counts the operation wrote.

import {GPUSubdividePolygons} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  applyVertexDrag,
  createPolygonSet,
  uploadPolygonSet,
  writePolygonSet
} from './shared/polygon-set';
import {carryNearestVertex, createDragFields, getHeroRest} from './shared/pointer-support';
import {addKernel} from './shared/shape-segments';
import {parameters} from './subdivide-polygons.parameters';

const PIECES_PER_POLYGON = 24;
const OPTIONS = {vertexCount: 12, rows: 2, columns: 4};

export default defineOperationRun({
  parameters,
  caption: 'Each star split into convex pieces, one color per piece.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {...OPTIONS, irregularity: 0.45});
    return {seed, world, set, ...createDragFields(set, 2)};
  },
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const maximumVertices = values.maximumVertices;
    const pieceCapacity = inputs.set.count * PIECES_PER_POLYGON;
    const vertexCapacity = pieceCapacity * maximumVertices;
    const pieceVertices = builder.allocate('piece-vertices', 'float32x2', vertexCapacity);
    const pieces = builder.allocate('pieces', 'uint32x4', pieceCapacity);
    builder.graph.add(
      new GPUSubdividePolygons({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        method: 'convex',
        maximumVertices,
        output: {
          positions: pieceVertices.view,
          pieces: pieces.view,
          status: builder.allocate('status', 'uint32x4', 1).view
        }
      })
    );
    // Link each piece's vertices into a ring and give them the piece number as category.
    const edges = builder.allocate('piece-edges', 'uint32x2', vertexCapacity);
    const categories = builder.allocate('piece-categories', 'uint32', vertexCapacity);
    addKernel(builder, {
      id: 'piece-rings',
      invocationCount: vertexCapacity,
      bindings: [
        {name: 'pieces', buffer: pieces, access: 'read'},
        {name: 'edges', buffer: edges, access: 'read_write'},
        {name: 'categories', buffer: categories, access: 'read_write'}
      ],
      body: `let piece = index / ${maximumVertices}u;
  let slot = index % ${maximumVertices}u;
  let vertexCount = pieces[piecesOffset + piece * 4u];
  let isWritten = slot < vertexCount;
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = select(index, piece * ${maximumVertices}u + (slot + 1u) % max(vertexCount, 1u), isWritten);
  categories[categoriesOffset + index] = select(0xffffffffu, piece, isWritten);`
    });
    return {
      outputs: {...polygons, pieceVertices, edges, categories, pieces},
      update: current => {
        const next = createPolygonSet(makeRandom(inputs.seed), inputs.world, {
          ...OPTIONS,
          irregularity: current.irregularity
        });
        applyVertexDrag(next, inputs.drag);
        inputs.set = next;
        writePolygonSet(polygons, next);
      }
    };
  },
  // The pointer pushes a vertex inward to make a reflex corner; convex pieces re-form around it.
  interaction: {
    kind: 'drag',
    hint: 'push a vertex inward',
    rest: ({inputs, toUnit}) => toUnit(getHeroRest(inputs)),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const carried = carryNearestVertex(inputs, pointer, 0, 0.4);
      const feature = inputs.drag?.feature ?? 0;
      const pieces = peek(outputs.pieces);
      let pieceCount = 0;
      if (pieces) {
        for (let row = 0; row < pieces.length; row += 4) {
          if (pieces[row] > 0 && pieces[row + 1] === feature) pieceCount++;
        }
      }
      return {
        rerun: true,
        glyphs: carried.glyphs,
        readout: `maximumVertices ${values.maximumVertices} · ${pieces ? pieceCount : '...'} pieces`
      };
    }
  },
  rebuildOn: ['maximumVertices'],
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {
      layer: 'segments',
      nodes: outputs.pieceVertices,
      edges: outputs.edges,
      values: outputs.categories,
      radius: 2.2
    }
  ]
});
