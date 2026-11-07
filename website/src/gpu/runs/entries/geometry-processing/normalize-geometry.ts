// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNormalizeGeometry canonicalizes a collection of rings, lines and points. The input lists its
// children in a shuffled order with ring starts rotated; the output is colored by each leaf's
// canonical rank in the collection (right), while the input (left) is colored by its own shuffled
// order, so only the left colors move when the child order changes.

import {GPUNormalizeGeometry} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createGeometryTree} from './shared/geometry-tree';
import {addKernel, addPanel} from './shared/shape-segments';
import {parameters} from './normalize-geometry.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Leaves colored by canonical rank; the input order and ring starts do not matter.',
  createInputs: ({world}) => ({world, tree: createGeometryTree(world, {shuffleSeed: 3, start: 2})}),
  build(builder, {world, tree}) {
    const positions = builder.upload('positions', 'float32x2', tree.positions);
    const nodes = builder.upload('nodes', 'uint32x4', tree.nodes);
    const children = builder.upload('children', 'uint32', tree.children);
    const normalized = builder.allocate('normalized', 'float32x2', tree.positions.length / 2);
    const normalizedChildren = builder.allocate(
      'normalized-children',
      'uint32',
      tree.children.length
    );
    builder.graph.add(
      new GPUNormalizeGeometry({
        positions: positions.view,
        nodes: nodes.view,
        children: children.view,
        output: {
          positions: normalized.view,
          children: normalizedChildren.view,
          status: builder.allocate('status', 'uint32', tree.nodes.length / 4).view
        }
      })
    );
    // rank[leaf] = position of the leaf in the normalized root's child list.
    const rank = builder.allocate('rank', 'uint32', tree.leafCount);
    addKernel(builder, {
      id: 'leaf-ranks',
      invocationCount: tree.leafCount,
      bindings: [
        {name: 'children', buffer: normalizedChildren, access: 'read'},
        {name: 'rank', buffer: rank, access: 'read_write'}
      ],
      body: 'rank[rankOffset + children[childrenOffset + index]] = index;'
    });
    const leafOfVertex = builder.upload('leaf-of-vertex', 'uint32', tree.leafOfVertex);
    const vertexRanks = builder.allocate('vertex-ranks', 'uint32', tree.leafOfVertex.length);
    addKernel(builder, {
      id: 'vertex-ranks',
      invocationCount: tree.leafOfVertex.length,
      bindings: [
        {name: 'rank', buffer: rank, access: 'read'},
        {name: 'leafOfVertex', buffer: leafOfVertex, access: 'read'},
        {name: 'vertexRanks', buffer: vertexRanks, access: 'read_write'}
      ],
      body: 'vertexRanks[vertexRanksOffset + index] = rank[rankOffset + leafOfVertex[leafOfVertexOffset + index]];'
    });
    const inputRanks = builder.allocate('input-ranks', 'uint32', tree.leafOfVertex.length);
    const half = {scaleX: 0.5, scaleY: 0.8, offsetY: 0.1};
    return {
      outputs: {
        input: addPanel(builder, 'input', positions, {...half, offsetX: 0}),
        normalized: addPanel(builder, 'normalized', normalized, {
          ...half,
          offsetX: world.width / 2
        }),
        inputRanks,
        vertexRanks,
        edges: builder.upload('edges', 'uint32x2', tree.edges)
      },
      update: values => {
        const next = createGeometryTree(world, {shuffleSeed: values.shuffle, start: 2});
        positions.write(next.positions);
        children.write(next.children);
        // Input rank of every vertex: where its leaf sits in the shuffled child list.
        const rankOfLeaf = new Uint32Array(next.leafCount);
        next.children.forEach((leaf, index) => (rankOfLeaf[leaf] = index));
        inputRanks.write(Uint32Array.from(next.leafOfVertex, leaf => rankOfLeaf[leaf]));
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.input,
      edges: outputs.edges,
      values: outputs.inputRanks,
      radius: 2.6
    },
    {layer: 'dots', positions: outputs.input, values: outputs.inputRanks, radius: 4},
    {
      layer: 'segments',
      nodes: outputs.normalized,
      edges: outputs.edges,
      values: outputs.vertexRanks,
      radius: 2.6
    },
    {layer: 'dots', positions: outputs.normalized, values: outputs.vertexRanks, radius: 4}
  ]
});
