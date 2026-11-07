// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCanonicalGeometryEquality compares two normalized collections leaf by leaf. The right tree
// lists its children in another order, rotates the ring starts and displaces leaf k by a growing
// amount. Each leaf is colored by the equality flag the operation wrote for its pair, and the
// tolerance decides how many leaves still count as equal.

import {
  GPUCanonicalGeometryEquality,
  GPUNormalizeGeometry
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createGeometryTree} from './shared/geometry-tree';
import {addKernel} from './shared/shape-segments';
import {parameters} from './canonical-geometry-equality.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Leaves equal within the tolerance, after both trees are normalized.',
  createInputs: ({world}) => ({world, left: createGeometryTree(world, {shuffleSeed: 1, start: 0})}),
  build(builder, {world, left}, values) {
    const normalize = (
      id: string,
      tree: typeof left,
      positions: ReturnType<typeof builder.upload<'float32x2'>>
    ) => {
      const nodes = builder.upload(`${id}-nodes`, 'uint32x4', tree.nodes);
      const children = builder.upload(`${id}-children`, 'uint32', tree.children);
      const output = {
        positions: builder.allocate(`${id}-normalized`, 'float32x2', tree.positions.length / 2),
        children: builder.allocate(`${id}-normalized-children`, 'uint32', tree.children.length)
      };
      builder.graph.add(
        new GPUNormalizeGeometry({
          id,
          positions: positions.view,
          nodes: nodes.view,
          children: children.view,
          output: {
            positions: output.positions.view,
            children: output.children.view,
            status: builder.allocate(`${id}-status`, 'uint32', tree.nodes.length / 4).view
          }
        })
      );
      return {
        tree: {positions: output.positions.view, nodes: nodes.view, children: output.children.view},
        positions: output.positions
      };
    };
    const leftPositions = builder.upload('left-positions', 'float32x2', left.positions);
    const rightPositions = builder.upload('right-positions', 'float32x2', left.positions);
    const leftTree = normalize('left', left, leftPositions);
    const rightTree = normalize('right', left, rightPositions);
    // One pair per leaf (same leaf row on both sides), then the roots.
    const pairs = Uint32Array.from([
      ...Array.from({length: left.leafCount}, (_, leaf) => [leaf, leaf]).flat(),
      left.rootId,
      left.rootId
    ]);
    const equal = builder.allocate('equal', 'uint32', left.leafCount + 1);
    builder.graph.add(
      new GPUCanonicalGeometryEquality({
        left: leftTree.tree,
        right: rightTree.tree,
        pairs: builder.upload('pairs', 'uint32x2', pairs).view,
        tolerance: values.tolerance,
        output: equal.view
      })
    );
    const leafOfVertex = builder.upload('leaf-of-vertex', 'uint32', left.leafOfVertex);
    const vertexEqual = builder.allocate('vertex-equal', 'uint32', left.leafOfVertex.length);
    addKernel(builder, {
      id: 'vertex-equal',
      invocationCount: left.leafOfVertex.length,
      bindings: [
        {name: 'equal', buffer: equal, access: 'read'},
        {name: 'leafOfVertex', buffer: leafOfVertex, access: 'read'},
        {name: 'vertexEqual', buffer: vertexEqual, access: 'read_write'}
      ],
      body: 'vertexEqual[vertexEqualOffset + index] = equal[equalOffset + leafOfVertex[leafOfVertexOffset + index]];'
    });
    return {
      outputs: {
        positions: leftPositions,
        vertexEqual,
        edges: builder.upload('edges', 'uint32x2', left.edges)
      },
      update: current => {
        const right = createGeometryTree(world, {
          shuffleSeed: 5,
          start: 3,
          displacement: current.displacement
        });
        rightPositions.write(right.positions);
      }
    };
  },
  rebuildOn: ['tolerance'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: outputs.edges,
      values: outputs.vertexEqual,
      radius: 3
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.vertexEqual,
      radius: 6
    }
  ]
});
