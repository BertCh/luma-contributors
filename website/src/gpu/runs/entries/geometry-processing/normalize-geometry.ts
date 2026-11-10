// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNormalizeGeometry canonicalizes a collection of rings, lines and points. The input lists its
// children in a shuffled order with ring starts rotated; the output is colored by each leaf's
// canonical rank in the collection (right), while the input (left) is colored by its own shuffled
// order, so only the left colors move when the child order changes.

import {GPUNormalizeGeometry} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {createGeometryTree} from './shared/geometry-tree';
import {addKernel, addPanel} from './shared/shape-segments';
import {parameters} from './normalize-geometry.parameters';

const PANEL = {scaleX: 0.5, scaleY: 0.8, offsetY: 0.1};

export default defineOperationRun({
  parameters,
  caption: 'Leaves colored by canonical rank; the input order and ring starts do not matter.',
  createInputs: ({world}) => {
    const tree = createGeometryTree(world, {shuffleSeed: 3, start: 2});
    // Leaf centers in the input panel; the normalized panel is the same layout shifted right.
    const centers = new Float32Array(tree.leafCount * 2);
    for (let leaf = 0; leaf < tree.leafCount; leaf++) {
      const first = tree.nodes[leaf * 4 + 1];
      const count = tree.nodes[leaf * 4 + 2];
      for (let vertex = first; vertex < first + count; vertex++) {
        centers[leaf * 2] += (tree.positions[vertex * 2] * PANEL.scaleX) / count;
        centers[leaf * 2 + 1] +=
          (tree.positions[vertex * 2 + 1] * PANEL.scaleY + PANEL.offsetY) / count;
      }
    }
    return {
      world,
      tree,
      centers,
      index: createPointIndex(centers),
      focus: {leaf: -1},
      inputRank: Uint32Array.from(tree.children).map((_, leaf) => tree.children.indexOf(leaf))
    };
  },
  build(builder, inputs) {
    const {world, tree} = inputs;
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
    const half = PANEL;
    return {
      outputs: {
        rank,
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
        inputs.inputRank = rankOfLeaf;
        inputRanks.write(Uint32Array.from(next.leafOfVertex, leaf => rankOfLeaf[leaf]));
      }
    };
  },
  // The pointer picks a leaf in either panel; a link joins it to its counterpart and the ranks are read.
  interaction: {
    kind: 'probe',
    hint: 'hover a leaf',
    rest: ({inputs, toUnit}) => toUnit([inputs.centers[5 * 2], inputs.centers[5 * 2 + 1]]),
    resolve(pointer, {inputs, outputs, world, peek}) {
      const isRight = pointer.x > world.width / 2;
      const localX = isRight ? pointer.x - world.width / 2 : pointer.x;
      const leaf = chooseSticky(inputs.index, localX, pointer.y, inputs.focus.leaf);
      inputs.focus.leaf = leaf;
      if (leaf < 0) return {};
      const node = [inputs.tree.nodes[leaf * 4 + 1], inputs.tree.nodes[leaf * 4 + 2]];
      const glyphs: RunGlyph[] = [];
      const from: [number, number] = [inputs.centers[leaf * 2], inputs.centers[leaf * 2 + 1]];
      const to: [number, number] = [from[0] + world.width / 2, from[1]];
      for (const offset of [0, world.width / 2]) {
        if (node[1] === 1) {
          glyphs.push({shape: 'ring', at: [from[0] + offset, from[1]], radius: 0.02});
        } else {
          const points: number[] = [];
          for (let vertex = node[0]; vertex < node[0] + node[1]; vertex++) {
            points.push(
              inputs.tree.positions[vertex * 2] * PANEL.scaleX + offset,
              inputs.tree.positions[vertex * 2 + 1] * PANEL.scaleY + PANEL.offsetY
            );
          }
          glyphs.push({shape: 'polyline', points});
        }
      }
      glyphs.push({shape: 'line', from, to, dashed: true});
      const canonical = peek(outputs.rank)?.[leaf];
      return {
        focus: leaf,
        glyphs,
        readout: `input #${inputs.inputRank[leaf]} -> canonical #${canonical ?? '...'}`
      };
    }
  },
  draw: ({outputs, inputs}) => {
    const look = {ramp: 'sequential', range: [-5, inputs.tree.leafCount - 1]} as const;
    return [
      {
        layer: 'segments',
        nodes: outputs.input,
        edges: outputs.edges,
        values: outputs.inputRanks,
        radius: 2.6,
        ...look
      },
      {layer: 'dots', positions: outputs.input, values: outputs.inputRanks, radius: 4, ...look},
      {
        layer: 'segments',
        nodes: outputs.normalized,
        edges: outputs.edges,
        values: outputs.vertexRanks,
        radius: 2.6,
        ...look
      },
      {
        layer: 'dots',
        positions: outputs.normalized,
        values: outputs.vertexRanks,
        radius: 4,
        ...look
      }
    ];
  }
});
