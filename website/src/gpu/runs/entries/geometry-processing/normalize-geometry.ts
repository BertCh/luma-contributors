// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNormalizeGeometry canonicalizes a collection of rings, lines and points. Each panel lists the
// collection's children in slots, in reading order: the input (left) in its shuffled child order,
// the output (right) in the order the operation wrote, so a leaf jumps slots between the panels.
// Color is the kind of leaf (ring, line, point), the same in both panels.

import {GPUNormalizeGeometry} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../define-run';
import type {RunLabel} from '../../text';
import {createGeometryTree} from './shared/geometry-tree';
import {addKernel} from './shared/shape-segments';
import {parameters} from './normalize-geometry.parameters';

const PANEL = {scaleX: 0.5, scaleY: 0.8, offsetY: 0.1};
const SLOT_COLUMNS = 4;
const SLOT_ROWS = 3;
const KIND_NAMES = ['ring', 'line', 'point'] as const;

type World = {width: number; height: number};

/** Center of a slot in a panel, in world units. Slot 0 is the top left; slots read along rows. */
function getSlotCenter(world: World, slot: number, offsetX: number): [number, number] {
  const column = slot % SLOT_COLUMNS;
  const row = Math.floor(slot / SLOT_COLUMNS);
  return [
    ((column + 0.5) / SLOT_COLUMNS) * world.width * PANEL.scaleX + offsetX,
    (1 - (row + 0.5) / SLOT_ROWS) * world.height * PANEL.scaleY + PANEL.offsetY * world.height
  ];
}

/** The slot of a panel nearest a world position. */
function getSlotAt(world: World, x: number, offsetX: number, y: number): number {
  const columnFraction = (x - offsetX) / (world.width * PANEL.scaleX);
  const rowFraction = 1 - (y - PANEL.offsetY * world.height) / (world.height * PANEL.scaleY);
  const column = Math.min(SLOT_COLUMNS - 1, Math.max(0, Math.floor(columnFraction * SLOT_COLUMNS)));
  const row = Math.min(SLOT_ROWS - 1, Math.max(0, Math.floor(rowFraction * SLOT_ROWS)));
  return row * SLOT_COLUMNS + column;
}

/**
 * Places every leaf of a panel in the slot given per vertex: the vertex keeps its offset from the
 * center of its leaf's original cell, shrunk to the panel.
 */
function addSlotPanel(
  builder: RunBuilder,
  id: string,
  positions: RunBuffer<'float32x2'>,
  leafOfVertex: RunBuffer<'uint32'>,
  slotOfVertex: RunBuffer<'uint32'>,
  world: World,
  offsetX: number
): RunBuffer<'float32x2'> {
  const placed = builder.allocate(`${id}-slots`, 'float32x2', positions.length);
  const width = world.width.toFixed(6);
  const height = world.height.toFixed(6);
  addKernel(builder, {
    id: `${id}-slots-kernel`,
    invocationCount: positions.length,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'leafOfVertex', buffer: leafOfVertex, access: 'read'},
      {name: 'slotOfVertex', buffer: slotOfVertex, access: 'read'},
      {name: 'placed', buffer: placed, access: 'read_write'}
    ],
    body: `let leaf = leafOfVertex[leafOfVertexOffset + index];
  let slot = slotOfVertex[slotOfVertexOffset + index];
  let originalX = (f32(leaf % ${SLOT_COLUMNS}u) + 0.5) / ${SLOT_COLUMNS}.0 * ${width};
  let originalY = (f32(leaf / ${SLOT_COLUMNS}u) + 0.5) / ${SLOT_ROWS}.0 * ${height};
  let slotX = (f32(slot % ${SLOT_COLUMNS}u) + 0.5) / ${SLOT_COLUMNS}.0 * ${width} * ${PANEL.scaleX} + ${offsetX.toFixed(6)};
  let slotY = (1.0 - (f32(slot / ${SLOT_COLUMNS}u) + 0.5) / ${SLOT_ROWS}.0) * ${height} * ${PANEL.scaleY} + ${PANEL.offsetY} * ${height};
  placed[placedOffset + index * 2u] = (positions[positionsOffset + index * 2u] - originalX) * ${PANEL.scaleX} + slotX;
  placed[placedOffset + index * 2u + 1u] = (positions[positionsOffset + index * 2u + 1u] - originalY) * ${PANEL.scaleY} + slotY;`
  });
  return placed;
}

export default defineOperationRun({
  parameters,
  caption:
    'Leaves colored by kind (ring, line, point); the input order and ring starts do not matter.',
  createInputs: ({world}) => {
    const tree = createGeometryTree(world, {shuffleSeed: 3, start: 2});
    return {
      world,
      tree,
      focus: {slot: -1},
      // Slot of every leaf in the shuffled child list.
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
    const kindOfVertex = builder.upload(
      'kind-of-vertex',
      'uint32',
      Uint32Array.from(tree.leafOfVertex, leaf => leaf % KIND_NAMES.length)
    );
    return {
      outputs: {
        rank,
        input: addSlotPanel(builder, 'input', positions, leafOfVertex, inputRanks, world, 0),
        normalized: addSlotPanel(
          builder,
          'normalized',
          normalized,
          leafOfVertex,
          vertexRanks,
          world,
          world.width * PANEL.scaleX
        ),
        kindOfVertex,
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
  // The pointer picks a slot in either panel; a link joins the slot of that leaf in the input to its
  // slot in the output, and both ranks are read.
  interaction: {
    kind: 'probe',
    hint: 'hover a leaf',
    rest: ({inputs, toUnit}) => toUnit(getSlotCenter(inputs.world, 5, 0)),
    resolve(pointer, {inputs, outputs, world, peek}) {
      const rankOfLeaf = peek(outputs.rank);
      const outputX = world.width * PANEL.scaleX;
      const isRight = pointer.x > outputX;
      const slot = getSlotAt(world, pointer.x, isRight ? outputX : 0, pointer.y);
      if (!rankOfLeaf) return {};
      // The leaf sitting in that slot of the panel that was hovered.
      const leaf = isRight ? Array.from(rankOfLeaf).indexOf(slot) : inputs.tree.children[slot];
      if (leaf < 0) return {};
      inputs.focus.slot = slot;
      const inputSlot = inputs.inputRank[leaf];
      const outputSlot = rankOfLeaf[leaf];
      const from = getSlotCenter(world, inputSlot, 0);
      const to = getSlotCenter(world, outputSlot, outputX);
      const radius =
        0.4 *
        Math.min(
          (world.width * PANEL.scaleX) / SLOT_COLUMNS,
          (world.height * PANEL.scaleY) / SLOT_ROWS
        );
      return {
        focus: leaf,
        glyphs: [
          {shape: 'ring', at: from, radius},
          {shape: 'ring', at: to, radius},
          {shape: 'arrow', from, to}
        ],
        readout: `${KIND_NAMES[leaf % KIND_NAMES.length]}: input #${inputSlot} -> canonical #${outputSlot}`
      };
    }
  },
  draw: ({outputs, theme}) => {
    const titles: RunLabel[] = [
      {
        text: 'input order (shuffled)',
        at: [0.25, 0],
        space: 'canvas',
        anchor: 'top',
        offset: [0, -4],
        role: 'title'
      },
      {
        text: 'canonical order',
        at: [0.75, 0],
        space: 'canvas',
        anchor: 'top',
        offset: [0, -4],
        role: 'title'
      }
    ];
    const kinds: RunLabel[] = KIND_NAMES.map((name, index) => ({
      text: name,
      at: [0.38 + index * 0.11, 1],
      space: 'canvas',
      anchor: 'left',
      offset: [0, 9],
      swatch: index,
      role: 'label'
    }));
    return [
      {
        layer: 'segments',
        nodes: outputs.input,
        edges: outputs.edges,
        values: outputs.kindOfVertex,
        radius: 2.6
      },
      {layer: 'dots', positions: outputs.input, values: outputs.kindOfVertex, radius: 4},
      {
        layer: 'segments',
        nodes: outputs.normalized,
        edges: outputs.edges,
        values: outputs.kindOfVertex,
        radius: 2.6
      },
      {layer: 'dots', positions: outputs.normalized, values: outputs.kindOfVertex, radius: 4},
      {layer: 'text', labels: [...titles, ...kinds]}
    ];
  }
});
