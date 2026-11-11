// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display adapters for the network time runs: small GPU kernels that derive a one-row budget from a
// cost tree, shape per-slot slack into drawable values, and pick one percentile column. They read
// operation outputs on the GPU; nothing is read back.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/**
 * The prism budget: `forward[destination] * (1 + spare)` in one row, the fastest trip from the
 * origin to the destination plus a share of spare time. `spare` is a one-row float32 buffer.
 */
export function addPrismBudget(
  builder: RunBuilder,
  options: {
    forward: RunBuffer<'float32'>;
    destination: RunBuffer<'uint32'>;
    spare: RunBuffer<'float32'>;
  }
): RunBuffer<'float32'> {
  const budget = builder.allocate('budget', 'float32', 1);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: 'prism-budget',
      operation: 'display-prism-budget',
      invocationCount: 1,
      bindings: [
        {name: 'forward', view: options.forward.view, type: 'f32', access: 'read'},
        {name: 'destination', view: options.destination.view, type: 'u32', access: 'read'},
        {name: 'spare', view: options.spare.view, type: 'f32', access: 'read'},
        {name: 'budget', view: budget.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  budget[budgetOffset] = forward[forwardOffset + destination[destinationOffset]] * (1.0 + spare[spareOffset]);`
    })
  );
  return budget;
}

/**
 * Tightness of the edges in the prism, `1 - slack / spare time`, written to both endpoints of every slot
 * line (the layout the segments layer reads): 1 (bright) on the shortest path, 0.15 where the whole spare time is left. Edges outside the prism get 0, the dark end of the ramp, so they stay quiet beside the context roads.
 */
export function addPrismTightness(
  builder: RunBuilder,
  options: {
    inPrism: RunBuffer<'uint32'>;
    slack: RunBuffer<'float32'>;
    budget: RunBuffer<'float32'>;
    spare: RunBuffer<'float32'>;
    slotCount: number;
  }
): RunBuffer<'float32'> {
  const doubled = builder.allocate('prism-tightness', 'float32', options.slotCount * 2);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: 'prism-tightness',
      operation: 'display-prism-tightness',
      invocationCount: options.slotCount,
      bindings: [
        {name: 'inPrism', view: options.inPrism.view, type: 'u32', access: 'read'},
        {name: 'slack', view: options.slack.view, type: 'f32', access: 'read'},
        {name: 'budget', view: options.budget.view, type: 'f32', access: 'read'},
        {name: 'spare', view: options.spare.view, type: 'f32', access: 'read'},
        {name: 'doubled', view: doubled.view, type: 'f32', access: 'read_write'}
      ],
      body: `
  let isInside = inPrism[inPrismOffset + index] == 1u;
  let room = max(budget[budgetOffset] * spare[spareOffset] / (1.0 + spare[spareOffset]), 1e-6);
  let tightness = 1.0 - slack[slackOffset + index] / room;
  let value = select(0.0, 0.15 + 0.85 * clamp(tightness, 0.0, 1.0), isInside);
  doubled[doubledOffset + index] = value;
  doubled[doubledOffset + ${options.slotCount}u + index] = value;`
    })
  );
  return doubled;
}

/**
 * Column `selection[0]` of a vertex-major table with `stride` columns per vertex: one value per
 * vertex, for the segments layer. `selection[0] == stride` selects last minus second column.
 */
export function addTableColumn(
  builder: RunBuilder,
  options: {
    id: string;
    table: RunBuffer<'float32'>;
    stride: number;
    vertexCount: number;
    selection: RunBuffer<'uint32'>;
  }
): RunBuffer<'float32'> {
  const column = builder.allocate(`${options.id}-column`, 'float32', options.vertexCount);
  builder.graph.add(
    createWGSLKernelNode(builder.graph, {
      id: `${options.id}-column`,
      operation: 'display-table-column',
      invocationCount: options.vertexCount,
      bindings: [
        {name: 'table', view: options.table.view, type: 'f32', access: 'read'},
        {name: 'selection', view: options.selection.view, type: 'u32', access: 'read'},
        {name: 'column', view: column.view, type: 'f32', access: 'read_write'}
      ],
      // A selection equal to the stride draws the spread of the last column over the second (p95 - p50).
      body: `
  let row = tableOffset + index * ${options.stride}u;
  let picked = selection[selectionOffset];
  column[columnOffset + index] = select(
    table[row + picked],
    table[row + ${options.stride - 1}u] - table[row + 1u],
    picked == ${options.stride}u
  );`
    })
  );
  return column;
}
