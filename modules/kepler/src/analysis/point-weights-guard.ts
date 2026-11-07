// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {addKeplerLinearKernel} from './analysis-scaling';

/**
 * Detects a 32-bit wrap of the neighbor search's total slot count, which would otherwise make a
 * very large distance band look like it fits its capacity. Per-row counts are summed into an exact
 * 64-bit value (low word with carry count). A carry marks the build as overflowed and saturates
 * `requiredCount`; unwrapped totals are untouched. Runs after the search node.
 */
export function addKeplerNeighborTotalGuard<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  neighborCounts: GraphDataView<'uint32'>,
  status: {overflow: GraphDataView<'uint32'>; requiredCount?: GraphDataView<'uint32'>}
): void {
  const accumulator = createTransientView(graph, `${id}-total-accumulator`, 'uint32', 2);
  addKeplerLinearKernel(
    graph,
    `${id}-total-initialize`,
    1,
    [{name: 'accumulator', view: accumulator, write: true, atomic: true}],
    `atomicStore(&accumulator[accumulatorOffset], 0u);
atomicStore(&accumulator[accumulatorOffset + 1u], 0u);`
  );
  addKeplerLinearKernel(
    graph,
    `${id}-total-sum`,
    neighborCounts.length,
    [
      {name: 'neighborCounts', view: neighborCounts},
      {name: 'accumulator', view: accumulator, write: true, atomic: true}
    ],
    `let count = neighborCounts[neighborCountsOffset + index];
let previous = atomicAdd(&accumulator[accumulatorOffset], count);
if (previous > 0xffffffffu - count) { atomicAdd(&accumulator[accumulatorOffset + 1u], 1u); }`
  );
  const targets = [
    ['overflow', status.overflow, 'atomicMax(&target[targetOffset], 1u);'] as const,
    ...(status.requiredCount
      ? [
          [
            'required',
            status.requiredCount,
            'atomicStore(&target[targetOffset], 0xffffffffu);'
          ] as const
        ]
      : [])
  ];
  for (const [name, view, body] of targets) {
    addKeplerLinearKernel(
      graph,
      `${id}-total-${name}`,
      1,
      [
        {name: 'accumulator', view: accumulator},
        {name: 'target', view, write: true, atomic: true}
      ],
      `if (accumulator[accumulatorOffset + 1u] != 0u) { ${body} }`
    );
  }
}
