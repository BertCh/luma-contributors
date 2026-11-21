// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {SCAN_EXACT_POPULATION_ACCUMULATOR_WGSL} from './scan-statistic-nearest-zones';

/** Ten uint32 limbs cover the exact sum of at most 2^24 positive finite float32 inputs. */
const POPULATION_WORD_COUNT = 10;

/** @internal Round the study total once, like complete-circle populations, via a parallel tree. */
export function createScanExactPopulationTotalNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    baseline: GraphDataView<'float32'>;
    totals: GraphDataView<'uint32'>;
    cells: number;
    blocks: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, baseline, totals, cells, blocks} = props;
  const cellsPerBlock = Math.ceil(cells / blocks);
  let partials = createTransientView(
    graph,
    `${id}-exact-population-blocks`,
    'uint32',
    blocks * POPULATION_WORD_COUNT
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-exact-population-blocks`,
      operation,
      invocationCount: blocks,
      bindings: [
        {name: 'baseline', view: baseline, type: 'u32', access: 'read'},
        {name: 'partials', view: partials, type: 'u32', access: 'read_write'}
      ],
      declarations: SCAN_EXACT_POPULATION_ACCUMULATOR_WGSL,
      body: `resetBoundaryPopulation();
  let first = index * ${cellsPerBlock}u;
  for (var cell = first; cell < min(first + ${cellsPerBlock}u, ${cells}u); cell++) {
    addBoundaryPopulationBits(baseline[baselineOffset + cell]);
  }
  for (var word = 0u; word < ${POPULATION_WORD_COUNT}u; word++) {
    partials[partialsOffset + index * ${POPULATION_WORD_COUNT}u + word] = boundaryPopulationWords[word];
  }`
    })
  ];
  let rows = blocks;
  let level = 0;
  do {
    const nextRows = Math.ceil(rows / 2);
    const final = nextRows === 1;
    const destination = final
      ? totals
      : createTransientView(
          graph,
          `${id}-exact-population-level-${level}`,
          'uint32',
          nextRows * POPULATION_WORD_COUNT
        );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-exact-population-merge-${level}`,
        operation,
        invocationCount: nextRows,
        bindings: [
          {name: 'partials', view: partials, type: 'u32', access: 'read'},
          {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
        ],
        declarations: SCAN_EXACT_POPULATION_ACCUMULATOR_WGSL,
        body: `resetBoundaryPopulation();
  for (var child = index * 2u; child < min(index * 2u + 2u, ${rows}u); child++) {
    for (var word = 0u; word < ${POPULATION_WORD_COUNT}u; word++) {
      addBoundaryPopulationWord(word, partials[partialsOffset + child * ${POPULATION_WORD_COUNT}u + word]);
    }
  }
  ${
    final
      ? `destination[destinationOffset + 1u] = bitcast<u32>(readBoundaryPopulation());
  for (var word = 0u; word < 10u; word++) {
    destination[destinationOffset + 7u + word] = boundaryPopulationWords[word];
  }`
      : `for (var word = 0u; word < ${POPULATION_WORD_COUNT}u; word++) {
    destination[destinationOffset + index * ${POPULATION_WORD_COUNT}u + word] = boundaryPopulationWords[word];
  }`
  }`
      })
    );
    partials = destination;
    rows = nextRows;
    level++;
  } while (rows > 1);
  return nodes;
}
