// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {SCAN_TOTAL_WGSL} from '../../analysis-core/kernels/scan-total';

/** Ordered inclusive saturated prefixes, independently per interleaved lane. Two reads per stage. @internal */
export function createScanOrderedPrefixNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    input: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    rowCount: number;
    laneCount?: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, input, output, rowCount} = props;
  const laneCount = props.laneCount ?? 1;
  const length = rowCount * laneCount;
  const stageCount = Math.ceil(Math.log2(rowCount));
  const scratch =
    stageCount > 1 ? createTransientView(graph, `${id}-scratch`, 'uint32', length) : output;
  const nodes: GPUCommandNode<Parameters>[] = [];
  let source = input;
  for (let stage = 0; stage < Math.max(1, stageCount); stage++) {
    // Choose parity so the final prefix is always in the caller's output, without an extra copy.
    const destination = (stageCount - stage) % 2 === 1 || stageCount === 0 ? output : scratch;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${stage}`,
        operation,
        variant: 'ordered-prefix',
        invocationCount: length,
        bindings: [
          {name: 'prefixInput', view: source, type: 'u32', access: 'read'},
          {name: 'prefixOutput', view: destination, type: 'u32', access: 'read_write'}
        ],
        declarations: `${SCAN_TOTAL_WGSL}\noverride PREFIX_DISTANCE: u32;\noverride PREFIX_LANES: u32;`,
        constants: {PREFIX_DISTANCE: 2 ** stage, PREFIX_LANES: laneCount},
        body: `var value = prefixInput[prefixInputOffset + index];
  if (index / PREFIX_LANES >= PREFIX_DISTANCE) {
    value = getSaturatedScanSum(prefixInput[prefixInputOffset + index - PREFIX_DISTANCE * PREFIX_LANES], value);
  }
  prefixOutput[prefixOutputOffset + index] = value;`
      })
    );
    source = destination;
  }
  return nodes;
}
