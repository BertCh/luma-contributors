// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUNetworkPathExtractionOperationProps} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

// Directed path 0->1->2, weights 2,3; IDs deliberately differ from row numbers.
// Target 2 has node path [0,1,2], edge rows [0,1], cost 5; target 0 is the root.
export const PATH_INPUTS: Record<string, OperationGraphInput> = {
  predecessors: {format: 'uint32', values: Uint32Array.of(4294967295, 0, 1)},
  predecessorEdges: {format: 'uint32', values: Uint32Array.of(4294967295, 0, 1)},
  costs: {format: 'float32', values: Float32Array.of(0, 2, 5)},
  targets: {format: 'uint32', values: Uint32Array.of(2, 0)},
  nodeIds: {format: 'uint32', values: Uint32Array.of(41, 17, 90)},
  offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
  neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
  weights: {format: 'float32', values: Float32Array.of(2, 3)},
  edgeIds: {format: 'uint32', values: Uint32Array.of(501, 92)}
};

export function createPathOracleProps(
  capacity: number,
  selected?: 'lines' | 'pathCosts' | 'edgeLines' | 'edges'
): CPUNetworkPathExtractionOperationProps {
  const props = {
    nodeCount: 3,
    targetCapacity: 2,
    predecessors: 'predecessors',
    costs: 'costs',
    targets: 'targets',
    nodeIds: 'nodeIds',
    output: {
      capacity,
      ids: 'ids',
      lineOffsets: selected === 'lines' ? 'lines' : undefined,
      pathCosts: selected === 'pathCosts' ? 'pathCosts' : undefined,
      status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
    }
  };
  if (selected !== 'edges' && selected !== 'edgeLines') return props;
  return {
    ...props,
    predecessorEdges: 'predecessorEdges',
    edges: {
      edgeCount: 2,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      edgeIds: 'edgeIds',
      output: {
        capacity: 1,
        ids: 'edgeIdsOut',
        lineOffsets: selected === 'edgeLines' ? 'edgeLines' : undefined,
        status: {count: 'edgeCount', requiredCount: 'edgeRequired', overflow: 'edgeOverflow'}
      }
    }
  };
}
