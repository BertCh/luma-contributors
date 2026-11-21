// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NetworkPathExtractionOperationProps} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';

const status = {count: 'count', requiredCount: 'required', overflow: 'overflow'};
const nodeOutput: NetworkPathExtractionOperationProps['output'] = {
  capacity: 2,
  ids: 'ids',
  status,
  // @ts-expect-error Edge rows belong only to the edge output.
  edgeRows: 'wrong'
};
const edgeOutput: NonNullable<NetworkPathExtractionOperationProps['edges']>['output'] = {
  capacity: 2,
  ids: 'edge-ids',
  status,
  // @ts-expect-error Node rows belong only to the node output.
  nodeRows: 'wrong'
};
void [nodeOutput, edgeOutput];

import {
  computeNetworkPathExtractionOnCPU,
  type CPUNetworkPathExtractionInput
} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
const input = {
  predecessors: Uint32Array.of(0xffffffff, 0),
  costs: Float32Array.of(0, 1),
  targets: Uint32Array.of(1),
  capacity: 2
};
const resolver = {
  offsets: Uint32Array.of(0, 1, 1),
  neighbors: Uint32Array.of(1),
  weights: Float32Array.of(1),
  capacity: 1
};
const definite = computeNetworkPathExtractionOnCPU({
  ...input,
  edges: resolver,
  predecessorEdges: Uint32Array.of(0xffffffff, 0)
});
const edgeCount: Uint32Array = definite.edges.count;
function checkRetiredEdgeInference() {
  // @ts-expect-error Edge extraction requires authoritative predecessor slots.
  return computeNetworkPathExtractionOnCPU({...input, edges: resolver});
}
void checkRetiredEdgeInference;
const omitted = computeNetworkPathExtractionOnCPU(input);
const absent: undefined = omitted.edges;
function conditional(input: CPUNetworkPathExtractionInput) {
  const result = computeNetworkPathExtractionOnCPU(input);
  // @ts-expect-error A conditional resolver still requires an undefined check.
  const required: Uint32Array = result.edges.count;
  return [result.edges?.count, required];
}
void [edgeCount, absent, conditional];
