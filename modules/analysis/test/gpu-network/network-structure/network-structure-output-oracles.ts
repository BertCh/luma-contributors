// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {NetworkStructureOperationProps} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';

// Undirected triangle 0,1,2; bridge 2--3; self-loop at 3; isolated vertex 4; unused edge tail.
// Removing 2 disconnects 3; removing any other vertex does not disconnect its component.
// Only slots 6/7 bridge. Blocks use their minimum CSR slot: triangle 0, bridge 6, loop 8.
export const structureOracleInput = {
  offsets: Uint32Array.of(0, 2, 4, 7, 9, 9),
  neighbors: Uint32Array.of(1, 2, 0, 2, 0, 1, 3, 2, 3, 99),
  reverseEdgeSlots: Uint32Array.of(2, 4, 0, 5, 1, 3, 7, 6, 8, 4294967295)
};
export const structureOracleExpected = {
  bridges: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
  articulationPoints: [0, 0, 1, 0, 0],
  biconnectedComponents: [0, 0, 0, 0, 0, 0, 6, 6, 8, 4294967295],
  strongComponents: [0, 0, 0, 0, 4]
};
export const structureOracleInputs: Record<string, OperationGraphInput> = {
  offsets: {format: 'uint32', values: structureOracleInput.offsets},
  neighbors: {format: 'uint32', values: structureOracleInput.neighbors},
  reverseEdgeSlots: {format: 'uint32', values: structureOracleInput.reverseEdgeSlots}
};
export const structureOracleSelections = [
  'all',
  'bridges',
  'articulationPoints',
  'biconnectedComponents',
  'absent'
] as const;
export function createStructureOracleProps(
  mode: 'exact' | 'parallel',
  selected: (typeof structureOracleSelections)[number]
): NetworkStructureOperationProps {
  const output: NetworkStructureOperationProps['output'] = {};
  for (const port of ['bridges', 'articulationPoints', 'biconnectedComponents'] as const)
    if (selected === 'all' || selected === port) output[port] = port;
  if (selected === 'all' || selected === 'absent') output.strongComponents = 'strongComponents';
  return {
    nodeCount: 5,
    edgeCount: 10,
    mode,
    strongComponentsMaximumIterations: 64,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    reverseEdgeSlots: 'reverseEdgeSlots',
    output
  };
}
