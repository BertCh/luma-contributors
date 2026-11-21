// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkPathExtractionOnCPU,
  CPUNetworkPathExtractionOperation
} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {PATH_INPUTS, createPathOracleProps} from './network-path-extraction-output-oracles.fixture';

it('path CPU twin uses the same hand-enumerated path literals', () => {
  for (const capacity of [2, 4]) {
    const result = computeNetworkPathExtractionOnCPU({
      predecessors: PATH_INPUTS['predecessors'].values as Uint32Array,
      predecessorEdges: PATH_INPUTS['predecessorEdges'].values as Uint32Array,
      costs: PATH_INPUTS['costs'].values as Float32Array,
      targets: PATH_INPUTS['targets'].values as Uint32Array,
      nodeIds: PATH_INPUTS['nodeIds'].values as Uint32Array,
      capacity,
      edges: {
        offsets: PATH_INPUTS['offsets'].values as Uint32Array,
        neighbors: PATH_INPUTS['neighbors'].values as Uint32Array,
        weights: PATH_INPUTS['weights'].values as Float32Array,
        edgeIds: PATH_INPUTS['edgeIds'].values as Uint32Array,
        capacity: 1
      }
    });
    expect(Array.from(result.ids)).toEqual(capacity === 2 ? [41, 17] : [41, 17, 90, 41]);
    expect([result.count[0], result.requiredCount[0], result.overflow[0]]).toEqual([
      capacity,
      4,
      capacity === 2 ? 1 : 0
    ]);
    expect(Array.from(result.lineOffsets)).toEqual([0, 3, 4]);
    expect(Array.from(result.pathCosts)).toEqual([5, 0]);
    expect(Array.from(result.edges!.ids)).toEqual([501]);
    expect([
      result.edges!.count[0],
      result.edges!.requiredCount[0],
      result.edges!.overflow[0]
    ]).toEqual([1, 2, 1]);
    expect(Array.from(result.edges!.lineOffsets)).toEqual([0, 2, 2]);
  }
});

it('path CPU operation publishes each optional port against the hand-enumerated path literals', async () => {
  const device = createNullWebGPUDevice();
  try {
    // The same independent derivation as the GPU case: [0,1,2] then [0], with node
    // offsets [0,3,4], edge offsets [0,2,2] and costs [5,0]. Port selection must not
    // alter mandatory IDs/counts; the executor exercises the operation's named outputs.
    for (const capacity of [2, 4])
      for (const selected of [undefined, 'lines', 'pathCosts', 'edges', 'edgeLines'] as const) {
        const operation = new CPUNetworkPathExtractionOperation(
          createPathOracleProps(capacity, selected)
        );
        const outputs = [
          'ids',
          'count',
          'required',
          'overflow',
          ...(selected === 'lines' ? ['lines'] : []),
          ...(selected === 'pathCosts' ? ['pathCosts'] : []),
          ...(selected === 'edges' || selected === 'edgeLines'
            ? ['edgeIdsOut', 'edgeCount', 'edgeRequired', 'edgeOverflow']
            : []),
          ...(selected === 'edgeLines' ? ['edgeLines'] : [])
        ];
        const values = await runOperationGraph(device, {
          inputs: PATH_INPUTS,
          operations: [operation],
          outputs
        });
        expect(Array.from(values['ids'])).toEqual(capacity === 2 ? [41, 17] : [41, 17, 90, 41]);
        expect(Array.from(values['count'])).toEqual([capacity]);
        expect(Array.from(values['required'])).toEqual([4]);
        expect(Array.from(values['overflow'])).toEqual([capacity === 2 ? 1 : 0]);
        if (selected === 'lines') expect(Array.from(values['lines'])).toEqual([0, 3, 4]);
        if (selected === 'pathCosts') expect(Array.from(values['pathCosts'])).toEqual([5, 0]);
        if (selected === 'edges' || selected === 'edgeLines') {
          expect(Array.from(values['edgeIdsOut'])).toEqual([501]);
          expect(Array.from(values['edgeCount'])).toEqual([1]);
          expect(Array.from(values['edgeRequired'])).toEqual([2]);
          expect(Array.from(values['edgeOverflow'])).toEqual([1]);
          if (selected === 'edgeLines') expect(Array.from(values['edgeLines'])).toEqual([0, 2, 2]);
        }
      }
  } finally {
    device.destroy();
  }
});
