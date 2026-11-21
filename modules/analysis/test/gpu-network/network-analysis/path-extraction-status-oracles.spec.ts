// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkPathExtractionOperation} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';
import {CPUNetworkPathExtractionOperation} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {poisonOperationOutputs} from '../adjacency-matrix/network-output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {PATH_INPUTS, createPathOracleProps} from './network-path-extraction-output-oracles.fixture';

for (const maximumPathLength of [1, 4]) {
  it(`pins all poisoned compact flags from path cardinalities (${maximumPathLength})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Target 2 in 0->1->2 needs 3 nodes/2 edges; target 0 needs 1 node/0
    // edges. Bound 1 cannot inspect target 2's predecessor 1; only root remains.
    // Bound 4 finds all four node rows and two edge rows, truncated to 2/1.
    for (const Operation of [
      GPUNetworkPathExtractionOperation,
      CPUNetworkPathExtractionOperation
    ]) {
      const props = createPathOracleProps(2, 'edges');
      props.maximumPathLength = maximumPathLength;
      props.output.status.candidateOverflow = 'candidate';
      props.edges!.output.status.candidateOverflow = 'edgeCandidate';
      let operation = new Operation(props);
      if (operation instanceof GPUNetworkPathExtractionOperation)
        operation = poisonOperationOutputs(operation);
      const names = [
        'count',
        'required',
        'overflow',
        'candidate',
        'edgeCount',
        'edgeRequired',
        'edgeOverflow',
        'edgeCandidate'
      ];
      const rows = await runOperationGraph(device, {
        inputs: PATH_INPUTS,
        operations: [operation],
        outputs: names
      });
      expect(names.map(name => rows[name][0])).toEqual(
        maximumPathLength === 1 ? [1, 1, 0, 1, 0, 0, 0, 1] : [2, 4, 1, 0, 1, 2, 1, 0]
      );
    }
  });
}
