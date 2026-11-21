// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUKernelFlowSmoothing} from '../../../src/gpu-network/flow-aggregation/gpu-kernel-flow-smoothing';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('CPL-09 refuses bare or conflicting counts and accepts explicit completeness', () => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {graph, input, output} = context;
  const count = input('count', 'uint32', [1]);
  const props = {
    positions: input('positions', 'float32x2', [0, 0, 1, 0]),
    locationWeights: input('locations', 'float32', [1, 1]),
    parameters: input('parameters', 'float32', [0, 0.5, 0, 0]),
    flows: {
      kind: 'pairs' as const,
      originZoneIds: input('origin', 'uint32', [0]),
      destinationZoneIds: input('destination', 'uint32', [1]),
      weights: input('weights', 'float32', [2]),
      count
    },
    output: {
      smoothed: output('smoothed', 'float32', 4),
      status: {complete: output('complete', 'uint32', 1)}
    }
  };
  try {
    expect(() => new GPUKernelFlowSmoothing(props)).toThrow(
      /requires producer status or completeness/
    );
    expect(
      () =>
        new GPUKernelFlowSmoothing({
          ...props,
          flows: {
            ...props.flows,
            completeness: 'known-complete',
            // @ts-expect-error Untyped callers can still supply the retired key.
            validPairCount: count
          }
        })
    ).toThrow(/flows.validPairCount was removed; use flows.count/);
    expect(() =>
      new GPUKernelFlowSmoothing({
        ...props,
        flows: {...props.flows, completeness: 'known-complete'}
      }).getCommandNodes(graph)
    ).not.toThrow();
    const {count: pairCount, ...pairRows} = props.flows;
    expect(() =>
      new GPUKernelFlowSmoothing({
        ...props,
        id: 'producer-smoothing',
        flows: {
          ...pairRows,
          status: {
            count: pairCount,
            overflow: input('overflow', 'uint32', [0]),
            candidateOverflow: input('candidate', 'uint32', [0])
          }
        }
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    context.destroy();
    device.destroy();
  }
});
