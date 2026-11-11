// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUAlternativeRoutesOperation} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {
  GPUAlternativeRoutesOperation,
  type AlternativeRoutesOperationProps
} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getAlternativeRoutesInput} from './alternative-routes-operation-test-utils';

for (const trivial of [false, true])
  it(`alternative route twins compare every output (trivial/invalid endpoints: ${trivial})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getAlternativeRoutesInput();
    if (trivial) {
      input.sources = Uint32Array.of(0, 99);
      input.targets = Uint32Array.of(0, 4);
    }
    const props: AlternativeRoutesOperationProps = {
      nodeCount: 5,
      queryCount: 2,
      routeCount: 4,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      sources: 'sources',
      targets: 'targets',
      maximumExpansions: 1024,
      maximumFrontierSize: 64,
      output: {
        vertices: 'vertices',
        edges: 'edges',
        lengths: 'lengths',
        costs: 'costs',
        counts: 'counts',
        complete: 'complete'
      }
    };
    const inputs = {
      offsets: {format: 'uint32' as const, values: input.topology.offsets},
      neighbors: {format: 'uint32' as const, values: input.topology.neighbors},
      weights: {format: 'float32' as const, values: input.topology.weights},
      sources: {format: 'uint32' as const, values: input.sources},
      targets: {format: 'uint32' as const, values: input.targets}
    };
    const outputs = ['vertices', 'edges', 'lengths', 'costs', 'counts', 'complete'];
    const gpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new GPUAlternativeRoutesOperation(props)],
      outputs
    });
    const cpuRows = await runOperationGraph(device, {
      inputs,
      operations: [new CPUAlternativeRoutesOperation(props)],
      outputs
    });
    // Exact binary fractions make all costs exact; all path IDs, counts, lengths and padding are exact.
    // Numeric equality deliberately ignores the sign of zero.
    for (const output of outputs)
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    expect(Array.from(cpuRows['counts'])).toEqual(trivial ? [1, 0] : [4, 2]);
  });
