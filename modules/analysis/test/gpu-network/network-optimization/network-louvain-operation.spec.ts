// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkLouvainOperation} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
import {
  GPUNetworkLouvainOperation,
  type NetworkLouvainOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-louvain-operation';
import {runOperationGraph, type OperationGraphRows} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// At gamma=1, Q subtracts two normalized terms in [0,1]. Near Q=0, relative error
// is undefined: allow eight float32 spacings at unit magnitude for their arithmetic,
// independently of the small residual observed on any one adapter. Nonzero Q stays relative.
const OBJECTIVE_CANCELLATION_TOLERANCE = 8 * 2 ** -23;

for (const scene of [
  'weighted',
  'directed-seeded',
  'coarsening',
  'zero',
  'invalid',
  'budget'
] as const) {
  it(`Louvain twins compare all labels, trace rows and summary words (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const nodeCount = scene === 'coarsening' ? 4 : 6;
    const weights = new Float32Array(nodeCount ** 2);
    if (scene === 'coarsening') {
      weights.set([0, 2, 1.25, 1.25, 2, 0, 1.25, 1.25, 1.25, 1.25, 0, 2, 1.25, 1.25, 2, 0]);
    } else if (scene !== 'zero') {
      for (let source = 0; source < nodeCount; source++)
        for (let target = 0; target < nodeCount; target++)
          if (source !== target && Math.floor(source / 3) === Math.floor(target / 3))
            weights[source * nodeCount + target] = scene === 'directed-seeded' ? 2 + source : 3;
      weights[2 * nodeCount + 3] = 0.25;
      if (scene !== 'directed-seeded') weights[3 * nodeCount + 2] = 0.25;
      weights[0] = 1;
    }
    if (scene === 'invalid') weights[1] = -1;
    const props: NetworkLouvainOperationProps = {
      id: 'louvain',
      nodeCount,
      directed: scene === 'directed-seeded',
      seed: scene === 'directed-seeded' ? 0x9e3779b9 : undefined,
      maximumLevels: scene === 'budget' ? 1 : 8,
      maximumIterations: scene === 'budget' ? 1 : 16,
      weights: 'weights',
      parameters: 'parameters',
      output: {communities: 'communities', objectiveTrace: 'trace', summary: 'summary'}
    };
    const inputs = {
      weights: {format: 'float32' as const, values: weights},
      parameters: {
        format: 'float32' as const,
        values: Float32Array.of(scene === 'directed-seeded' ? 0.5 : 1, 1e-5)
      }
    };
    const outputs = ['communities', 'trace', 'summary'];
    const results: Record<string, Record<string, OperationGraphRows>> = {};
    for (const location of ['gpu', 'cpu'])
      results[location] = await runOperationGraph(device, {
        inputs,
        operations: [
          location === 'gpu'
            ? new GPUNetworkLouvainOperation(props)
            : new CPUNetworkLouvainOperation(props)
        ],
        outputs
      });
    for (const port of ['communities', 'summary'])
      expect(Array.from(results['gpu'][port])).toEqual(Array.from(results['cpu'][port]));
    expect(results['gpu']['trace'].length).toBe(results['cpu']['trace'].length);
    for (let row = 0; row < results['cpu']['trace'].length; row++) {
      const expected = results['cpu']['trace'][row];
      if (Number.isNaN(expected)) expect(Number.isNaN(results['gpu']['trace'][row])).toBe(true);
      // Coarsening's Q=1-1 can leave a small finite GPU cancellation residual.
      else if (scene === 'coarsening' && expected === 0)
        expect(Math.abs(results['gpu']['trace'][row])).toBeLessThanOrEqual(
          OBJECTIVE_CANCELLATION_TOLERANCE
        );
      // Other entries keep relative 2e-5, including exact zero in the empty-graph branch.
      else
        expect(Math.abs(results['gpu']['trace'][row] - expected)).toBeLessThanOrEqual(
          2e-5 * Math.abs(expected)
        );
    }
    if (scene === 'weighted' || scene === 'directed-seeded') {
      const communities = results['cpu']['communities'];
      expect(communities[0]).toBe(communities[2]);
      expect(communities[3]).toBe(communities[5]);
      expect(communities[0]).not.toBe(communities[3]);
      expect(results['cpu']['summary'][1]).toBe(1);
    }
    if (scene === 'coarsening') {
      // Strengths 9/2 and total weight 18 give Q=-1/18 for the first-level pairs.
      // Coarsening permits merging the pairs: all internal weight gives Q=1-1=0.
      for (const location of ['gpu', 'cpu']) {
        expect(Array.from(results[location]['communities'])).toEqual([3, 3, 3, 3]);
        expect(Array.from(results[location]['summary'])).toEqual([1, 1, 3, 5]);
      }
      // The CPU oracle is exactly zero; the GPU evaluates normalized terms in float32.
      expect(results['cpu']['trace'][5] === 0).toBe(true);
      expect(Math.abs(results['gpu']['trace'][5])).toBeLessThanOrEqual(
        OBJECTIVE_CANCELLATION_TOLERANCE
      );
    }
  });
}
