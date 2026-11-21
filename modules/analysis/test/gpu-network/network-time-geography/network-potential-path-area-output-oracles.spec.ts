// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkPotentialPathArea} from '../../../src/gpu-network/network-time-geography/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('potential path area clears stale edge slots after the live CSR shrinks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withSummary of [true, false]) {
    const context = createOracleContext(device);
    const {graph, input, output, read, write} = context;
    const offsets = input('offsets', 'uint32', [0, 2, 2]);
    const result = {
      edgeSlack: output('slack', 'float32', 2),
      edgeInPrism: output('edges', 'uint32', 2),
      nodeInPrism: output('nodes', 'uint32', 2),
      summary: withSummary ? output('summary', 'uint32', 2) : undefined
    };
    graph.add(
      new GPUNetworkPotentialPathArea({
        topology: {
          offsets,
          neighbors: input('neighbors', 'uint32', [1, 1]),
          weights: input('weights', 'float32', [1, 1])
        },
        forwardCosts: input('forward', 'float32', [0, 1]),
        reverseCosts: input('reverse', 'float32', [1, 0]),
        budget: input('budget', 'float32', [2]),
        output: result
      })
    );
    try {
      // Two parallel unit arcs both have slack 2-(0+1+0)=1.
      context.run();
      expect(await read(result.edgeSlack)).toEqual([1, 1]);
      expect(await read(result.edgeInPrism)).toEqual([1, 1]);
      if (result.summary) expect(await read(result.summary)).toEqual([2, 2]);
      // The second slot is capacity only after this edit: it has no source row.
      write(offsets, [0, 1, 1]);
      context.run();
      expect(await read(result.edgeSlack)).toEqual([1, -Infinity]);
      expect(await read(result.edgeInPrism)).toEqual([1, 0]);
      expect(await read(result.nodeInPrism)).toEqual([1, 1]);
      if (result.summary) expect(await read(result.summary)).toEqual([2, 1]);
    } finally {
      context.destroy();
    }
  }
});

it('potential path area publishes exact time slack and overwrites it on a second submission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withSummary of [true, false]) {
    const context = createOracleContext(device);
    const {graph, input, output, read, write} = context;
    const budget = input('budget', 'float32', [12]);
    const result = {
      edgeSlack: output('slack', 'float32', 3),
      edgeInPrism: output('edges', 'uint32', 3),
      nodeInPrism: output('nodes', 'uint32', 3),
      summary: withSummary ? output('summary', 'uint32', 2) : undefined
    };
    // Parallel 0->1 arcs cost 3 and 7, then 1->2 cost 4. Anchor distances are supplied
    // directly. At budget 12, slack = [12-(2+3+4),12-(2+7+4),12-(5+4+0)].
    graph.add(
      new GPUNetworkPotentialPathArea({
        topology: {
          offsets: input('offsets', 'uint32', [0, 2, 3, 3]),
          neighbors: input('neighbors', 'uint32', [1, 1, 2]),
          weights: input('weights', 'float32', [3, 7, 4])
        },
        forwardCosts: input('forward', 'float32', [2, 5, 9]),
        reverseCosts: input('reverse', 'float32', [7, 4, 0]),
        budget,
        output: result
      })
    );
    try {
      context.run();
      expect(await read(result.edgeSlack)).toEqual([3, -1, 3]);
      expect(await read(result.edgeInPrism)).toEqual([1, 0, 1]);
      expect(await read(result.nodeInPrism)).toEqual([1, 1, 1]);
      if (result.summary) expect(await read(result.summary)).toEqual([3, 2]);
      write(budget, [8]);
      context.run();
      expect(await read(result.edgeSlack)).toEqual([-1, -5, -1]);
      expect(await read(result.edgeInPrism)).toEqual([0, 0, 0]);
      expect(await read(result.nodeInPrism)).toEqual([0, 0, 0]);
      if (result.summary) expect(await read(result.summary)).toEqual([0, 0]);
    } finally {
      context.destroy();
    }
  }
});

it('W3-01 and WGS-13 suppress incomplete trees and malformed CSR, then recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read, write} = context;
  const offsets = input('offsets', 'uint32', [0, 1, 2, 2]);
  const forwardComplete = input('forward-complete', 'uint32', [1]);
  const reverseComplete = input('reverse-complete', 'uint32', [1]);
  const result = {
    edgeSlack: output('slack', 'float32', 2),
    edgeInPrism: output('edges', 'uint32', 2),
    nodeInPrism: output('nodes', 'uint32', 3),
    summary: output('summary', 'uint32', 2),
    status: {
      complete: output('complete', 'uint32', 1),
      invalidInput: output('invalid', 'uint32', 1)
    }
  };
  graph.add(
    new GPUNetworkPotentialPathArea({
      topology: {
        offsets,
        neighbors: input('neighbors', 'uint32', [1, 2]),
        weights: input('weights', 'float32', [1, 1])
      },
      forwardCosts: input('forward', 'float32', [0, 1, 2]),
      reverseCosts: input('reverse', 'float32', [2, 1, 0]),
      forwardComplete,
      reverseComplete,
      budget: input('budget', 'float32', [2]),
      output: result
    })
  );
  try {
    for (const certificate of [forwardComplete, reverseComplete]) {
      write(certificate, [0]);
      context.run();
      expect(await read(result.summary)).toEqual([0, 0]);
      expect(await read(result.edgeInPrism)).toEqual([0, 0]);
      expect(await read(result.nodeInPrism)).toEqual([0, 0, 0]);
      expect(await read(result.edgeSlack)).toEqual([-Infinity, -Infinity]);
      expect(await read(result.status.complete)).toEqual([0]);
      write(certificate, [1]);
    }
    write(offsets, [0, 2, 1, 2]);
    context.run();
    expect(await read(result.status.invalidInput)).toEqual([1]);
    expect(await read(result.edgeInPrism)).toEqual([0, 0]);
    expect(await read(result.edgeSlack)).toEqual([-Infinity, -Infinity]);
    write(offsets, [0, 1, 2, 2]);
    context.run();
    // Each unit edge has slack 2 - 2 = 0; all three nodes fit.
    expect(await read(result.status.complete)).toEqual([1]);
    expect(await read(result.status.invalidInput)).toEqual([0]);
    expect(await read(result.summary)).toEqual([3, 2]);
    expect(await read(result.edgeSlack)).toEqual([0, 0]);
  } finally {
    context.destroy();
  }
});
