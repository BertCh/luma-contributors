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
import {expectTraversalPadding} from './traversal-enumeration-assertions';
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
    for (const rows of [gpuRows, cpuRows]) {
      expectTraversalPadding(
        {
          counts: rows['counts'],
          lengths: rows['lengths'],
          costs: rows['costs'],
          vertices: rows['vertices'],
          edges: rows['edges']
        },
        2,
        4,
        5
      );
    }
    // Exact binary fractions make all costs exact; all path IDs, counts, lengths and padding are exact.
    // Numeric equality deliberately ignores the sign of zero.
    for (const output of outputs)
      for (let index = 0; index < cpuRows[output].length; index++)
        expect(gpuRows[output][index] === cpuRows[output][index]).toBe(true);
    expect(Array.from(cpuRows['counts'])).toEqual(trivial ? [1, 0] : [4, 2]);
  });

it('alternative enumeration and exhausted prefixes match independent simple edge-slot DFS', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {enumerateAlternativeRoutes, getTraversalEnumerationInput, readTraversalRoutes} =
    await import('./alternative-routes-operation-test-utils');
  const input = getTraversalEnumerationInput();
  const expected = enumerateAlternativeRoutes(input, 8);
  const all = readTraversalRoutes(expected, 6);
  expect(Array.from(expected.counts)).toEqual([6, 0]);
  for (const [maximumFrontierSize, maximumExpansions] of [
    [64, 1024],
    [64, 4],
    [4, 1024]
  ]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: input.topology.offsets},
        neighbors: {format: 'uint32', values: input.topology.neighbors},
        weights: {format: 'float32', values: input.topology.weights},
        sources: {format: 'uint32', values: input.sources},
        targets: {format: 'uint32', values: input.targets}
      },
      operations: [
        new GPUAlternativeRoutesOperation({
          nodeCount: 5,
          queryCount: 2,
          routeCount: 8,
          maximumFrontierSize,
          maximumExpansions,
          topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          sources: 'sources',
          targets: 'targets',
          output: {
            vertices: 'vertices',
            edges: 'edges',
            lengths: 'lengths',
            costs: 'costs',
            counts: 'counts',
            complete: 'complete'
          }
        })
      ],
      outputs: ['vertices', 'edges', 'lengths', 'costs', 'counts', 'complete']
    });
    const complete = maximumFrontierSize === 64 && maximumExpansions === 1024;
    const count = rows['counts'][0];
    // This operator promises global (cost, lexicographic edge slots) order even after a stop.
    expect(count).toBe(complete ? 6 : 1);
    expect(
      readTraversalRoutes(
        {edges: rows['edges'], lengths: rows['lengths'], costs: rows['costs']},
        count
      )
    ).toEqual(all.slice(0, count));
    for (const [routeIndex, route] of all.slice(0, count).entries()) {
      expect(
        Array.from(rows['vertices'].slice(routeIndex * 5, routeIndex * 5 + route.edges.length + 1))
      ).toEqual([0, ...route.edges.map(edge => input.topology.neighbors[edge])]);
    }
    expectTraversalPadding(
      {
        counts: rows['counts'],
        lengths: rows['lengths'],
        costs: rows['costs'],
        vertices: rows['vertices'],
        edges: rows['edges']
      },
      2,
      8,
      5
    );
    expect(Array.from(rows['counts'])).toEqual([count, 0]);
    expect(Array.from(rows['complete'])).toEqual([Number(complete), 1]);
  }
});
