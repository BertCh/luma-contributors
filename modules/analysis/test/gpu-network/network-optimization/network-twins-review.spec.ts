// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkPointRoutingOperation} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing-operation';
import {CPUNetworkPointRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {GPUNetworkLineGraph} from '../../../src/gpu-network/network-line-graph/gpu-network-line-graph';
import {getGPUNetworkLineGraphParameterValues} from '../../../src/gpu-network/network-line-graph/network-line-graph-parameters';
import {prepareGPUNetworkPointRoutingTurns} from '../../../src/gpu-network/network-optimization/prepare-point-routing-turns';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';
import {getRoutingReviewFixtures, traceRoutingReviewPath} from './network-twins-review-fixtures';

for (const mode of ['reference', 'scalable'] as const) {
  for (const fixture of getRoutingReviewFixtures()) {
    it(`${fixture.name}: mixed-executor ${mode}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {input} = fixture;
      for (const Operation of [GPUNetworkPointRoutingOperation, CPUNetworkPointRoutingOperation]) {
        const rows = await runOperationGraph(device, {
          inputs: {
            edgeCosts: {format: 'float32', values: input.costs},
            turns: {format: 'uint32x2', values: input.turns},
            penalties: {format: 'float32', values: input.turnPenalties!},
            queries: {format: 'uint32x2', values: input.queries},
            fractions: {format: 'float32x2', values: input.fractions}
          },
          operations: [
            new Operation({
              mode,
              maximumRelaxations: fixture.maximumRelaxations ?? 8,
              edgeCount: input.costs.length,
              queryCount: fixture.expectedCosts.length,
              costs: 'edgeCosts',
              turns: 'turns',
              turnPenalties: 'penalties',
              queries: 'queries',
              fractions: 'fractions',
              output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
            })
          ],
          outputs: ['costs', 'status', 'parents']
        });
        expect(Array.from(rows['costs'])).toEqual(fixture.expectedCosts);
        expect(Array.from(rows['status'])).toEqual(fixture.expectedStatus);
        if (fixture.maximumRelaxations === 0)
          expect(Array.from(rows['parents'])).toEqual(
            new Array(input.costs.length * fixture.expectedCosts.length).fill(0xffffffff)
          );
        for (const [query, path] of (fixture.expectedPaths ?? []).entries()) {
          expect(
            traceRoutingReviewPath(
              rows['parents'],
              input.queries[query * 2],
              input.queries[query * 2 + 1],
              input.costs.length,
              query
            )
          ).toEqual(path);
        }
      }
    });
  }
}

it('R1 explicit reference flow rejects inconsistent CSR for zero demand and accepts aligned CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const valid of [false, true]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        edges: {format: 'uint32x2', values: Uint32Array.of(0, 1, 1, 2)},
        capacities: {format: 'uint32', values: Uint32Array.of(1, 1)},
        costs: {format: 'uint32', values: Uint32Array.of(0, 0)},
        parameters: {format: 'uint32', values: Uint32Array.of(0, 2, 0)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(valid ? 1 : 2, 2)}
      },
      operations: [
        new GPUNetworkFlowOperation({
          solver: 'reference',
          nodeCount: 3,
          edgeCount: 2,
          edges: 'edges',
          capacities: 'capacities',
          costs: 'costs',
          parameters: 'parameters',
          topology: {offsets: 'offsets', neighbors: 'neighbors'},
          output: {
            flow: 'flow',
            cut: 'cut',
            summary: 'summary',
            exactCost: 'exact',
            status: {invalidCount: 'invalid'}
          }
        })
      ],
      outputs: ['flow', 'cut', 'summary', 'exact', 'invalid']
    });
    expect(rows['invalid']).toEqual(Uint32Array.of(Number(!valid)));
    if (!valid) {
      expect(rows['summary']).toEqual(new Uint32Array(8));
      expect(rows['flow']).toEqual(new Uint32Array(2));
      expect(rows['cut']).toEqual(new Uint32Array(3));
      expect(rows['exact']).toEqual(new Uint32Array(2));
    } else expect(rows['summary'][0]).toBe(1);
  }
});

for (const withBans of [false, true]) {
  it(`R7 line-graph penalties survive large base costs through preparation and both routing modes (bans ${withBans})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const costs = Float32Array.of(0, 0, 2 ** 28);
    const output = {
      topology: {
        offsets: harness.output('uint32', 4),
        neighbors: harness.output('uint32', 3),
        weights: harness.output('float32', 3)
      },
      turnPenalties: harness.output('float32', 3),
      status: {overflow: harness.output('uint32', 1)}
    };
    const read = await harness.run(
      new GPUNetworkLineGraph({
        topology: {
          offsets: harness.input('uint32', Uint32Array.of(0, 1, 2, 3, 3)),
          neighbors: harness.input('uint32', Uint32Array.of(2, 2, 3)),
          weights: harness.input('float32', costs),
          nodePositions: harness.input('float32x2', Float32Array.of(-1, 0, 1, 0, 0, 0, 0, 1))
        },
        parameters: harness.input(
          'float32',
          getGPUNetworkLineGraphParameterValues({
            leftTurnCost: 7,
            rightTurnCost: 1,
            bannedTurnCount: Number(withBans)
          })
        ),
        bannedTurns: withBans ? harness.input('uint32', Uint32Array.of(0, 0)) : undefined,
        output
      })
    );
    expect(read(output.status.overflow)).toEqual([0]);
    expect(read(output.turnPenalties)).toEqual([7, 1, -1]);
    expect(read(output.topology.weights)).toEqual([2 ** 28, 2 ** 28, -1]);
    const prepared = prepareGPUNetworkPointRoutingTurns({
      costs,
      overflow: read(output.status.overflow)[0],
      offsets: Uint32Array.from(read(output.topology.offsets)),
      neighbors: Uint32Array.from(read(output.topology.neighbors)),
      weights: Float32Array.from(read(output.topology.weights)),
      turnPenalties: Float32Array.from(read(output.turnPenalties))
    });
    expect(prepared.turnPenalties).toEqual(Float32Array.of(7, 1));
    for (const mode of ['reference', 'scalable'] as const) {
      const rows = await runOperationGraph(device, {
        inputs: {
          costs: {format: 'float32', values: costs},
          turns: {format: 'uint32x2', values: prepared.turns},
          penalties: {format: 'float32', values: prepared.turnPenalties},
          queries: {format: 'uint32x2', values: Uint32Array.of(0, 2, 1, 2)},
          fractions: {format: 'float32x2', values: Float32Array.of(1, 0, 1, 0)}
        },
        operations: [
          new GPUNetworkPointRoutingOperation({
            mode,
            edgeCount: 3,
            queryCount: 2,
            costs: 'costs',
            turns: 'turns',
            turnPenalties: 'penalties',
            queries: 'queries',
            fractions: 'fractions',
            output: {costs: 'output', status: 'status', predecessorEdges: 'parents'}
          })
        ],
        outputs: ['output', 'status']
      });
      expect(rows['output']).toEqual(Float32Array.of(7, 1));
      expect(rows['status']).toEqual(Uint32Array.of(0, 0));
    }
  });
}
