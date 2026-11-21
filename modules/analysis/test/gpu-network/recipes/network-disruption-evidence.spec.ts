// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addNetworkDisruptionRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix';
import {getGPUNetworkAccessibilityParameterValues} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('noded disruption propagates source build loss and recovers without losing live bounds (B4)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read, write} = context;
  try {
    const topology = {
      offsets: input('offsets', 'uint32', [0, 1, 2, 2]),
      neighbors: input('neighbors', 'uint32', [1, 0]),
      weights: input('weights', 'float32', [1, 1])
    };
    const sourceLoss = input('source-loss', 'uint32', [1]);
    const difference = output('cost-differences', 'float32', 3);
    const scoreDifference = output('score-difference', 'float32');
    const result = addNetworkDisruptionRecipe(graph, {
      search: {
        topology,
        nodedTopology: {
          topology,
          nodes: {status: {count: input('live-count', 'uint32', [2])}},
          pieces: {
            status: {
              count: input('piece-count', 'uint32', [1]),
              overflow: input('piece-overflow', 'uint32', [0])
            }
          },
          status: {overflow: sourceLoss, uncertainCount: input('uncertain', 'uint32', [0])}
        },
        startNodes: input('starts', 'uint32', [0])
      },
      rowCount: 1,
      opportunityWeights: input('opportunities', 'float32', [1, 1, 1]),
      parameters: input('parameters', 'float32', [1, 0, 0, 1]),
      outputs: {costDifferences: difference, accessibilityDifferences: scoreDifference}
    });
    const evidence = output('evidence', 'uint32', 5);
    result.status.stages.forEach((stage, index) =>
      graph.add(
        createWGSLKernelNode(graph, {
          id: `copy-noded-evidence-${index}`,
          operation: 'noded-disruption-evidence',
          invocationCount: 1,
          bindings: [
            {name: 'source', view: stage.status.complete!, type: 'u32', access: 'read'},
            {name: 'destination', view: evidence, type: 'u32', access: 'read_write'}
          ],
          body: `destination[destinationOffset + ${index}u] = source[sourceOffset];`
        })
      )
    );
    for (const loss of [1, 0]) {
      write(sourceLoss, [loss]);
      context.run();
      expect(await read(evidence)).toEqual(loss ? [0, 0, 0, 0, 0] : [1, 1, 1, 1, 1]);
      expect(await read(difference)).toEqual(loss ? [NaN, NaN, NaN] : [0, 0, NaN]);
      expect(await read(scoreDifference)).toEqual(loss ? [NaN] : [0]);
    }
  } finally {
    context.destroy();
  }
});

it('an unchanged edge beyond the common horizon is not newly unreachable (B5)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, read} = context;
  try {
    const topology = {
      offsets: input('offsets', 'uint32', [0, 1, 1]),
      neighbors: input('neighbors', 'uint32', [1]),
      weights: input('weights', 'float32', [7])
    };
    const startNodes = input('starts', 'uint32', [0]);
    const baselineCosts = output('baseline-costs', 'float32', 2);
    const converged = output('baseline-converged', 'uint32');
    const baselineLimit = input('baseline-limit', 'float32', [9]);
    graph.add(
      new GPUNetworkCostMatrix({
        id: 'retained-unchanged',
        topology,
        startNodes,
        costLimit: baselineLimit,
        output: {costs: baselineCosts, status: {converged}}
      })
    );
    const newlyUnreachable = output('newly-unreachable', 'uint32', 2);
    const costDifferences = output('cost-differences', 'float32', 2);
    addNetworkDisruptionRecipe(graph, {
      search: {topology, startNodes, costLimit: input('scenario-limit', 'float32', [4])},
      baseline: {costs: baselineCosts, costLimit: baselineLimit, status: {converged}},
      rowCount: 1,
      opportunityWeights: input('opportunities', 'float32', [1, 1]),
      parameters: input('parameters', 'float32', [7, 0, 0, 1]),
      outputs: {newlyUnreachable, costDifferences}
    });
    context.run();
    expect(await read(newlyUnreachable)).toEqual([0, 0]);
    expect(await read(costDifferences)).toEqual([0, NaN]);
  } finally {
    context.destroy();
  }
});

it.each([
  false,
  true
])('disruption suppresses an insufficient horizon and recovers (retained=%s)', async retained => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `disruption-evidence-${retained}`);
  try {
    const horizon = fixture.output('horizon', 'float32', 1);
    const retainedHorizon = fixture.output('retained-horizon', 'float32', 1);
    const search = {
      // 0->1=1, 1->2=2, 0->2=5. Unit opportunities include the origin.
      topology: {
        offsets: fixture.input('offsets', Uint32Array.of(0, 2, 3, 3), 'uint32', 4),
        neighbors: fixture.input('neighbors', Uint32Array.of(1, 2, 2), 'uint32', 3),
        weights: fixture.input('weights', Float32Array.of(1, 5, 2), 'float32', 3)
      },
      startNodes: fixture.input('starts', Uint32Array.of(0), 'uint32', 1),
      costLimit: horizon.view,
      maximumIterations: 4
    };
    const baselineCosts = fixture.output('baseline-costs', 'float32', 3);
    const baselineConverged = fixture.output('baseline-converged', 'uint32', 1);
    if (retained)
      fixture.graph.add(
        new GPUNetworkCostMatrix({
          ...search,
          costLimit: retainedHorizon.view,
          id: 'retained-baseline',
          output: {costs: baselineCosts.view, status: {converged: baselineConverged.view}}
        })
      );
    const baselineScore = fixture.output('baseline-score', 'float32', 1);
    const scenarioScore = fixture.output('scenario-score', 'float32', 1);
    const delta = fixture.output('delta', 'float32', 1);
    const costDifferences = fixture.output('cost-differences', 'float32', 3);
    const result = addNetworkDisruptionRecipe(fixture.graph, {
      search,
      rowCount: 1,
      edgeMask: fixture.input('mask', Uint32Array.of(1, 1, 0), 'uint32', 3),
      opportunityWeights: fixture.input('opportunities', Float32Array.of(1, 1, 1), 'float32', 3),
      parameters: fixture.input(
        'parameters',
        getGPUNetworkAccessibilityParameterValues({threshold: 3}),
        'float32',
        4
      ),
      baseline: retained
        ? {
            costs: baselineCosts.view,
            costLimit: retainedHorizon.view,
            status: {converged: baselineConverged.view}
          }
        : undefined,
      outputs: {
        baselineAccessibility: baselineScore.view,
        scenarioAccessibility: scenarioScore.view,
        accessibilityDifferences: delta.view,
        costDifferences: costDifferences.view
      }
    });
    const evidence = fixture.output('evidence', 'uint32', 5);
    const sources = [
      result.status.stages[0].status.converged!,
      result.status.stages[1].status.converged!,
      result.status.baselineAccessibilityComplete,
      result.status.scenarioAccessibilityComplete,
      result.status.stages[4].status.complete!
    ];
    sources.forEach((source, index) =>
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: `read-evidence-${index}`,
          operation: 'disruption-evidence',
          invocationCount: 1,
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {name: 'destination', view: evidence.view, type: 'u32', access: 'read_write'}
          ],
          body: `destination[destinationOffset + ${index}u + index] = source[sourceOffset + index];`
        })
      )
    );
    for (const [baselineLimit, limit] of retained
      ? [
          [1, 3],
          [3, 1],
          [3, 3]
        ]
      : [
          [1, 1],
          [3, 3]
        ]) {
      horizon.buffer.write(Float32Array.of(limit));
      retainedHorizon.buffer.write(Float32Array.of(baselineLimit));
      fixture.run();
      expect(await fixture.readUint32(evidence, 5)).toEqual([
        1,
        1,
        baselineLimit >= 3 ? 1 : 0,
        limit >= 3 ? 1 : 0,
        baselineLimit >= 3 && limit >= 3 ? 1 : 0
      ]);
      // Hand count: horizon 1 reaches {0,1} in both; horizon 3 adds vertex 2 only in baseline.
      expect(await fixture.readFloat32(baselineScore, 1)).toEqual([baselineLimit === 1 ? 2 : 3]);
      expect(await fixture.readFloat32(scenarioScore, 1)).toEqual([2]);
      expect(await fixture.readFloat32(delta, 1)).toEqual([
        baselineLimit >= 3 && limit >= 3 ? -1 : NaN
      ]);
      // The two reached pairs remain independently valid even when the score horizon is insufficient.
      expect(await fixture.readFloat32(costDifferences, 3)).toEqual([0, 0, NaN]);
    }
  } finally {
    fixture.destroy();
  }
});
