// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix';
import {addNetworkDisruptionRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUNetworkAccessibility} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it.each([
  false,
  true
])('disruption retains routing identities and certifies both scores (retained=%s)', retained => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `disruption-${serial++}`, format, length);
  const costLimit = view('float32', 1);
  const baselineLimit = view('float32', 1);
  const baseline = {
    costs: view('float32', 3),
    costLimit: baselineLimit,
    status: {converged: view('uint32', 1)}
  };
  try {
    const result = addNetworkDisruptionRecipe(graph, {
      search: {
        topology: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 3),
          weights: view('float32', 3)
        },
        startNodes: view('uint32', 1),
        costLimit
      },
      rowCount: 1,
      opportunityWeights: view('float32', 3),
      parameters: view('float32', 4),
      baseline: retained ? baseline : undefined
    });
    expect(result.status.stages.map(stage => stage.stage)).toEqual([
      'baseline-routing',
      'scenario-routing',
      'baseline-accessibility',
      'scenario-accessibility',
      'accessibility'
    ]);
    if (retained) {
      expect(result.outputs.baselineCosts).toBe(baseline.costs);
      expect(result.status.stages[0].status.converged).toBe(baseline.status.converged);
    }
    const scorers = result.contributors.filter(
      (contributor): contributor is GPUNetworkAccessibility =>
        contributor instanceof GPUNetworkAccessibility
    );
    expect(scorers).toHaveLength(2);
    for (const [index, scorer] of scorers.entries()) {
      expect(scorer.props.costsComplete).toBe(result.status.stages[index].status.converged);
      expect(scorer.props.costLimit).toBe(retained && index === 0 ? baselineLimit : costLimit);
      expect(result.status.stages[index + 2].status.complete).toBe(scorer.props.status?.complete);
      expect(scorer.props.status?.complete).toBe(
        index === 0
          ? result.status.baselineAccessibilityComplete
          : result.status.scenarioAccessibilityComplete
      );
      expect(scorer.props.status?.complete).toBeDefined();
    }
    const delta = getWGSLKernelNodeRecord(graph, 'network-disruption-recipe-accessibility-delta')!;
    expect(delta.bindings.find(binding => binding.name === 'beforeComplete')?.view).toBe(
      result.status.baselineAccessibilityComplete
    );
    expect(delta.bindings.find(binding => binding.name === 'afterComplete')?.view).toBe(
      result.status.scenarioAccessibilityComplete
    );
    expect(result.status.baselineAccessibilityComplete).not.toBe(
      result.status.scenarioAccessibilityComplete
    );
    const source = getParsedKernelSources((graph as unknown as {nodes: unknown[]}).nodes).get(
      'network-disruption-recipe-accessibility-delta'
    );
    expect(source).toContain(
      'beforeComplete[beforeCompleteOffset] != 0u && afterComplete[afterCompleteOffset] != 0u'
    );
    const joint = getWGSLKernelNodeRecord(
      graph,
      'network-disruption-recipe-accessibility-complete'
    )!;
    expect(joint).toBeDefined();
    expect(result.status.accessibilityComplete).toBe(result.status.stages[4].status.complete);
    expect(joint.bindings.find(binding => binding.name === 'complete')?.view).toBe(
      result.status.stages[4].status.complete
    );
    // Reuse one output word: either failed score withholds the joint certificate, then recovers.
    const complete = Uint32Array.of(99);
    for (const [before, after, expected] of [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 1],
      [0, 0, 0]
    ]) {
      const storage: Record<string, Uint32Array> = {
        beforeComplete: Uint32Array.of(before),
        afterComplete: Uint32Array.of(after),
        complete
      };
      const bindings = Object.fromEntries(
        joint.bindings.map((binding, index) => [index, storage[binding.name]])
      );
      new WgslExec(
        new WgslParser().parse(joint.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
      ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: joint.constants});
      expect(Array.from(complete)).toEqual([expected]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('carries noded cardinality, replacement-weight identities and source evidence through disruption (retained=%s, B4)', retained => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `noded-recipe-${serial++}`, format, length);
  const topology = {
    offsets: view('uint32', 4),
    neighbors: view('uint32', 2),
    weights: view('float32', 2)
  };
  const count = view('uint32', 1);
  const result = addNetworkDisruptionRecipe(graph, {
    search: {
      topology,
      nodedTopology: {
        topology,
        nodes: {status: {count}},
        pieces: {status: {count: view('uint32', 1), overflow: view('uint32', 1)}},
        status: {overflow: view('uint32', 1), uncertainCount: view('uint32', 1)}
      },
      startNodes: view('uint32', 1)
    },
    rowCount: 1,
    opportunityWeights: view('float32', 3),
    parameters: view('float32', 4),
    baseline: retained
      ? {
          costs: view('float32', 3),
          status: {converged: view('uint32', 1), sourceComplete: view('uint32', 1)}
        }
      : undefined
  });
  const matrices = result.contributors.filter(
    (contributor): contributor is GPUNetworkCostMatrix =>
      contributor instanceof GPUNetworkCostMatrix
  );
  expect(matrices).toHaveLength(retained ? 1 : 2);
  for (const [index, matrix] of matrices.entries()) {
    expect(matrix.props.nodedTopology?.nodes.status.count).toBe(count);
    expect(matrix.props.nodedTopology?.topology.weights).toBe(matrix.props.topology.weights);
    expect(matrix.props.output.status?.sourceComplete).toBeDefined();
    expect(result.status.stages[index].status.complete).toBeDefined();
  }
  const scorers = result.contributors.filter(
    (contributor): contributor is GPUNetworkAccessibility =>
      contributor instanceof GPUNetworkAccessibility
  );
  for (const [index, scorer] of scorers.entries())
    expect(scorer.props.costsComplete).toBe(result.status.stages[index].status.complete);
  for (const name of ['baseline', 'scenario']) {
    const record = getWGSLKernelNodeRecord(
      graph,
      `network-disruption-recipe-${name}-routing-complete`
    )!;
    for (const [converged, sourceComplete, expected] of [
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 1]
    ]) {
      const storage: Record<string, Uint32Array> = {
        converged: Uint32Array.of(converged),
        sourceComplete: Uint32Array.of(sourceComplete),
        complete: Uint32Array.of(99)
      };
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => [index, storage[binding.name]])
      );
      new WgslExec(
        new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
      ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
      expect(Array.from(storage['complete'])).toEqual([expected]);
    }
  }
});

it('compares newly-unreachable pairs within the common horizon and preserves finite differences (B5)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `horizon-recipe-${serial++}`, format, length);
  addNetworkDisruptionRecipe(graph, {
    search: {
      topology: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 1),
        weights: view('float32', 1)
      },
      startNodes: view('uint32', 1),
      costLimit: view('float32', 1)
    },
    baseline: {
      costs: view('float32', 2),
      costLimit: view('float32', 1),
      status: {converged: view('uint32', 1)}
    },
    rowCount: 1,
    opportunityWeights: view('float32', 2),
    parameters: view('float32', 4)
  });
  const record = getWGSLKernelNodeRecord(graph, 'network-disruption-recipe-differences')!;
  const run = (first: number, second: number, beforeLimit: number, afterLimit: number) => {
    const storage: Record<string, Float32Array | Uint32Array> = {
      before: Float32Array.of(0, first),
      after: Float32Array.of(0, second),
      beforeConverged: Uint32Array.of(1),
      afterConverged: Uint32Array.of(1),
      beforeLimit: Float32Array.of(beforeLimit),
      afterLimit: Float32Array.of(afterLimit),
      difference: new Float32Array(2),
      unreachable: new Uint32Array(2)
    };
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => [index, storage[binding.name]])
    );
    new WgslExec(
      new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
    return {
      difference: Array.from(storage['difference']),
      unreachable: Array.from(storage['unreachable'])
    };
  };
  // An unchanged 7-cost edge is outside the shared 4-cost domain; no lost reachable trip.
  expect(run(7, Infinity, 9, 4)).toEqual({difference: [0, NaN], unreachable: [0, 0]});
  expect(run(3, Infinity, 9, 4)).toEqual({difference: [0, NaN], unreachable: [0, 1]});
  // A finite new path above the shared horizon is unreachable inside that domain, yet its
  // independently valid finite pair-cost difference remains 7-3=4.
  expect(run(3, 7, 4, 9)).toEqual({difference: [0, 4], unreachable: [0, 1]});
});
