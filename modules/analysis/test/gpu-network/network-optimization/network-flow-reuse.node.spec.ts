// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {computeNetworkFlowOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {createNetworkFlowPathNodes} from '../../../src/gpu-network/network-optimization/network-flow-paths';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  FLOW_REUSE_EDGES,
  FLOW_REUSE_COSTS,
  FLOW_REUSE_CAPACITIES,
  expectFlowReuseRows
} from './network-flow-reuse.test-utils';

for (const scenario of ['complete', 'path-overflow', 'solver-limit'] as const)
  it(`flow path storage reuse (${scenario}) clears populated rows on smaller, empty and invalid solves`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const pathCapacity = scenario === 'path-overflow' ? 1 : 2;
    const limited = scenario === 'solver-limit';
    const arrays = new Map<GraphDataView, Uint32Array>();
    const view = (name: string, length: number) =>
      createTransientView(graph, name, 'uint32', length);
    const values = (dataView: GraphDataView) => {
      if (!arrays.has(dataView))
        arrays.set(
          dataView,
          new Uint32Array(dataView.length * (dataView.format === 'uint32x2' ? 2 : 1)).fill(
            0xdeadbeef
          )
        );
      return arrays.get(dataView)!;
    };
    const edges = createTransientView(graph, 'edges', 'uint32x2', 4);
    const flow = view('flow', 4);
    const parameters = view('parameters', 3);
    const summary = view('summary', 8);
    const paths = {
      edgeRows: view('pathEdges', pathCapacity * 3),
      lengths: view('pathLengths', pathCapacity),
      volumes: view('pathVolumes', pathCapacity),
      status: {
        count: view('pathCount', 1),
        requiredCount: view('pathRequired', 1),
        overflow: view('pathOverflow', 1),
        invalidCount: view('pathInvalid', 1),
        converged: view('pathConverged', 1),
        iterationLimitReached: view('pathLimited', 1)
      }
    };
    try {
      // Declare the path graph once. All four submissions retain the same arrays, including tails.
      const programs = createNetworkFlowPathNodes(
        graph,
        'reuse',
        4,
        edges,
        flow,
        parameters,
        summary,
        paths
      ).map(node => {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        const source = record.source
          .replace(/^\s*_\s*=[^;]*;/gm, '')
          .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1, 1, 1)');
        return {record, program: new WgslParser().parse(source)};
      });
      values(edges).set(FLOW_REUSE_EDGES);
      for (let stage = 0; stage < 4; stage++) {
        const parameterValues = Uint32Array.of(stage === 3 ? 99 : 0, 3, 0xffffffff);
        const result = computeNetworkFlowOnCPU(
          {
            edges: FLOW_REUSE_EDGES,
            costs: FLOW_REUSE_COSTS,
            capacities: FLOW_REUSE_CAPACITIES[stage],
            parameters: parameterValues
          },
          {nodeCount: 4, maximumAugmentations: limited ? 1 : 4, pathCapacity}
        );
        values(flow).set(result.flow);
        values(parameters).set(parameterValues);
        values(summary).set(result.summary);
        for (const {record, program} of programs) {
          new WgslExec(program).dispatchWorkgroups(
            'main',
            [...record.workgroups],
            {
              0: Object.fromEntries(
                record.bindings.map((binding, index) => [index, values(binding.view)])
              )
            },
            {constants: record.constants}
          );
        }
        const rows: Record<string, Uint32Array> = {
          flow: values(flow),
          pathEdges: values(paths.edgeRows),
          pathLengths: values(paths.lengths),
          pathVolumes: values(paths.volumes),
          pathCount: values(paths.status.count),
          pathRequired: values(paths.status.requiredCount),
          pathOverflow: values(paths.status.overflow),
          pathInvalid: values(paths.status.invalidCount),
          pathConverged: values(paths.status.converged),
          pathLimited: values(paths.status.iterationLimitReached),
          invalid: result.status.invalidCount,
          converged: result.status.converged,
          limited: result.status.iterationLimitReached
        };
        expectFlowReuseRows(
          (name, expected) => expect(Array.from(rows[name]), `${stage} ${name}`).toEqual(expected),
          stage,
          pathCapacity,
          limited
        );
        expect(values(paths.edgeRows)).toEqual(result.paths!.edgeRows);
        expect(values(paths.lengths)).toEqual(result.paths!.lengths);
        expect(values(paths.volumes)).toEqual(result.paths!.volumes);
      }
    } finally {
      device.destroy();
    }
  });

it('S1 unit paths have distinct total costs so the limited incumbent is independent of ties', () => {
  // Paths [0,2] and [1,3] are disjoint; a unique cheaper path fixes the one-unit incumbent.
  expect(FLOW_REUSE_COSTS[0] + FLOW_REUSE_COSTS[2]).toBe(0);
  expect(FLOW_REUSE_COSTS[1] + FLOW_REUSE_COSTS[3]).toBe(1);
});
