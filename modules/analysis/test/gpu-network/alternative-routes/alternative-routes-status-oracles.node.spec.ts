// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeAlternativeRoutesOnCPU,
  CPUAlternativeRoutesOperation
} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {getOracleCPUVectors} from './output-oracle-harness';
import {
  GPUAlternativeRoutesOperation,
  type AlternativeRoutesOperationProps
} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import {
  GPUCommandGraph,
  createTransientView,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUAlternativeRoutes} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes';

for (const Operation of [GPUAlternativeRoutesOperation, CPUAlternativeRoutesOperation])
  it(`R10 ${Operation.name} captures each search-budget getter once`, async () => {
    let reads = 0;
    const operation = new Operation({
      nodeCount: 3,
      queryCount: 1,
      routeCount: 2,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      sources: 'sources',
      targets: 'targets',
      maximumFrontierSize: 2,
      get maximumExpansions() {
        if (++reads !== 1) throw new Error('Expansion budget read twice');
        return 1;
      },
      output: {
        vertices: 'vertices',
        edges: 'edges',
        lengths: 'lengths',
        costs: 'costs',
        counts: 'counts',
        complete: 'complete'
      }
    });
    expect(reads).toBe(1);
    if (operation instanceof CPUAlternativeRoutesOperation) {
      const result = await operation.execute({
        inputs: getOracleCPUVectors({
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          sources: {format: 'uint32', values: Uint32Array.of(0)},
          targets: {format: 'uint32', values: Uint32Array.of(2)}
        })
      });
      // Route 0->1->2 requires two examinations. The captured allowance of one
      // cannot certify it, independently of either backend's search code.
      expect([
        ...(result['counts'] as Uint32Array),
        ...(result['complete'] as Uint32Array)
      ]).toEqual([0, 0]);
    } else {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const views: Record<string, GraphDataView> = {
        offsets: createTransientView(graph, 'offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
        weights: createTransientView(graph, 'weights', 'float32', 2),
        sources: createTransientView(graph, 'sources', 'uint32', 1),
        targets: createTransientView(graph, 'targets', 'uint32', 1)
      };
      for (const [name, descriptor] of Object.entries(operation.outputs))
        views[name] = createTransientView(graph, name, descriptor.format, descriptor.length);
      const contributor = operation.createContributor(
        <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
      );
      if (!(contributor instanceof GPUAlternativeRoutes))
        throw new Error('Expected alternative-route contributor');
      expect(contributor.maximumExpansions).toBe(1);
    }
    expect(reads).toBe(1);
  });

it('R7 repeated lowering retains construction-time search budgets after caller mutation', () => {
  const props: AlternativeRoutesOperationProps = {
    nodeCount: 3,
    queryCount: 1,
    routeCount: 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    maximumExpansions: 1,
    maximumFrontierSize: 2,
    output: {
      vertices: 'vertices',
      edges: 'edges',
      lengths: 'lengths',
      costs: 'costs',
      counts: 'counts',
      complete: 'complete'
    }
  };
  const operation = new GPUAlternativeRoutesOperation(props);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const views: Record<string, GraphDataView> = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 4),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
    weights: createTransientView(graph, 'weights', 'float32', 2),
    sources: createTransientView(graph, 'sources', 'uint32', 1),
    targets: createTransientView(graph, 'targets', 'uint32', 1)
  };
  for (const [name, descriptor] of Object.entries(operation.outputs))
    views[name] = createTransientView(graph, name, descriptor.format, descriptor.length);
  const lower = () => {
    const contributor = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(name: string) => views[name] as GraphDataView<Format>
    );
    if (!(contributor instanceof GPUAlternativeRoutes))
      throw new Error('Expected alternative-route contributor');
    return contributor;
  };
  expect(lower().maximumExpansions).toBe(1);
  props.maximumExpansions = 2;
  props.maximumFrontierSize = 3;
  const contributor = lower();
  expect(contributor.maximumExpansions).toBe(1);
  expect(contributor.maximumFrontierSize).toBe(2);
});

for (const maximumExpansions of [1, 2]) {
  it(`certifies only independently exhausted chain queries with ${maximumExpansions} examinations`, () => {
    // The only route from 0 to 2 examines edges 0->1 and 1->2. With one examination,
    // the named unresolved item is edge 1->2. Source 1 needs one edge, source 2 none;
    // source 99 is invalid. Requesting two routes also proves exhaustion, not just top-1.
    const input = {
      topology: {
        offsets: Uint32Array.of(0, 1, 2, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(1, 1)
      },
      sources: Uint32Array.of(0, 1, 2, 99),
      targets: Uint32Array.of(2, 2, 2, 2)
    };
    const options = {routeCount: 2, maximumFrontierSize: 2, maximumExpansions};
    const result = computeAlternativeRoutesOnCPU(input, options);
    expect(Array.from(result.counts)).toEqual(
      maximumExpansions === 1 ? [0, 1, 1, 0] : [1, 1, 1, 0]
    );
    expect(Array.from(result.complete)).toEqual(
      maximumExpansions === 1 ? [0, 1, 1, 1] : [1, 1, 1, 1]
    );
    input.topology.weights.fill(-1);
    const empty = computeAlternativeRoutesOnCPU(input, options);
    expect(Array.from(empty.counts)).toEqual([0, 0, 1, 0]);
    expect(Array.from(empty.complete)).toEqual([1, 1, 1, 1]);
  });
}

for (const maximumFrontierSize of [1, 2]) {
  it(`derives diamond frontier completion from ${maximumFrontierSize} pending slots`, () => {
    // Expanding 0 requires both pending prefixes [0,1] and [0,2]. One slot refuses
    // the second before any destination is popped. Two slots enumerate exactly
    // [0,1,3] and [0,2,3] in four edge examinations, with no other simple paths.
    const result = computeAlternativeRoutesOnCPU(
      {
        topology: {
          offsets: Uint32Array.of(0, 2, 3, 4, 4),
          neighbors: Uint32Array.of(1, 2, 3, 3),
          weights: Float32Array.of(1, 1, 1, 1)
        },
        sources: Uint32Array.of(0),
        targets: Uint32Array.of(3)
      },
      {routeCount: 2, maximumFrontierSize, maximumExpansions: 4}
    );
    expect(Array.from(result.counts)).toEqual(maximumFrontierSize === 1 ? [0] : [2]);
    expect(Array.from(result.complete)).toEqual(maximumFrontierSize === 1 ? [0] : [1]);
  });
}
