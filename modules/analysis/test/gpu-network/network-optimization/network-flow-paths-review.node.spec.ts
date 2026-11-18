// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkFlowOnCPU,
  CPUNetworkFlowOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {prepareGPUNetworkFlowDemands} from '../../../src/gpu-network/network-optimization/prepare-flow-demands';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {validateNetworkFlowPaths} from '../../../src/gpu-network/network-optimization/network-flow-paths';

for (const costed of [false, true]) {
  it(`R1 ${costed ? 'primal-dual' : 'uncosted'} charges another phase after returning stranded flow`, () => {
    const input = prepareGPUNetworkFlowDemands({
      nodeCount: 4,
      edges: [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3]
      ],
      capacities: [2, 2, 1, 1],
      demands: [-2, 0, 0, 2],
      costs: costed ? [0, 0, 0, 0] : undefined
    });
    for (const budget of [1, 2]) {
      const result = computeNetworkFlowOnCPU(input, {
        ...input,
        solver: 'parallel',
        pathCapacity: 2,
        mode: costed ? 'primal-dual' : undefined,
        maximumRelabels: costed ? undefined : budget,
        maximumCostPhases: costed ? budget : undefined
      });
      expect(result.paths?.status.count).toEqual(Uint32Array.of(budget));
      expect(result.paths?.volumes).toEqual(Uint32Array.of(1, budget - 1));
      expect(result.summary[2]).toBe(budget);
      expect(result.summary[3]).toBe(2 - budget);
      expect(result.status.iterationCount).toEqual(Uint32Array.of(budget));
      expect(result.paths?.status.converged).toEqual(Uint32Array.of(budget - 1));
      expect(result.paths?.status.iterationLimitReached).toEqual(Uint32Array.of(2 - budget));
    }
  });
  it(`R2 ${costed ? 'primal-dual' : 'uncosted'} bounds sink labels independently of a source dead end`, () => {
    const result = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.of(0, 1, 0, 2, 2, 3),
        capacities: Uint32Array.of(1, 1, 1),
        costs: costed ? new Uint32Array(3) : undefined,
        parameters: Uint32Array.of(0, 1, 1)
      },
      {
        nodeCount: 4,
        solver: 'parallel',
        pathCapacity: 1,
        maximumRelaxationRounds: 2,
        maximumRelabels: costed ? undefined : 1,
        maximumCostPhases: costed ? 1 : undefined,
        mode: costed ? 'primal-dual' : undefined
      }
    );
    expect(result.paths?.status.count).toEqual(Uint32Array.of(1));
    expect(result.paths?.edgeRows).toEqual(Uint32Array.of(0, 0xffffffff, 0xffffffff));
    expect(result.paths?.volumes).toEqual(Uint32Array.of(1));
    expect(result.paths?.status.converged).toEqual(Uint32Array.of(1));
    expect(result.paths?.status.iterationLimitReached).toEqual(Uint32Array.of(0));
    expect(result.summary[7]).toBe(0); // The separate source-side cut search exhausted its two rounds.
  });
}

const PROPS: NetworkFlowOperationProps = {
  id: 'review-flow',
  nodeCount: 2,
  edgeCount: 1,
  pathCapacity: 1,
  solver: 'reference',
  edges: 'edges',
  capacities: 'capacities',
  parameters: 'parameters',
  output: {
    flow: 'flow',
    cut: 'cut',
    summary: 'summary',
    paths: {
      edgeRows: 'pathEdges',
      lengths: 'pathLengths',
      volumes: 'pathVolumes',
      status: {
        count: 'pathCount',
        requiredCount: 'pathRequired',
        overflow: 'pathOverflow',
        converged: 'pathConverged',
        iterationLimitReached: 'pathLimited',
        invalidCount: 'pathInvalid'
      }
    }
  }
};
const SHAPES = {
  edges: {format: 'uint32x2' as const, length: 1},
  capacities: {format: 'uint32' as const, length: 1},
  parameters: {format: 'uint32' as const, length: 3}
};

it('R3 all forms reject reference maximumRelabels with the caller ID and prop', () => {
  const props = {...PROPS, maximumRelabels: 0};
  for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation])
    expect(() => new Operation(props)).toThrow(/^review-flow.*maximumRelabels.*parallel/);
  expect(() =>
    constructNetworkOperationContributor(new GPUNetworkFlowOperation(props), SHAPES)
  ).toThrow(/^review-flow.*maximumRelabels.*parallel/);
  const contributor = constructNetworkOperationContributor(
    new GPUNetworkFlowOperation(PROPS),
    SHAPES
  ) as GPUNetworkFlow;
  expect(() => new GPUNetworkFlow({...contributor.props, maximumRelabels: 0})).toThrow(
    /^review-flow.*maximumRelabels.*parallel/
  );
  expect(() =>
    computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.of(0, 1),
        capacities: Uint32Array.of(1),
        parameters: Uint32Array.of(0, 1, 1)
      },
      {nodeCount: 2, solver: 'reference', maximumRelabels: 0, pathCapacity: 1}
    )
  ).toThrow(/^computeNetworkFlowOnCPU.*maximumRelabels.*parallel/);
});

it.each([
  'status',
  'volumes'
] as const)('R4 missing path %s has a named error in every form', missing => {
  const paths = {...PROPS.output.paths!};
  delete (paths as Partial<typeof paths>)[missing];
  const props = {...PROPS, output: {...PROPS.output, paths}};
  for (const Operation of [GPUNetworkFlowOperation, CPUNetworkFlowOperation])
    expect(() => new Operation(props)).toThrow(
      new RegExp(`^review-flow.*output\\.paths\\.${missing}.*required`)
    );
  // Direct contributor props are also checked, using valid operation views first.
  const contributor = constructNetworkOperationContributor(
    new GPUNetworkFlowOperation(PROPS),
    SHAPES
  ) as GPUNetworkFlow;
  const contributorProps = contributor.props;
  const malformedPaths = {...contributorProps.output.paths!};
  delete (malformedPaths as Partial<typeof malformedPaths>)[missing];
  expect(() => validateNetworkFlowPaths('review-flow', 2, 1, malformedPaths)).toThrow(
    new RegExp(`^review-flow.*output\\.paths\\.${missing}.*required`)
  );
  expect(
    () =>
      new GPUNetworkFlow({
        ...contributorProps,
        output: {...contributorProps.output, paths: malformedPaths}
      })
  ).toThrow(new RegExp(`^review-flow.*output\\.paths\\.${missing}.*required`));
});

it('R5 overflow retains the direct BFS path where edge-ordered DFS retains two edges', () => {
  const result = computeNetworkFlowOnCPU(
    {
      edges: Uint32Array.of(0, 1, 1, 2, 0, 2),
      capacities: Uint32Array.of(1, 1, 1),
      parameters: Uint32Array.of(0, 2, 2)
    },
    {nodeCount: 3, pathCapacity: 1}
  );
  expect(result.paths?.edgeRows).toEqual(Uint32Array.of(2, 0xffffffff));
  expect(result.paths?.lengths).toEqual(Uint32Array.of(1));
  expect(result.paths?.volumes).toEqual(Uint32Array.of(1));
  expect(result.paths?.status.requiredCount).toEqual(Uint32Array.of(2));
  expect(result.paths?.status.overflow).toEqual(Uint32Array.of(1));
  const findDepthFirstPath = (
    edges: Uint32Array,
    flow: Uint32Array,
    source: number,
    sink: number
  ): number[] | undefined => {
    if (source === sink) return [];
    for (let edge = 0; edge < flow.length; edge++) {
      if (!flow[edge] || edges[edge * 2] !== source) continue;
      const suffix = findDepthFirstPath(edges, flow, edges[edge * 2 + 1], sink);
      if (suffix) return [edge, ...suffix];
    }
    return undefined;
  };
  expect(findDepthFirstPath(Uint32Array.of(0, 1, 1, 2, 0, 2), result.flow, 0, 2)).toEqual([0, 1]);
  const priorEdges = Uint32Array.of(0, 1, 0, 2, 1, 3, 2, 3, 3, 4, 3, 5, 4, 6, 5, 6);
  const prior = computeNetworkFlowOnCPU(
    {
      edges: priorEdges,
      capacities: new Uint32Array(8).fill(1),
      parameters: Uint32Array.of(0, 6, 2)
    },
    {nodeCount: 7, pathCapacity: 1}
  );
  const depthFirst = findDepthFirstPath(priorEdges, prior.flow, 0, 6);
  expect(depthFirst).toEqual([0, 2, 4, 6]);
  expect(Array.from(prior.paths!.edgeRows.slice(0, 4))).toEqual(depthFirst); // Old fixture could not distinguish DFS.
});
