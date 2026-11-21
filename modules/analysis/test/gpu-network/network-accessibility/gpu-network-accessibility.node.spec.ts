// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addNetworkCatchmentAdapter} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  getGPUNetworkAccessibilityParameterValues,
  type GPUNetworkAccessibilityProps,
  type GPUNetworkCostMatrixProps,
  type GPUNetworkSnappingProps
} from '../../../src/gpu-network/network-accessibility/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

let viewCount = 0;

/** Creates a transient view with a graph-unique ID. */
function createView<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

function createMatrixProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkCostMatrixProps> = {},
  topology: Partial<GPUNetworkCostMatrixProps['topology']> = {},
  output: Partial<GPUNetworkCostMatrixProps['output']> = {}
): GPUNetworkCostMatrixProps {
  return {
    id: 'matrix',
    topology: {
      offsets: createView(graph, 'uint32', 9),
      neighbors: createView(graph, 'uint32', 12),
      weights: createView(graph, 'float32', 12),
      ...topology
    },
    startNodes: createView(graph, 'uint32', 10),
    output: {costs: createView(graph, 'float32', 5 * 8), ...output},
    ...overrides
  };
}

it('GPUNetworkCostMatrix schedules lane expansion and one reachability per batch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostMatrix({
    ...createMatrixProps(graph, {}, {}, {status: {converged: createView(graph, 'uint32', 1)}}),
    startsPerRow: 2,
    laneCount: 2,
    maximumIterations: 3
  });
  expect(contributor.rowCount).toBe(5);
  expect(contributor.nodeCount).toBe(8);
  expect(contributor.batchCount).toBe(3);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  // Lanes share the CSR, so only the seeds are expanded.
  expect(ids[0]).toBe('matrix-expand-seeds');
  expect(ids.some(id => id.includes('expand-offsets') || id.includes('expand-edges'))).toBe(false);
  for (const batch of [0, 1, 2]) {
    expect(ids).toContain(`matrix-batch-${batch}-seed`);
    expect(ids).toContain(`matrix-batch-${batch}-relax-2`);
    expect(ids).not.toContain(`matrix-batch-${batch}-relax-3`);
  }
  expect(ids.at(-1)).toBe('matrix-converged');
  // Deterministic across calls.
  const again = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    new GPUNetworkCostMatrix({
      ...createMatrixProps(again, {}, {}, {status: {converged: createView(again, 'uint32', 1)}}),
      startsPerRow: 2,
      laneCount: 2,
      maximumIterations: 3
    })
      .getCommandNodes(again)
      .map(node => node.id)
  ).toEqual(ids);
});

it('GPUNetworkCostMatrix validates shapes and lane counts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(new GPUNetworkCostMatrix(createMatrixProps(graph)).laneCount).toBe(5);
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {}, {}, {costs: createView(graph, 'float32', 13)})
      )
  ).toThrow(/multiple of the node count/);
  expect(() => new GPUNetworkCostMatrix(createMatrixProps(graph, {laneCount: 6}))).toThrow(
    /laneCount/
  );
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {
          startCosts: createView(graph, 'float32', 3)
        })
      )
  ).toThrow(/startCosts length/);
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {}, {weights: createView(graph, 'float32', 2)})
      )
  ).toThrow(/topology.weights length/);
  expect(() => new GPUNetworkCostMatrix(createMatrixProps(graph, {startsPerRow: 0}))).toThrow(
    /startsPerRow/
  );
  const missingWeights = createMatrixProps(graph);
  expect(
    () =>
      new GPUNetworkCostMatrix({
        ...missingWeights,
        topology: {...missingWeights.topology, weights: undefined as never}
      })
  ).toThrow(/topology.weights is required/);
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {}, {offsets: createView(graph, 'float32' as never, 9) as never})
      )
  ).toThrow(/topology.offsets/);
});

function createAccessibilityProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkAccessibilityProps> = {}
): GPUNetworkAccessibilityProps {
  return {
    id: 'access',
    costs: createView(graph, 'float32', 4 * 10),
    opportunityWeights: createView(graph, 'float32', 4),
    parameters: createView(graph, 'float32', 4),
    ...overrides
  } as GPUNetworkAccessibilityProps;
}

it('accessibility props require at least one result selection, including combinations', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const common = {
    costs: createView(graph, 'float32', 4),
    opportunityWeights: createView(graph, 'float32', 2),
    parameters: createView(graph, 'float32', 4)
  };
  const cumulative = createView(graph, 'float32', 2);
  const gravity = createView(graph, 'float32', 2);
  const catchment = {
    demand: createView(graph, 'float32', 2),
    output: createView(graph, 'float32', 2)
  };
  const accept = (_props: GPUNetworkAccessibilityProps) => {};
  for (const selection of [
    {cumulative},
    {gravity},
    {catchment},
    {cumulative, gravity},
    {cumulative, catchment},
    {gravity, catchment},
    {cumulative, gravity, catchment}
  ])
    accept({...common, ...selection});
  // @ts-expect-error At least one public score output is required.
  accept(common);
  // @ts-expect-error Explicit undefined selections do not request a score.
  accept({...common, cumulative: undefined, gravity: undefined, catchment: undefined});
});

it('catchment adapter preserves trip direction and only transposes storage axes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const matrix = new GPUNetworkCostMatrix(
    createMatrixProps(
      graph,
      {direction: 'start-to-target'},
      {},
      {status: {converged: createView(graph, 'uint32', 1)}}
    )
  );
  const common = {
    matrix: matrix.result,
    facilityAxis: 'columns' as const,
    trips: 'demand-to-facility' as const,
    supply: createView(graph, 'float32', 8),
    demand: createView(graph, 'float32', 5),
    parameters: createView(graph, 'float32', 4),
    output: createView(graph, 'float32', 5),
    complete: createView(graph, 'uint32', 1)
  };
  const adapted = addNetworkCatchmentAdapter(graph, common);
  expect(adapted.matrix.values).toBe(common.matrix.values);
  expect(adapted.matrix.direction).toBe('start-to-target');
  expect(adapted.contributor.props.costs === matrix.result.values).toBe(false);
  expect(adapted.contributor.rowCount).toBe(8);
  expect(adapted.contributor.nodeCount).toBe(5);
  expect(() =>
    addNetworkCatchmentAdapter(graph, {...common, id: 'wrong-trip', trips: 'facility-to-demand'})
  ).toThrow(/transposing storage does not reverse trips/);
});

it.each([
  'method',
  'parameters'
])('preflights invalid catchment %s before mutating the graph (S1)', invalid => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const matrix = new GPUNetworkCostMatrix(
    createMatrixProps(graph, {}, {}, {status: {converged: createView(graph, 'uint32', 1)}})
  );
  const props = {
    matrix: matrix.result,
    facilityAxis: 'columns' as const,
    trips: 'demand-to-facility' as const,
    supply: createView(graph, 'float32', 8),
    demand: createView(graph, 'float32', 5),
    parameters: createView(graph, 'float32', 4),
    output: createView(graph, 'float32', 5),
    complete: createView(graph, 'uint32', 1)
  };
  expect(() =>
    addNetworkCatchmentAdapter(graph, {
      ...props,
      ...(invalid === 'method'
        ? {method: 'unknown' as never}
        : {parameters: createView(graph, 'float32', 2)})
    })
  ).toThrow(new RegExp(invalid));
  expect(() => addNetworkCatchmentAdapter(graph, props)).not.toThrow();
});

it('GPUNetworkAccessibility schedules scores and catchment nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      cumulative: createView(graph, 'float32', 10),
      catchment: {
        demand: createView(graph, 'float32', 10),
        output: createView(graph, 'float32', 10)
      }
    })
  );
  expect(contributor.rowCount).toBe(4);
  expect(contributor.nodeCount).toBe(10);
  // A-19: the catchment composes GPUNetworkCostWeights (facility rows and the transpose) with the
  // one floating catchment kernel, GPUCatchmentAccessibility.
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('access-scores');
  expect(ids.slice(-2)).toEqual(['access-catchment-ratios', 'access-catchment-scores']);
  for (const side of ['facility', 'demand']) {
    for (const stage of ['count', 'fill', 'finalize']) {
      expect(ids).toContain(`access-catchment-${side}-weights-${stage}`);
    }
  }
  expect(ids.indexOf('access-catchment-demand-weights-finalize')).toBeLessThan(
    ids.indexOf('access-catchment-ratios')
  );
  const threeStep = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      id: 'access3',
      catchment: {
        method: '3sfca',
        demand: createView(graph, 'float32', 10),
        output: createView(graph, 'float32', 10)
      }
    })
  );
  expect(threeStep.getCommandNodes(graph).map(node => node.id)).toContain(
    'access3-catchment-selection'
  );
  const origin = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      orientation: 'origin-rows',
      opportunityWeights: createView(graph, 'float32', 10),
      gravity: createView(graph, 'float32', 4)
    })
  );
  expect(origin.rowCount).toBe(4);
  expect(origin.nodeCount).toBe(10);
});

it('GPUNetworkAccessibility validates orientation, outputs, and parameters', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUNetworkAccessibility(createAccessibilityProps(graph))).toThrow(
    /at least one/
  );
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          gravity: createView(graph, 'float32', 4)
        })
      )
  ).toThrow(/origin count/);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          parameters: createView(graph, 'float32', 2),
          gravity: createView(graph, 'float32', 10)
        })
      )
  ).toThrow(/parameters/);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          orientation: 'origin-rows',
          opportunityWeights: createView(graph, 'float32', 10),
          catchment: {
            demand: createView(graph, 'float32', 10),
            output: createView(graph, 'float32', 10)
          }
        })
      )
  ).toThrow(/opportunity-rows/);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          opportunityWeights: createView(graph, 'float32', 3)
        })
      )
  ).toThrow(/multiple/);
});

it('GPUNetworkAccessibility carries matrix completeness and the cost-limit rule into status.complete (NET-101, NET-103)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const gravity = createView(graph, 'float32', 10);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          gravity,
          status: {complete: createView(graph, 'uint32', 1)}
        })
      )
  ).toThrow(/status\.complete requires costsComplete/);
  for (const [name, overrides] of [
    ['costsComplete', {costsComplete: createView(graph, 'uint32', 2)}],
    ['costLimit', {costLimit: createView(graph, 'float32', 2)}],
    [
      'status.complete',
      {
        costsComplete: createView(graph, 'uint32', 1),
        status: {complete: createView(graph, 'uint32', 2)}
      }
    ]
  ] as const) {
    expect(
      () => new GPUNetworkAccessibility(createAccessibilityProps(graph, {gravity, ...overrides})),
      name
    ).toThrow(new RegExp(`${name.replace('.', '\\.')} must contain exactly one row`));
  }
  const limited = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      id: 'limited',
      gravity,
      costsComplete: createView(graph, 'uint32', 1),
      costLimit: createView(graph, 'float32', 1),
      status: {complete: createView(graph, 'uint32', 1)},
      catchment: {
        demand: createView(graph, 'float32', 10),
        output: createView(graph, 'float32', 10)
      }
    })
  );
  const limitedNodes = limited.getCommandNodes(graph);
  expect(limitedNodes.at(-1)?.id).toBe('limited-complete');
  expect(getParsedKernelSources(limitedNodes).get('limited-complete')).toContain(
    'let isComplete = costsComplete[costsCompleteOffset] != 0u && parameters[parametersOffset] <= costLimit[costLimitOffset];'
  );
  const unlimited = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      id: 'unlimited',
      gravity: createView(graph, 'float32', 10),
      costsComplete: createView(graph, 'uint32', 1),
      status: {complete: createView(graph, 'uint32', 1)}
    })
  );
  const unlimitedSource = getParsedKernelSources(unlimited.getCommandNodes(graph)).get(
    'unlimited-complete'
  );
  expect(unlimitedSource).toContain('let isComplete = costsComplete[costsCompleteOffset] != 0u;');
  expect(unlimitedSource).not.toContain('parameters');
  // Without public status.complete, supplied loss is certified internally and suppresses gravity.
  const plain = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      id: 'plain',
      gravity: createView(graph, 'float32', 10),
      costsComplete: createView(graph, 'uint32', 1)
    })
  );
  expect(plain.getCommandNodes(graph).map(node => node.id)).toEqual([
    'plain-input-complete',
    'plain-scores',
    'plain-suppress-gravity'
  ]);
});

it('getGPUNetworkAccessibilityParameterValues packs threshold, decay code, beta, minimum cost', () => {
  expect(Array.from(getGPUNetworkAccessibilityParameterValues({threshold: 30}))).toEqual([
    30, 0, 0, 1
  ]);
  expect(
    Array.from(
      getGPUNetworkAccessibilityParameterValues({
        threshold: Infinity,
        decay: 'power',
        beta: 2,
        minimumCost: 0.5
      })
    )
  ).toEqual([Infinity, 2, 2, 0.5]);
  expect(
    getGPUNetworkAccessibilityParameterValues({
      threshold: 1,
      decay: 'exponential',
      beta: 1
    })[1]
  ).toBe(1);
});

function createSnappingProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkSnappingProps> = {},
  output: Partial<GPUNetworkSnappingProps['output']> = {},
  topology: Partial<GPUNetworkSnappingProps['topology']> = {}
): GPUNetworkSnappingProps {
  return {
    id: 'snap',
    positions: createView(graph, 'float32x2', 6),
    topology: {
      offsets: createView(graph, 'uint32', 5),
      neighbors: createView(graph, 'uint32', 5),
      nodePositions: createView(graph, 'float32x2', 4),
      ...topology
    },
    output: {snappedEdges: createView(graph, 'uint32', 6), ...output},
    ...overrides
  };
}

it('GPUNetworkSnapping schedules a scan or a BVH join and optional outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const scan = new GPUNetworkSnapping(
    createSnappingProps(
      graph,
      {},
      {
        startNodes: createView(graph, 'uint32', 12),
        startCosts: createView(graph, 'float32', 12),
        snappedPositions: createView(graph, 'float32x2', 6)
      }
    )
  );
  expect(scan.getCommandNodes(graph).map(node => node.id)).toEqual([
    'snap-edge-sources-clear',
    'snap-edge-sources',
    'snap-scan',
    'snap-costs',
    'snap-positions',
    'snap-seeds'
  ]);
  const joinGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const join = new GPUNetworkSnapping(
    createSnappingProps(
      joinGraph,
      {
        maximumSnapDistance: createView(joinGraph, 'float32', 1),
        candidateCapacity: 64
      },
      {status: {overflow: createView(joinGraph, 'uint32', 1)}}
    )
  );
  const ids = join.getCommandNodes(joinGraph).map(node => node.id);
  // The edge-source rows are cleared first, so rows past offsets[nodeCount] are never stale.
  expect(ids[0]).toBe('snap-edge-sources-clear');
  expect(ids[1]).toBe('snap-edge-sources');
  expect(ids[2]).toBe('snap-segments');
  expect(ids.some(id => id.startsWith('snap-join-'))).toBe(true);
  expect(ids.at(-1)).toBe('snap-project');
});

it('GPUNetworkSnapping validates topology, capacities, and output lengths', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {}, {nodePositions: createView(graph, 'float32x2', 3)})
      )
  ).toThrow(/topology.nodePositions must contain one row per node/);
  const missingPositions = createSnappingProps(graph);
  expect(
    () =>
      new GPUNetworkSnapping({
        ...missingPositions,
        topology: {...missingPositions.topology, nodePositions: undefined as never}
      })
  ).toThrow(/topology.nodePositions is required/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {}, {weights: createView(graph, 'float32', 4)})
      )
  ).toThrow(/topology.weights length/);
  expect(() => new GPUNetworkSnapping(createSnappingProps(graph, {candidateCapacity: 8}))).toThrow(
    /requires maximumSnapDistance/
  );
  // Rule 9: a candidate capacity can drop candidates, so its overflow flag is required.
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {
          candidateCapacity: 8,
          maximumSnapDistance: createView(graph, 'float32', 1)
        })
      )
  ).toThrow(/candidateCapacity requires output\.status\.overflow/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {startNodes: createView(graph, 'uint32', 12)})
      )
  ).toThrow(/together/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {snapFractions: createView(graph, 'float32', 5)})
      )
  ).toThrow(/position count/);
});

it('GPUNetworkSnapping refuses unindexed work above its limit and validates maximumUnindexedWork', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 6 points times 5 edges is 30 tests: a limit of 29 refuses, 30 accepts.
  expect(
    () => new GPUNetworkSnapping(createSnappingProps(graph, {maximumUnindexedWork: 29}))
  ).toThrow(/^snap positions times topology edges without candidateCapacity must be at most 29$/);
  expect(
    () => new GPUNetworkSnapping(createSnappingProps(graph, {maximumUnindexedWork: 30}))
  ).not.toThrow();
  expect(
    () => new GPUNetworkSnapping(createSnappingProps(graph, {maximumUnindexedWork: 0}))
  ).toThrow(/^snap maximumUnindexedWork must be an integer in \[1, /);
});
