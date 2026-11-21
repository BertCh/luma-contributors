// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createHash} from 'node:crypto';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getResourceBuffer} from '../network-resource-buffers';
import {expect, it} from 'vitest';
import {
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  GPUMapMatching,
  type GPUMapMatchingProps
} from '../../../src/gpu-network/map-matching/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  createManhattanGrid,
  createRouteQueries,
  getExactRouteDistance,
  searchRouteMirror,
  type RouteTablePolicy
} from './route-search-mirror';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  output: Partial<GPUMapMatchingProps['output']> = {}
): GPUMapMatchingProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `map-matching-view-${serial++}`, format, length);
  return {
    id: 'map-matching',
    positions: view('float32x2', 6),
    trackOffsets: view('uint32', 2),
    topology: {
      nodePositions: view('float32x2', 4),
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8)
    },
    parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
    cellSize: 50,
    bounds: {minimum: [0, 0], maximum: [200, 200]},
    output: {matchedEdges: view('uint32', 6), ...output}
  };
}

type BudgetTally = {missed: number; wrong: number; silent: number; droppingSearches: number};

/** Runs the mirror over the B-02 queries and tallies misses against exact Dijkstra. */
function tallyRouteBudget(
  sampleSpacing: number,
  budget: number,
  policy: RouteTablePolicy
): BudgetTally {
  const grid = createManhattanGrid(41, 100);
  const tally: BudgetTally = {missed: 0, wrong: 0, silent: 0, droppingSearches: 0};
  for (const {start, target, limit} of createRouteQueries(41, 100, sampleSpacing, 227)) {
    const exact = getExactRouteDistance(grid, start, target, limit);
    const {targetDistances, droppedCount} = searchRouteMirror(
      grid,
      start,
      limit,
      [target],
      budget,
      policy
    );
    const found = targetDistances[0];
    const isWrong = exact < 0 ? found >= 0 : found < 0 || Math.abs(found - exact) > 1e-6;
    tally.missed += exact >= 0 && found < 0 ? 1 : 0;
    tally.wrong += isWrong ? 1 : 0;
    tally.silent += isWrong && droppedCount === 0 ? 1 : 0;
    tally.droppingSearches += droppedCount > 0 ? 1 : 0;
  }
  return tally;
}

// Review B-02: the route search's node table silently dropped nodes in discovery order. The mirror
// follows the WGSL search line for line (A*, table of `routeNodeBudget` nodes) on a 100 m
// Manhattan grid with the default bound 3 * straight + 50; the GPU spec runs the real kernel.
it('route search budget: the old drop policy misses most 1 km routes silently', () => {
  const old = tallyRouteBudget(1000, 64, 'reject');
  // 171 of 227 routes missed at 1 km spacing (the review's own mimic, with plain Dijkstra and
  // its own sampling, found 186), and nothing reported them before the fix.
  expect(old).toEqual({missed: 171, wrong: 171, silent: 0, droppingSearches: 173});
  expect(tallyRouteBudget(500, 64, 'reject').wrong).toBe(0);
});

it('route search budget: eviction keeps more routes and every wrong one is counted', () => {
  // Evicting the worst open key keeps the search heading to the target: the same budget covers
  // 800 m samples (64 nodes) and 1.1 km (128 nodes) exactly, where dropping missed 92 and 25.
  expect(tallyRouteBudget(800, 64, 'reject').missed).toBe(92);
  expect(tallyRouteBudget(800, 64, 'evict-worst')).toEqual({
    missed: 0,
    wrong: 0,
    silent: 0,
    droppingSearches: 92
  });
  expect(tallyRouteBudget(1100, 128, 'reject').missed).toBe(25);
  expect(tallyRouteBudget(1100, 128, 'evict-worst').wrong).toBe(0);
  // Past the budget's reach it still misses routes, but never silently: each wrong route comes
  // from a search that dropped nodes, which raises candidateOverflow and routeOverflow.
  const beyond = tallyRouteBudget(1000, 64, 'evict-worst');
  expect(beyond).toEqual({missed: 80, wrong: 80, silent: 0, droppingSearches: 173});
});

it('GPUMapMatching reports route-search drops through candidateOverflow and routeOverflow', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-status'});
  const overflow = createTransientView(graph, 'mm-overflow', 'uint32', 1);
  const routeOverflow = createTransientView(graph, 'mm-route-overflow', 'uint32', 1);
  const nodes = new GPUMapMatching(
    createProps(graph, {status: {candidateOverflow: overflow, routeOverflow}})
  ).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  const routes = ids.indexOf('map-matching-routes');
  const status = ids.indexOf('map-matching-route-status');
  expect(routes).toBeGreaterThanOrEqual(0);
  expect(status).toBeGreaterThan(routes);
  expect(ids.indexOf('map-matching-clear-route-overflow')).toBeGreaterThanOrEqual(0);
  const statusBuffers = nodes[status].resources?.map(resource => getResourceBuffer(resource)) ?? [];
  expect(statusBuffers).toContain(overflow);
  expect(statusBuffers).toContain(routeOverflow);
  // The route kernel stays within the default 8 storage buffers.
  expect(nodes[routes].resources?.length).toBe(8);

  expect(
    () =>
      new GPUMapMatching(
        createProps(graph, {
          status: {routeOverflow: createTransientView(graph, 'mm-bad', 'uint32', 2)}
        })
      )
  ).toThrow(/output.status.routeOverflow must contain 1 rows/);
});

it('GPUMapMatching clears the whole edge table before the per-node edge kernel', () => {
  // Review B-04: the per-node kernel writes only rows offsets[i]..offsets[i + 1], so the rows
  // past offsets[NODE_COUNT] kept stale words and became zero-length or phantom edges.
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-edge-table'});
  const ids = new GPUMapMatching(createProps(graph)).getCommandNodes(graph).map(node => node.id);
  const clear = ids.indexOf('map-matching-clear-edge-table');
  expect(clear).toBeGreaterThanOrEqual(0);
  expect(clear).toBeLessThan(ids.indexOf('map-matching-edge-table'));
});

it('GPUMapMatching treats point 0 as a start, bounds whole routes and folds its summary counts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = (length: number) =>
    createTransientView(graph, `summary-view-${serial++}`, 'uint32', length);
  const contributor = new GPUMapMatching(
    createProps(graph, {matchedCount: view(1), breakCount: view(1)})
  );
  // Review B-17: a candidate is a directed edge row, so the default keeps 4 two-way roads.
  expect(contributor.candidateCount).toBe(8);
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  const routes = sources.get('map-matching-routes') ?? '';
  // Review B-23: point 0 has no predecessor even before the first track.
  expect(routes).toContain('if (point == 0u || pointStarts[pointStartsOffset + point] != 0u)');
  // Review B-24: the bound covers the whole route, partial end edges included.
  expect(routes).toContain('if (route > bound)');
  // Rule 14: one atomic per count per workgroup.
  expect(sources.get('map-matching-summary')).toContain('workgroupCounters');
});

it('GPUMapMatching validates its topology', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-topology'});
  const base = createProps(graph);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `map-matching-topology-${serial++}`, format, length);
  expect(
    () =>
      new GPUMapMatching({
        ...base,
        topology: {offsets: view('uint32', 5), neighbors: view('uint32', 8)} as never
      })
  ).toThrow(/topology\.nodePositions is required/);
  expect(
    () =>
      new GPUMapMatching({
        ...base,
        topology: {...base.topology, nodePositions: view('float32x2', 3)}
      })
  ).toThrow(/topology\.nodePositions must contain one row per node/);
});

it('great-circle snapping compares local candidate terms before inverse trig rounding', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  const nodes = new GPUMapMatching({
    ...props,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    bounds: {minimum: [-1, -1], maximum: [1, 2]},
    cellSize: 1
  }).getCommandNodes(graph);
  const source = getParsedKernelSources(nodes).get('map-matching-candidates') ?? '';
  const selector = source.slice(
    source.indexOf('fn geodesicSegmentFraction'),
    source.indexOf('fn geodesicSegmentDistance')
  );
  expect(selector).toContain('geodesicSegmentHaversineTerm(point, start, end, fraction)');
  expect(selector).toContain('min(startTerm, endTerm) < projectedTerm');
});

function createRouteTable(graph: GPUCommandGraph, withComplete = true) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `route-table-view-${serial++}`, format, length);
  return {
    offsets: view('uint32', 5),
    targets: view('uint32', 16),
    costs: view('float32', 16),
    costLimit: view('float32', 1),
    ...(withComplete ? {status: {complete: view('uint32', 1)}} : {})
  };
}

it('unfinished cost tables fall through to A* within eight storage bindings', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const complete = createTransientView(graph, 'table-complete', 'uint32', 1);
  const routeTable = {...createRouteTable(graph), status: {complete}};
  const nodes = new GPUMapMatching({...createProps(graph), routeTable}).getCommandNodes(graph);
  const source = getParsedKernelSources(nodes).get('map-matching-route-bounds') ?? '';
  expect(source).toContain('if (tableComplete[tableCompleteOffset] == 0u)');
  expect(getStorageBindingCount(source)).toBe(8);
  expect(
    nodes.find(node => node.id === 'map-matching-route-bounds')?.resources?.map(getResourceBuffer)
  ).toContain(complete);
});

it('GPUMapMatching without routeTable keeps its node list and routes WGSL', () => {
  const plainGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-plain'});
  const plain = new GPUMapMatching(createProps(plainGraph)).getCommandNodes(plainGraph);
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-with-table'});
  const table = new GPUMapMatching({
    ...createProps(graph),
    routeTable: createRouteTable(graph)
  }).getCommandNodes(graph);
  const plainIds = plain.map(node => node.id);
  expect(plainIds.some(id => id.includes('route-bounds') || id.includes('route-table'))).toBe(
    false
  );
  // The table adds exactly two kernels, right before the A* route kernel; nothing else moves.
  const tableIds = table.map(node => node.id);
  expect(tableIds.filter(id => !plainIds.includes(id))).toEqual([
    'map-matching-route-bounds',
    'map-matching-route-table'
  ]);
  expect(tableIds.indexOf('map-matching-route-table') + 1).toBe(
    tableIds.indexOf('map-matching-routes')
  );
  expect(tableIds.filter(id => plainIds.includes(id))).toEqual(plainIds);
  const plainRoutes = getParsedKernelSources(plain).get('map-matching-routes') ?? '';
  expect(plainRoutes).not.toContain('route-table');
  expect(getParsedKernelSources(table).get('map-matching-routes')).toContain(
    'if (routes[routesOffset + index * ROUTE_STRIDE + K] >= 0.0)'
  );
});

it('GPUMapMatching routeTable kernels parse, bind the table and stay within 8 bindings', () => {
  for (const withComplete of [true, false]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: `map-matching-${withComplete}`
    });
    const routeTable = createRouteTable(graph, withComplete);
    const nodes = new GPUMapMatching({...createProps(graph), routeTable}).getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    if (!withComplete) {
      expect(sources.has('map-matching-route-bounds')).toBe(false);
      expect(sources.has('map-matching-route-table')).toBe(false);
      expect(sources.get('map-matching-routes')).not.toContain(
        'marks a row the route table resolved'
      );
      continue;
    }
    for (const [id, bindingCount] of [
      ['map-matching-route-bounds', 8],
      ['map-matching-route-table', 6]
    ] as const) {
      for (const {source} of sources.getConfigurations(id))
        expect(getStorageBindingCount(source), id).toBe(bindingCount);
    }
    const byId = (id: string) =>
      nodes.find(node => node.id === id)?.resources?.map(resource => getResourceBuffer(resource)) ??
      [];
    expect(byId('map-matching-route-table')).toEqual(
      expect.arrayContaining([routeTable.offsets, routeTable.targets, routeTable.costs])
    );
    expect(byId('map-matching-route-bounds')).toContain(routeTable.costLimit);
    expect(byId('map-matching-route-bounds').includes(routeTable.status?.complete)).toBe(true);
    // The row search is a binary search of ascending targets.
    expect(sources.get('map-matching-route-table')).toContain('fn findTableCost');
    // The table answers only costs within the allowance, as the A* search does.
    expect(sources.get('map-matching-route-table')).toContain('network <= remaining');
    expect(nodes.find(node => node.id === 'map-matching-routes')?.resources?.length).toBe(8);
  }
});

it('GPUMapMatching validates routeTable', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-table-bad'});
  const make = (overrides: Partial<ReturnType<typeof createRouteTable>>) => () =>
    new GPUMapMatching({
      ...createProps(graph),
      routeTable: {...createRouteTable(graph), ...overrides}
    });
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `route-table-bad-${serial++}`, format, length);
  expect(make({offsets: view('uint32', 4)})).toThrow(
    /routeTable.offsets must contain nodeCount \+ 1 rows/
  );
  expect(make({costs: view('float32', 15)})).toThrow(
    /routeTable.costs length must equal routeTable.targets length/
  );
  expect(make({costLimit: view('float32', 2)})).toThrow(
    /routeTable.costLimit must contain exactly one row/
  );
  expect(make({status: {complete: view('uint32', 2)}})).toThrow(
    /routeTable.status.complete must contain exactly one row/
  );
  // A table written by the matcher's own outputs would alias; outputs may not share its buffers.
  const props = createProps(graph);
  const routeTable = createRouteTable(graph);
  expect(
    () =>
      new GPUMapMatching({
        ...props,
        routeTable,
        output: {...props.output, status: {overflow: routeTable.status?.complete}}
      })
  ).toThrow(/outputs must not share buffers with inputs/);
});

it('NET-126 pins matcher search, sparse fallback, Viterbi and route publication before extraction', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const withTable of [false, true]) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const props = createProps(graph);
      const contributor = new GPUMapMatching({
        ...props,
        traversedEdgeCapacity: 24,
        ...(withTable ? {routeTable: createRouteTable(graph)} : {}),
        output: {
          ...props.output,
          traversedEdges: {
            edges: createTransientView(graph, 'pin-traversed', 'uint32', 24),
            counts: createTransientView(graph, 'pin-counts', 'uint32', 1),
            status: {overflow: createTransientView(graph, 'pin-overflow', 'uint32', 1)}
          },
          trackLogLikelihoods: createTransientView(graph, 'pin-likelihood', 'float32', 1)
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      const digest = createHash('sha256')
        .update(JSON.stringify([...getParsedKernelSources(nodes)]))
        .digest('hex');
      const bindings = new Map(
        nodes.flatMap(
          node =>
            getWGSLKernelNodeRecord(graph, node.id)?.bindings.map(
              binding => [`${node.id}:${binding.name}`, binding.view] as const
            ) ?? []
        )
      );
      const {output} = contributor.props;
      const expectedBindings: Record<string, GraphDataView> = {
        'map-matching-matched:matchedEdges': output.matchedEdges,
        'map-matching-backward:logLikelihoods': output.trackLogLikelihoods!,
        'map-matching-route-edges:traversedEdges': output.traversedEdges!.edges,
        'map-matching-route-edges:traversedCounts': output.traversedEdges!.counts,
        'map-matching-traversed-status:outputOverflow': output.traversedEdges!.status!.overflow!
      };
      const assertDeclaration = (bindingViews: ReadonlyMap<string, GraphDataView>) => {
        for (const [name, expectedView] of Object.entries(expectedBindings)) {
          expect(bindingViews.get(name), name).toBe(expectedView);
        }
        expect
          .soft(digest, `withTable=${withTable}`)
          .toBe(
            withTable
              ? 'fa1e5d52f0362b444e6eb5c412255a2bb7d6483bc329c73e8fdd048de3772fe1'
              : 'b01b508b87fbbc33b67767dc46b51bd12b8a2caffdc38f894c09efa10a8fe73a'
          );
      };
      assertDeclaration(bindings);
      const swapped = new Map(bindings);
      swapped.set(
        'map-matching-matched:matchedEdges',
        expectedBindings['map-matching-route-edges:traversedCounts']
      );
      swapped.set(
        'map-matching-route-edges:traversedCounts',
        expectedBindings['map-matching-matched:matchedEdges']
      );
      expect(() => assertDeclaration(swapped)).toThrow();
    }
  } finally {
    device.destroy();
  }
});
