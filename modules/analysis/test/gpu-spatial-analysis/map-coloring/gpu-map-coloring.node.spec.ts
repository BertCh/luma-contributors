// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUMapColoringParameterValues,
  GPUMapColoring,
  GPU_MAP_COLORING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/map-coloring/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {computeContiguityOracle} from '../spatial-weights/spatial-weights-oracle';
import fixture from '../../oracles/fixtures/geometry-map-coloring.json';
import {createSeededRandom} from '../spatial-weights/spatial-weights-oracle';
import {
  computeMapColoringOracle,
  countColoringConflicts,
  createCliqueCSR,
  createOracleCSRFromEdges,
  createRandomPlanarCSR,
  createRookLatticeCSR,
  createStarCSR,
  getMapColoringOrder,
  simulateMapColoringRounds
} from './map-coloring-oracle';

function setup(id: string = 'map-coloring-nodes') {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const props = {
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    parameters: view('uint32', GPU_MAP_COLORING_PARAMETER_LENGTH),
    output: {colors: view('uint32', 4)}
  };
  return {graph, view, props};
}

it('AMG-54 accepts packed scalar outputs with a weights loss port', () => {
  const {graph, view, props} = setup('packed-loss');
  const packed = view('uint32', 2);
  const colorCount = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const conflictCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  expect(() =>
    graph.add(
      new GPUMapColoring({
        ...props,
        weights: {...props.weights, status: {overflow: view('uint32', 1)}},
        output: {...props.output, colorCount, conflictCount}
      }).getCommandNodes(graph)
    )
  ).not.toThrow();
});

it('GPUMapColoring validates props and declares gated rounds', () => {
  const {graph, view, props: base} = setup();
  expect(() => new GPUMapColoring({...base, output: {colors: view('uint32', 3)}})).toThrow(
    /row count/
  );
  expect(() => new GPUMapColoring({...base, maximumIterations: 0})).toThrow(/maximumIterations/);
  expect(() => new GPUMapColoring({...base, parameters: view('uint32', 0)})).toThrow(/parameters/);
  const ids = new GPUMapColoring({
    ...base,
    maximumIterations: 3,
    output: {...base.output, conflictCount: view('uint32', 1)}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('map-coloring-round-2-select');
  expect(ids).not.toContain('map-coloring-round-3-select');
  expect(ids.at(-1)).toBe('map-coloring-conflicts');
});

it('GPUMapColoring packs the seed per frame', () => {
  expect(Array.from(getGPUMapColoringParameterValues())).toEqual([0]);
  expect(Array.from(getGPUMapColoringParameterValues({seed: 0xffffffff}))).toEqual([0xffffffff]);
  expect(() => getGPUMapColoringParameterValues({seed: -1})).toThrow(/seed/);
  expect(() => getGPUMapColoringParameterValues({seed: 1.5})).toThrow(/seed/);
});

it('GPUMapColoring keeps one pipeline per kernel across rounds (rule 12)', () => {
  const build = (maximumIterations: number) => {
    const {graph, view, props} = setup();
    return new GPUMapColoring({
      ...props,
      maximumIterations,
      output: {...props.output, conflictCount: view('uint32', 1)}
    }).getCommandNodes(graph);
  };
  expect(getGPUCommandNodePipelineKeys(build(4)).size).toBe(
    getGPUCommandNodePipelineKeys(build(3)).size
  );
});

it('GPUMapColoring validates and compiles the priority option', () => {
  const {props} = setup();
  expect(() => new GPUMapColoring({...props, priority: 'smallest-last' as 'random'})).toThrow(
    /priority/
  );
  expect(new GPUMapColoring(props).priority).toBe('largest-degree-first');
  const kernelKeys = (priority?: 'random' | 'largest-degree-first') => {
    const built = setup();
    return getGPUCommandNodePipelineKeys(
      new GPUMapColoring({...built.props, priority}).getCommandNodes(built.graph)
    );
  };
  expect(kernelKeys().size).toBe(kernelKeys('random').size);
  expect([...kernelKeys()]).not.toEqual([...kernelKeys('random')]);
  expect([...kernelKeys('largest-degree-first')]).toEqual([...kernelKeys()]);
});

const SCENES: [string, () => ReturnType<typeof createRookLatticeCSR>][] = [
  ['lattice', () => createRookLatticeCSR(12, 9)],
  ['planar', () => createRandomPlanarCSR(30, 30, createSeededRandom(5))],
  ['planar-2', () => createRandomPlanarCSR(40, 25, createSeededRandom(11))],
  ['star', () => createStarCSR(40)],
  ['clique', () => createCliqueCSR(9)],
  ['empty', () => createOracleCSRFromEdges(0, [])],
  ['isolated', () => createOracleCSRFromEdges(7, [[1, 2]])]
];

it('the rounds model equals sequential greedy in priority order for both priorities', () => {
  for (const [name, create] of SCENES) {
    const csr = create();
    for (const priority of ['largest-degree-first', 'random'] as const) {
      for (const seed of [0, 7, 0xffffffff]) {
        const {colors, rounds} = simulateMapColoringRounds(csr, seed, priority);
        const expected = computeMapColoringOracle(csr, seed, priority);
        expect(colors, `${name} ${priority} ${seed}`).toEqual(expected);
        expect(countColoringConflicts(csr, colors)).toBe(0);
        expect(rounds).toBeLessThanOrEqual(64);
      }
    }
  }
});

it('largest-degree-first order is degree descending, then hash, then ID', () => {
  const csr = createStarCSR(5);
  const order = getMapColoringOrder(csr, 3);
  expect(order[0]).toBe(0);
  expect(order.slice().sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
  // Equal degrees (isolated rows) fall back to the hash and then the ID.
  expect(getMapColoringOrder(createOracleCSRFromEdges(4, []), 0, 'largest-degree-first')).toEqual(
    getMapColoringOrder(createOracleCSRFromEdges(4, []), 0, 'random')
  );
  expect(computeMapColoringOracle(createStarCSR(40), 0)).toEqual([
    0,
    ...new Array<number>(40).fill(1)
  ]);
  expect(
    computeMapColoringOracle(createCliqueCSR(9), 0)
      .slice()
      .sort((a, b) => a - b)
  ).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
});

it('largest-degree-first needs no more colors than random on planar maps (aggregate) and few rounds', () => {
  let degreeFirstTotal = 0;
  let randomTotal = 0;
  let maximumRounds = 0;
  for (const seed of [0, 1, 2, 3, 4, 5, 6, 7]) {
    const csr = createRandomPlanarCSR(60, 60, createSeededRandom(100 + seed));
    const degreeFirst = simulateMapColoringRounds(csr, seed, 'largest-degree-first');
    const random = simulateMapColoringRounds(csr, seed, 'random');
    degreeFirstTotal += Math.max(...degreeFirst.colors) + 1;
    randomTotal += Math.max(...random.colors) + 1;
    maximumRounds = Math.max(maximumRounds, degreeFirst.rounds);
  }
  expect(degreeFirstTotal).toBeLessThanOrEqual(randomTotal);
  // Chains of strictly decreasing degree are short on planar maps: the default cap of 64 is ample.
  expect(maximumRounds).toBeLessThanOrEqual(32);
});

it('largest-degree-first stays within one color of networkx largest_first on Delaunay maps', () => {
  for (const graph of fixture.graphs) {
    const csr = {offsets: graph.offsets, neighbors: graph.neighbors, weights: [], distances: []};
    const degreeFirst = simulateMapColoringRounds(csr, 0, 'largest-degree-first');
    const random = simulateMapColoringRounds(csr, 0, 'random');
    expect(countColoringConflicts(csr, degreeFirst.colors), graph.name).toBe(0);
    const colorCount = Math.max(...degreeFirst.colors) + 1;
    // Same heuristic, different tie-breaking (seeded hash instead of networkx's stable sort).
    expect(colorCount, graph.name).toBeLessThanOrEqual(graph.networkxColorCounts.largest_first + 1);
    expect(colorCount, graph.name).toBeLessThanOrEqual(Math.max(...random.colors) + 1);
    expect(degreeFirst.rounds, graph.name).toBeLessThanOrEqual(64);
    console.log(
      `${graph.name}: ldf colors=${colorCount} rounds=${degreeFirst.rounds}; random colors=${Math.max(...random.colors) + 1} rounds=${random.rounds}; networkx`,
      graph.networkxColorCounts
    );
  }
});

// GEO-107: truncated adjacency gives a coloring that is proper on the stored pairs but not on the
// true adjacency, so a weights loss flag must not be ignored (decision LR-D4).
it('GPUMapColoring suppresses its outputs behind every weights loss flag', () => {
  // The defect, on the sequential oracle: drop each row's last queen neighbor of a 4 x 3 grid.
  const polygons = [];
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < 4; x++) {
      polygons.push([
        [
          [x, y],
          [x + 1, y],
          [x + 1, y + 1],
          [x, y + 1]
        ] as [number, number][]
      ]);
    }
  }
  const complete = computeContiguityOracle(polygons, 'queen');
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row + 1 < complete.offsets.length; row++) {
    const start = complete.offsets[row];
    const end = complete.offsets[row + 1];
    neighbors.push(...complete.neighbors.slice(start, Math.max(start, end - 1)));
    offsets.push(neighbors.length);
  }
  const truncated = {offsets, neighbors, weights: neighbors.map(() => 1), distances: []};
  const truncatedColors = computeMapColoringOracle(truncated, 0);
  expect(countColoringConflicts(truncated, truncatedColors)).toBe(0);
  expect(countColoringConflicts(complete, truncatedColors)).toBeGreaterThan(0);

  const {graph, view, props} = setup();
  const output = {
    ...props.output,
    colorCount: view('uint32', 1),
    conflictCount: view('uint32', 1),
    status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
  };
  const withoutStatus = new GPUMapColoring({...props, output}).getCommandNodes(graph);
  expect(withoutStatus.some(node => node.id.includes('incomplete'))).toBe(false);
  const nodes = new GPUMapColoring({
    ...props,
    id: 'map-coloring-loss',
    weights: {
      ...props.weights,
      status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
    },
    output
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.filter(id => id.includes('incomplete'))).toHaveLength(8);
  expect(ids.indexOf('map-coloring-loss-conflicts')).toBeLessThan(ids.length - 8);
  const sources = getGPUCommandNodeKernelSources(nodes.slice(-8));
  expect(sources).toHaveLength(8);
  for (const {source, constants} of sources) {
    expect(source).not.toContain('iterationCount[');
    expect(auditWGSLModule(source, constants)).toEqual([]);
  }
  expect(
    sources.filter(({source}) => source.includes('colors[colorsOffset + index] = 4294967295u;'))
  ).toHaveLength(2);
  for (const name of ['colorCount', 'conflictCount', 'converged']) {
    expect(
      sources.filter(({source}) => source.includes(`${name}[${name}Offset] = 0u;`))
    ).toHaveLength(2);
  }
});
