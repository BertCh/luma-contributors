// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import path from 'node:path';
import typeScript from 'typescript';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {computeNetworkPathExtractionOnCPU} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {formatNetworkPathReadout} from '../../../../../examples/deck/spatial-analysis-shared/network-path-readout';

const EXPLORER = 'examples/deck/spatial-analysis-explorer/modes/network-analysis-mode.ts';
const SHOWCASE = 'examples/deck/spatial-analysis-showcase/src/scenes/networks/routing.compute.ts';
const LENS =
  'modules/lenses/src/computation-lenses/scenarios/gpu-network/network-path-extraction.lens.ts';
const EXAMPLE_MODULE =
  '../../../../lenses/src/computation-lenses/scenarios/gpu-network/network-path-extraction.example';

function source(file: string): typeScript.SourceFile {
  return typeScript.createSourceFile(
    file,
    readFileSync(path.resolve(file), 'utf8'),
    typeScript.ScriptTarget.Latest,
    true
  );
}

function findExpression(file: string, readout: string): string {
  const tree = source(file);
  let expression: typeScript.Expression | undefined;
  const visit = (node: typeScript.Node) => {
    if (
      typeScript.isCallExpression(node) &&
      ((readout === 'explorer' && node.expression.getText(tree) === 'routeReadout.setValue') ||
        (node.expression.getText(tree) === 'ctx.setReadout' &&
          typeScript.isStringLiteral(node.arguments[0]) &&
          node.arguments[0].text === readout))
    ) {
      expression = node.arguments[readout === 'explorer' ? 0 : 1];
    }
    typeScript.forEachChild(node, visit);
  };
  visit(tree);
  if (!expression) throw new Error(`Missing ${readout} readout in ${file}`);
  return expression.getText(tree);
}

function evaluate(text: string, bindings: Record<string, unknown>): unknown {
  const code = typeScript.transpileModule(`return (${text});`, {
    compilerOptions: {target: typeScript.ScriptTarget.ES2022}
  }).outputText;
  return new Function(...Object.keys(bindings), code)(...Object.values(bindings));
}

function findShowcaseReadouts(): string {
  const tree = source(SHOWCASE);
  let update: typeScript.FunctionDeclaration | undefined;
  const visit = (node: typeScript.Node) => {
    if (typeScript.isFunctionDeclaration(node) && node.name?.text === 'updateRouteReadouts')
      update = node;
    typeScript.forEachChild(node, visit);
  };
  visit(tree);
  if (!update) throw new Error('Missing showcase readout callback');
  return update.getText(tree);
}

it('showcase clears stale turn comparisons when either route loses its usable edge sequence', () => {
  for (const failure of [
    {treeFound: Uint32Array.of(0), turnFound: Uint32Array.of(0)},
    {treeFound: Uint32Array.of(0)},
    {turnFound: Uint32Array.of(0)},
    {treeOutputOverflow: 1},
    {turnOutputOverflow: 1},
    {treeConverged: 0},
    {turnConverged: 0},
    {turnInputOverflow: 1}
  ]) {
    const summary = {
      treeFound: Uint32Array.of(1),
      treeCost: Float32Array.of(60),
      treeConverged: 1,
      treeCandidateOverflow: 0,
      treeOutputOverflow: 0,
      treeOffsets: Uint32Array.of(0, 2),
      treeEdges: Uint32Array.of(0, 1),
      turnFound: Uint32Array.of(1),
      turnCost: Float32Array.of(90),
      turnConverged: 1,
      turnCandidateOverflow: 0,
      turnOutputOverflow: 0,
      turnInputOverflow: 0,
      turnOffsets: Uint32Array.of(0, 2),
      turnEdges: Uint32Array.of(2, 3)
    };
    const readouts = new Map<string, string>();
    const charts = new Map<string, {values?: number[]} | null>();
    const run = evaluate(findShowcaseReadouts(), {
      ctx: {
        options: {destinationCount: 1, destinationPlace: 'first', showTurnRoute: true},
        setReadout: (name: string, value: string) => readouts.set(name, value),
        setChart: (name: string, chart: {values?: number[]} | null) => charts.set(name, chart)
      },
      summary,
      formatMinutes: (seconds: number) => `${seconds / 60} min`,
      formatNetworkPathReadout,
      cpuCosts: null,
      cpuFreeFlowCosts: null,
      destinationNodes: [1],
      EXTRA_DESTINATIONS: [],
      ROUTING_PLACES: {first: {label: 'First'}},
      MAXIMUM_PATH_LENGTH: 4,
      countTurns: (edges: Uint32Array) =>
        edges[0] === 0 ? {left: 4, right: 6, uTurns: 0} : {left: 3, right: 5, uTurns: 0},
      edgeCount: 4,
      network: {length: [100, 100, 100, 100]},
      costs: [30, 30, 40, 40],
      ROUTE_COLOR: 0,
      TURN_ROUTE_COLOR: 1
    }) as () => void;
    run();
    expect(charts.get('turnComparison')?.values).toEqual([4, 6, 3, 5]);
    Object.assign(summary, failure);
    run();
    expect.soft(charts.get('turnComparison'), JSON.stringify(failure)).toBeNull();
    if (failure.treeFound && failure.turnFound) {
      expect(readouts.get('routeTime')).toBe('no route found');
      expect(readouts.get('turnTime')).toBe('no route found');
    }
    if (failure.treeFound || failure.turnFound || failure.treeConverged === 0)
      expect.soft(readouts.get('turnCost')).toBe('-');
  }
});

for (const nodeCount of [3, 2049]) {
  it(`mixed ${nodeCount}-node batch preserves found costs in all three showcase readouts`, () => {
    for (const reachable of [true, false]) {
      const result = computeNetworkPathExtractionOnCPU({
        predecessors: Uint32Array.from({length: nodeCount}, (_, row) =>
          row ? row - 1 : 0xffffffff
        ),
        costs: Float32Array.from({length: nodeCount}, (_, row) =>
          row === nodeCount - 1 && !reachable ? Infinity : row * 60
        ),
        targets: Uint32Array.of(1, nodeCount - 1),
        maximumPathLength: nodeCount - 1,
        capacity: nodeCount * 2,
        predecessorsComplete: Uint32Array.of(1)
      });
      expect(Array.from(result.pathFound)).toEqual([1, 0]);
      expect(Array.from(result.pathCosts)).toEqual([60, Infinity]);
      expect(result.overflow[0]).toBe(0);
      expect(result.candidateOverflow[0]).toBe(Number(reachable));
      const readouts = new Map<string, string>();
      const run = evaluate(findShowcaseReadouts(), {
        ctx: {
          options: {destinationCount: 2, destinationPlace: 'first', showTurnRoute: true},
          setReadout: (name: string, value: string) => readouts.set(name, value),
          setChart: () => {}
        },
        summary: {
          treeFound: result.pathFound,
          treeCost: result.pathCosts,
          treeConverged: 1,
          treeCandidateOverflow: result.candidateOverflow[0],
          treeOutputOverflow: result.overflow[0],
          treeOffsets: Uint32Array.of(0, 1, 1),
          treeEdges: Uint32Array.of(0),
          turnFound: result.pathFound,
          turnCost: result.pathCosts,
          turnConverged: 1,
          turnCandidateOverflow: result.candidateOverflow[0],
          turnOutputOverflow: result.overflow[0],
          turnInputOverflow: 0,
          turnOffsets: Uint32Array.of(0, 1, 1),
          turnEdges: Uint32Array.of(0)
        },
        formatMinutes: (seconds: number) => `${seconds / 60} min`,
        formatNetworkPathReadout,
        cpuCosts: null,
        cpuFreeFlowCosts: null,
        destinationNodes: [1, nodeCount - 1],
        EXTRA_DESTINATIONS: ['second'],
        ROUTING_PLACES: {first: {label: 'First'}, second: {label: 'Second'}},
        MAXIMUM_PATH_LENGTH: nodeCount - 1,
        countTurns: () => ({left: 0, right: 0, uTurns: 0}),
        edgeCount: 1,
        network: {length: [100]},
        costs: [60],
        ROUTE_COLOR: 0,
        TURN_ROUTE_COLOR: 1
      }) as () => void;
      run();
      expect.soft(readouts.get('routeTime')).toBe('1 min');
      expect.soft(readouts.get('turnTime')).toBe('1 min');
      expect
        .soft(readouts.get('routeTimes'))
        .toBe(
          `1 min / ${reachable ? 'incomplete batch (path walk budget exhausted)' : 'no route found'}`
        );
    }
  });
}

for (const nodeCount of [3, 2049, 4097]) {
  it(`readouts distinguish an exhausted ${nodeCount}-node path walk from an unreachable target`, () => {
    for (const reachable of [true, false]) {
      const result = computeNetworkPathExtractionOnCPU({
        predecessors: Uint32Array.from({length: nodeCount}, (_, row) =>
          row ? row - 1 : 0xffffffff
        ),
        costs: Float32Array.from({length: nodeCount}, (_, row) =>
          row === nodeCount - 1 && !reachable ? Infinity : row
        ),
        targets: Uint32Array.of(nodeCount - 1),
        maximumPathLength: nodeCount - 1,
        capacity: nodeCount,
        predecessorsComplete: Uint32Array.of(1)
      });
      expect(Array.from(result.pathFound)).toEqual([0]);
      expect(Array.from(result.count)).toEqual([0]);
      expect(result.overflow[0]).toBe(0);
      expect(result.candidateOverflow[0]).toBe(Number(reachable));
      const words = Uint32Array.of(
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        48,
        result.candidateOverflow[0],
        result.candidateOverflow[0]
      );
      const bindings = {
        found: false,
        turnFound: false,
        words,
        floats: new Float32Array(10),
        formatNetworkPathReadout,
        summary: {
          treeCost: result.pathCosts,
          treeCandidateOverflow: result.candidateOverflow[0],
          treeOutputOverflow: 0,
          treeConverged: 1,
          turnCost: result.pathCosts,
          turnCandidateOverflow: result.candidateOverflow[0],
          turnOutputOverflow: 0,
          turnConverged: 1,
          turnInputOverflow: 0
        },
        formatMinutes: (seconds: number) => `${seconds / 60} min`
      };
      const expected = reachable
        ? 'incomplete batch (path walk budget exhausted)'
        : 'no route found';
      expect(evaluate(findExpression(EXPLORER, 'explorer'), bindings)).toBe(expected);
      expect(evaluate(findExpression(SHOWCASE, 'routeTime'), bindings)).toBe(expected);
      expect(evaluate(findExpression(SHOWCASE, 'turnTime'), bindings)).toBe(expected);
    }
  });
}

it('explorer forwards the same convergence view from search into path extraction', () => {
  const tree = source(EXPLORER);
  let extraction: typeScript.ObjectLiteralExpression | undefined;
  let search: typeScript.ObjectLiteralExpression | undefined;
  const visit = (node: typeScript.Node) => {
    if (
      typeScript.isNewExpression(node) &&
      node.arguments?.[0] &&
      typeScript.isObjectLiteralExpression(node.arguments[0])
    ) {
      const props = node.arguments[0];
      if (node.expression.getText(tree) === 'GPUNetworkPathExtraction') extraction = props;
      if (node.expression.getText(tree) === 'GPUNetworkReachability') search = props;
    }
    typeScript.forEachChild(node, visit);
  };
  visit(tree);
  const complete = extraction?.properties.find(
    property => property.name?.getText(tree) === 'predecessorsComplete'
  );
  expect(complete).toBeDefined();
  if (!complete || !typeScript.isPropertyAssignment(complete))
    throw new Error('Missing completeness binding');
  expect(search?.getText(tree)).toContain(`converged: ${complete.initializer.getText(tree)}`);
});

it('path lens requests candidate loss and diagnoses it before an unfound target', async () => {
  const tree = source(LENS);
  let capture: typeScript.MethodDeclaration | undefined;
  const visit = (node: typeScript.Node) => {
    if (typeScript.isMethodDeclaration(node) && node.name.getText(tree) === 'capture')
      capture = node;
    typeScript.forEachChild(node, visit);
  };
  visit(tree);
  if (!capture?.body) throw new Error('Missing path lens capture');
  const code = typeScript.transpileModule(capture.body.getText(tree), {
    compilerOptions: {target: typeScript.ScriptTarget.ES2022}
  }).outputText;
  const runCapture = new Function('final', code);
  for (const candidateOverflow of [1, 0]) {
    expect(() =>
      runCapture((name: string) => ({
        values: Uint32Array.of(
          name === 'converged' ? 1 : name === 'candidateOverflow' ? candidateOverflow : 0
        )
      }))
    ).toThrow(candidateOverflow ? /walk.*budget|budget.*exhaust/i : /target was not found/);
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {runExample} = await import(EXAMPLE_MODULE);
  const example = await runExample(device, {
    graph,
    roads: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'weights', 'float32', 2)
    },
    startNodes: createTransientView(graph, 'start', 'uint32', 1),
    targets: createTransientView(graph, 'targets', 'uint32', 1)
  });
  expect(example.paths.props.output.status.candidateOverflow).toBeDefined();
  expect(readFileSync(LENS, 'utf8')).toMatch(/retain:[\s\S]*?['"]candidateOverflow['"]/);
  device.destroy();
});

it('path readouts preserve certified costs while qualifying absent paths with batch loss', () => {
  expect(
    formatNetworkPathReadout(
      {found: true, converged: 0, candidateOverflow: 0, overflow: 0},
      'found'
    )
  ).toBe('search incomplete');
  expect(
    formatNetworkPathReadout(
      {found: true, converged: 1, candidateOverflow: 1, overflow: 0},
      'found'
    )
  ).toBe('found');
  expect(
    formatNetworkPathReadout(
      {found: true, converged: 1, candidateOverflow: 0, overflow: 1},
      'found'
    )
  ).toBe('found');
  expect(
    formatNetworkPathReadout(
      {found: false, converged: 1, candidateOverflow: 0, overflow: 1},
      'found'
    )
  ).toBe('incomplete batch (path output capacity exceeded)');
});

it('explorer copies both walk-loss flags into the words consumed by its route readout', () => {
  const tree = source(EXPLORER);
  let arrow: typeScript.ArrowFunction | undefined;
  const visit = (node: typeScript.Node) => {
    if (
      typeScript.isVariableDeclaration(node) &&
      node.name.getText(tree) === 'summarySources' &&
      node.initializer &&
      typeScript.isArrowFunction(node.initializer)
    )
      arrow = node.initializer;
    typeScript.forEachChild(node, visit);
  };
  visit(tree);
  if (!arrow) throw new Error('Missing explorer summary sources');
  const names = [
    'routePathFound',
    'routePathCost',
    'routeNodeCount',
    'routeNodeOverflow',
    'routeEdgeCount',
    'routeEdgeOverflow',
    'routeConverged',
    'routeIterations',
    'routeNodeCandidateOverflow',
    'routeEdgeCandidateOverflow'
  ];
  const plan = evaluate(
    arrow.getText(tree),
    Object.fromEntries(names.map(name => [name, name]))
  ) as (target: string) => [string, number][];
  expect(plan('route')).toEqual(names.map(name => [name, 1]));
});
