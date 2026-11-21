// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import typeScript from 'typescript';
import {expect, it} from 'vitest';
import type {LocalPooling} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/rate-smoothing.stats';
import {computeKShortestPathsOnCPU} from '../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';
import {
  getSpurInterruptionInput,
  getCandidateSpurInterruptionInput
} from '../gpu-network/k-shortest-paths/k-shortest-paths-test-utils';

function getInitializer(source: string, name: string): string {
  const file = typeScript.createSourceFile(
    'consumer.ts',
    source,
    typeScript.ScriptTarget.Latest,
    true
  );
  let initializer: string | undefined;
  const visit = (node: typeScript.Node) => {
    if (typeScript.isVariableDeclaration(node) && node.name.getText(file) === name)
      initializer = node.initializer?.getText(file);
    typeScript.forEachChild(node, visit);
  };
  visit(file);
  if (!initializer) throw new Error(`Missing ${name} callback`);
  return initializer;
}

it('spatial-2 H2 selected county suppresses incomplete pooling and outline, then recovers', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/statistics/rate-smoothing.compute.ts',
    'utf8'
  );
  const javascript = typeScript.transpileModule(
    `const setSelection = ${getInitializer(source, 'setSelection')};
     const publishSelection = ${getInitializer(source, 'publishSelection')};`,
    {compilerOptions: {target: typeScript.ScriptTarget.ES2022}}
  ).outputText;
  const readouts = new Map<string, unknown>();
  let outline: Float32Array = new Float32Array();
  const womenYears = [1000, 1000, 1000, 1000];
  const callbacks = new Function(
    'ctx',
    'selectionBuffer',
    'neighborhoodBuffer',
    'womenYears',
    `let selectedRow = -1, selectionCount = 0, neighborhoodCount = 0, latest;
     const events = [10,20,30,90];
     const rawPerThousand = [10,20,30,90], RATE_SCALE = 1000;
     const countyName = row => 'County ' + row;
     const formatCount = String, formatPercent = value => (100*value) + '%';
     const publishFunnel = () => {};
     const collectSegments = rows => Float32Array.from(rows.flatMap(row => [row,0,row,1]));
     ${javascript}
     return {select: setSelection, update: value => {latest=value;},
       counts: () => [selectionCount,neighborhoodCount]};`
  )(
    {setReadout: (name: string, value: unknown) => readouts.set(name, value), requestLayers() {}},
    {write() {}},
    {
      write: (values: Float32Array) => {
        outline = values;
      }
    },
    womenYears
  ) as {
    select(row: number): void;
    update(value: unknown): void;
    counts(): number[];
  };
  // Hand oracle: row 2 and neighbors 1,3 have (30+20+90)/(1000+1000+1000)
  // births per woman-year, hence 46.7 per 1000. With only neighbor 1 eligible the
  // rate is (30+20)/2000*1000 = 25.0. Zero exposure excludes row 3 from pooling
  // but preserves its outline when weights are complete.
  const selected = 'County 2: 30 births / 1000 woman-years = 30.0 per 1,000; own-data weight 50%';
  for (const [weightsOverflow, weightsCandidateOverflow, ineligibleDonor] of [
    [0, 0, false],
    [0, 1, false],
    [1, 0, false],
    [0, 0, true],
    [0, 0, false]
  ] as const) {
    const incomplete = Boolean(weightsOverflow || weightsCandidateOverflow);
    womenYears[3] = ineligibleDonor ? 0 : 1000;
    const eligibleNeighborCounts = incomplete ? [1, 1, 1, 1] : [1, 2, 2, 1];
    if (ineligibleDonor) {
      eligibleNeighborCounts[2] = 1;
      eligibleNeighborCounts[3] = 0;
    }
    callbacks.update({
      weightsOverflow,
      weightsCandidateOverflow,
      offsets: incomplete ? [0, 1, 2, 3, 4] : [0, 1, 3, 5, 6],
      neighbors: incomplete ? [1, 0, 1, 2] : [1, 0, 2, 1, 3, 2],
      weight: [0.5, 0.5, 0.5, 0.5],
      local: {
        pooledRates: Float32Array.of(0, 0, incomplete || ineligibleDonor ? 0.025 : 140 / 3000, 0),
        eligibleNeighborCounts: Uint32Array.from(eligibleNeighborCounts)
      } satisfies Pick<LocalPooling, 'pooledRates' | 'eligibleNeighborCounts'>
    });
    callbacks.select(2);
    expect(readouts.get('selected')).toBe(selected);
    expect(readouts.get('localPool')).toBe(
      incomplete
        ? 'unavailable: incomplete weights'
        : ineligibleDonor
          ? '25.0 per 1,000 among 1 neighbours'
          : '46.7 per 1,000 among 2 neighbours'
    );
    expect(callbacks.counts()).toEqual([1, incomplete ? 0 : 2]);
    expect(Array.from(outline)).toEqual(incomplete ? [] : [1, 0, 1, 1, 3, 0, 3, 1]);
  }
});

it('network H2 an interrupted later spur vetoes the round despite an earlier local stop', () => {
  // Hand oracle: query A has a certified first route of cost 2 (3 in the candidate-stop
  // control); query B's first route costs 1. A later active spur exceeds the 64-hop budget,
  // so neither query may publish a second route, whichever query is enumerated first.
  for (const candidateStop of [false, true])
    for (const reversed of [false, true]) {
      const input = candidateStop
        ? getCandidateSpurInterruptionInput()
        : getSpurInterruptionInput();
      if (reversed) {
        input.sources.reverse();
        input.targets.reverse();
      }
      const result = computeKShortestPathsOnCPU(input, {
        routeCount: 2,
        maximumPathLength: candidateStop ? 4 : 3,
        maximumIterations: 1,
        candidateCapacity: candidateStop ? 1 : 256
      });
      const firstCost = candidateStop ? 3 : 2;
      expect(Array.from(result.counts)).toEqual([1, 1]);
      expect(Array.from(result.costs)).toEqual(
        reversed ? [1, Infinity, firstCost, Infinity] : [firstCost, Infinity, 1, Infinity]
      );
      expect(Array.from(result.lengths)).toEqual(
        reversed ? [2, 0, firstCost + 1, 0] : [firstCost + 1, 0, 2, 0]
      );
      expect(Array.from(result.converged)).toEqual([0]);
      expect(Array.from(result.iterationLimitReached)).toEqual([1]);
      expect(Array.from(result.candidateOverflow)).toEqual([0]);
    }
});
