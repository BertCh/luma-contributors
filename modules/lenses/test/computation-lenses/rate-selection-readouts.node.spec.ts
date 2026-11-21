// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {computeContiguityWeightsOnCPU} from '../../../analysis/src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {getLocalPooling} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/rate-smoothing.stats';
import {
  formatCount,
  formatPercent
} from '../../../../examples/deck/spatial-analysis-showcase/src/cartography/live-text';

it.each([
  'candidate',
  'final'
] as const)('H2 selected row hides %s loss and its neighborhood, then recovers', cause => {
  const positions = Float32Array.from(
    Array.from({length: 4}, (_, column) => [
      ...Array.from({length: 128}, (_, row) => [column, row / 127]),
      ...Array.from({length: 128}, (_, row) => [column + 1, (127 - row) / 127])
    ]).flat(2)
  );
  const input = {
    positions,
    ringOffsets: Uint32Array.of(0, 256, 512, 768, 1024),
    polygonOffsets: Uint32Array.of(0, 1, 2, 3, 4),
    featureOffsets: Uint32Array.of(0, 1, 2, 3, 4)
  };
  const partial = computeContiguityWeightsOnCPU(input, {
    criterion: 'queen',
    capacity: 96,
    pairCapacity: 384
  });
  const complete = computeContiguityWeightsOnCPU(input, {
    criterion: 'queen',
    capacity: 96,
    pairCapacity: 2048
  });
  const finalLoss = computeContiguityWeightsOnCPU(input, {
    criterion: 'queen',
    capacity: 3,
    pairCapacity: 2048
  });
  expect(Array.from(partial.status.overflow)).toEqual([0]);
  expect(Array.from(partial.status.candidateOverflow)).toEqual([1]);
  expect(Array.from(finalLoss.status.overflow)).toEqual([1]);
  expect(Array.from(finalLoss.status.candidateOverflow)).toEqual([0]);
  const events = Float32Array.of(10, 20, 40, 80);
  const womenYears = new Float32Array(4).fill(1000);
  const state = (weights: typeof partial) => ({
    offsets: weights.offsets,
    neighbors: weights.neighbors,
    weightsOverflow: Boolean(weights.status.overflow[0]),
    weightsCandidateOverflow: Boolean(weights.status.candidateOverflow[0]),
    local: getLocalPooling(events, womenYears, weights.offsets, weights.neighbors),
    weight: new Float32Array(4).fill(0.5)
  });
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/statistics/rate-smoothing.compute.ts',
    'utf8'
  );
  const start = source.indexOf('const setSelection = (row: number) => {');
  const end = source.indexOf('const formatRateText', start);
  expect(start).toBeGreaterThan(0);
  expect(end).toBeGreaterThan(start);
  // Remove only the two local TypeScript annotations; execute the production closures verbatim.
  const body = source
    .slice(start, end)
    .replace('const setSelection = (row: number)', 'const setSelection = (row)')
    .replace('const members: number[]', 'const members');
  const readouts = new Map<string, unknown>();
  const neighborhoodWrites: Float32Array[] = [];
  const factory = new Function(
    'events',
    'womenYears',
    'formatCount',
    'formatPercent',
    'ctx',
    'collectSegments',
    'selectionBuffer',
    'neighborhoodBuffer',
    `let latest; let selectedRow = -1; let selectionCount = 0; let neighborhoodCount = 0; const rawPerThousand = events.map(value => value); const RATE_SCALE = 1000; const countyName = row => 'County ' + row; const publishFunnel = () => {}; ${body} return {select(state) {latest = state; setSelection(2);}, outlineCount() {return neighborhoodCount;}};`
  );
  const selection = factory(
    events,
    womenYears,
    formatCount,
    formatPercent,
    {setReadout: (key: string, value: unknown) => readouts.set(key, value), requestLayers() {}},
    (members: number[]) => new Float32Array(members.flatMap(member => [member, 0, member, 1])),
    {write() {}},
    {write: (values: Float32Array) => neighborhoodWrites.push(values)}
  );
  selection.select(state(complete));
  expect(readouts.get('localPool')).toBe('46.7 per 1,000 among 2 neighbours');
  expect(selection.outlineCount()).toBe(2);
  const own = readouts.get('selected');
  selection.select(state(cause === 'final' ? finalLoss : partial));
  expect(readouts.get('localPool')).toMatch(/unavailable/);
  expect(selection.outlineCount()).toBe(0);
  expect(neighborhoodWrites.at(-1)).toHaveLength(0);
  expect(readouts.get('selected')).toBe(own);
  selection.select(state(complete));
  expect(readouts.get('localPool')).toBe('46.7 per 1,000 among 2 neighbours');
  expect(selection.outlineCount()).toBe(2);
});
