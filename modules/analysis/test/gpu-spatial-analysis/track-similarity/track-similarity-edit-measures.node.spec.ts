// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTrackSimilarity,
  GPUTrackSimilaritySearch
} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  computeEditDistanceOnRealSequenceLowerBoundOracle,
  computeEditDistanceOnRealSequenceOracle,
  computeEditDistanceWithRealPenaltyLowerBoundOracle,
  computeEditDistanceWithRealPenaltyOracle,
  computeHausdorffLowerBoundOracle,
  computeSymmetricSegmentPathLowerBoundOracle,
  computeSymmetricSegmentPathOracle,
  createLowerBoundPairs
} from './track-similarity-edit-oracle';
import {computeHausdorffOracle, type Polyline} from './track-similarity-oracle';
import {EDIT_MEASURE_FIXTURE} from './track-similarity-edit-measures-harness';

it('the TypeScript double references reproduce the Python fixture (independent formulations)', () => {
  expect(EDIT_MEASURE_FIXTURE.planar.length).toBeGreaterThan(10);
  for (const entry of EDIT_MEASURE_FIXTURE.planar) {
    for (const {tolerance, distance} of entry.edr) {
      expect(computeEditDistanceOnRealSequenceOracle(entry.a, entry.b, tolerance), entry.name).toBe(
        distance
      );
    }
    for (const {gap, distance} of entry.erp) {
      expect(
        computeEditDistanceWithRealPenaltyOracle(entry.a, entry.b, gap),
        entry.name
      ).toBeCloseTo(distance, 9);
    }
    expect(computeSymmetricSegmentPathOracle(entry.a, entry.b), entry.name).toBeCloseTo(
      entry.sspd ?? NaN,
      9
    );
    expect(computeHausdorffOracle(entry.a, entry.b), entry.name).toBeCloseTo(
      entry.hausdorff ?? NaN,
      9
    );
  }
});

it('the fixture holds hand-checked values', () => {
  // exact-integer: a = (0,0),(10,0); b = (3,4),(13,4). Distances 5, 13.6, 8.06, 5.
  const exact = EDIT_MEASURE_FIXTURE.planar.find(entry => entry.name === 'exact-integer')!;
  expect(exact.edr.find(entry => entry.tolerance === 5)?.distance).toBe(0);
  expect(exact.edr.find(entry => entry.tolerance === 4)?.distance).toBe(2);
  expect(exact.hausdorff).toBe(5);
  // ERP with gap (0,0): substitute both (5 + 5), the cheaper of that and gap detours.
  expect(exact.erp[0].distance).toBe(10);
  const identical = EDIT_MEASURE_FIXTURE.planar.find(entry => entry.name === 'identical')!;
  expect([
    identical.hausdorff,
    identical.sspd,
    identical.edr[0].distance,
    identical.erp[0].distance
  ]).toEqual([0, 0, 0, 0]);
});

type Measure = {
  name: string;
  bound: (a: Polyline, b: Polyline) => number;
  value: (a: Polyline, b: Polyline) => number;
};

const TOLERANCE = 1.5;
const GAP: [number, number] = [20, -10];
const MEASURES: Measure[] = [
  {
    name: 'hausdorff',
    bound: computeHausdorffLowerBoundOracle,
    value: (a, b) => computeHausdorffOracle(a, b)
  },
  {
    name: 'sspd',
    bound: computeSymmetricSegmentPathLowerBoundOracle,
    value: computeSymmetricSegmentPathOracle
  },
  {
    name: 'edr',
    bound: (a, b) => computeEditDistanceOnRealSequenceLowerBoundOracle(a, b, TOLERANCE),
    value: (a, b) => computeEditDistanceOnRealSequenceOracle(a, b, TOLERANCE)
  },
  {
    name: 'erp',
    bound: (a, b) => computeEditDistanceWithRealPenaltyLowerBoundOracle(a, b, GAP),
    value: (a, b) => computeEditDistanceWithRealPenaltyOracle(a, b, GAP)
  }
];

it('every lower bound is at most the true distance (double precision, 404 pairs per measure)', () => {
  const pairs = createLowerBoundPairs(7);
  for (const measure of MEASURES) {
    let tight = 0;
    let positive = 0;
    for (const [a, b] of pairs) {
      const [bound, value] = [measure.bound(a, b), measure.value(a, b)];
      expect(bound, `${measure.name} ${JSON.stringify([a.length, b.length])}`).toBeLessThanOrEqual(
        value + 1e-12
      );
      positive += bound > 0 ? 1 : 0;
      tight += value > 0 && bound >= 0.5 * value ? 1 : 0;
    }
    // Teeth: the bounds are not vacuous (a zero bound passes the inequality trivially).
    expect(positive, `${measure.name} positive bounds`).toBeGreaterThan(100);
    expect(tight, `${measure.name} bounds within a factor of two`).toBeGreaterThan(40);
  }
});

it('a mutated bound is caught by the same check (the inequality has teeth)', () => {
  const pairs = createLowerBoundPairs(7);
  for (const measure of MEASURES) {
    const violations = pairs.filter(
      ([a, b]) => measure.bound(a, b) * 1.5 + 0.5 > measure.value(a, b) + 1e-12
    ).length;
    expect(violations, measure.name).toBeGreaterThan(0);
  }
});

it('GPUTrackSimilarity edit-measure kernels fit the core limits at maximal options', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${Math.random()}`, format, length);
  const nodes = new GPUTrackSimilarity({
    linesA: {kind: 'lines', positions: view('float32x2', 64), lineOffsets: view('uint32', 5)},
    pairA: view('uint32', 16),
    pairB: view('uint32', 16),
    maximumFrechetVertexCount: 2048,
    densify: 1 / 4096,
    output: {
      editDistanceOnRealSequence: view('float32', 16),
      editDistanceWithRealPenalty: view('float32', 16),
      symmetricSegmentPathDistance: view('float32', 16)
    }
  }).getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  for (const id of [
    'track-similarity-edit-distance-on-real-sequence',
    'track-similarity-edit-distance-with-real-penalty',
    'track-similarity-symmetric-segment-path'
  ]) {
    const source = sources.get(id);
    expect(source, id).toBeDefined();
    expect(getStorageBindingCount(source!), id).toBeLessThanOrEqual(8);
    for (const kernel of sources.getConfigurations(id)) {
      expect(getWorkgroupMemoryBytes(kernel.source, kernel.constants), id).toBeLessThanOrEqual(
        16384
      );
    }
    // Fixed-order sums only: no atomics.
    expect(source, id).not.toContain('atomic');
  }
});

it('GPUTrackSimilarity validates the edit-measure props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${Math.random()}`, format, length);
  const props = {
    linesA: {
      kind: 'lines' as const,
      positions: view('float32x2', 8),
      lineOffsets: view('uint32', 3)
    },
    pairA: view('uint32', 4),
    pairB: view('uint32', 4),
    output: {editDistanceOnRealSequence: view('float32', 4)}
  };
  expect(() => new GPUTrackSimilarity(props)).not.toThrow();
  expect(() => new GPUTrackSimilarity({...props, editDistanceTolerance: -1})).toThrow(
    /editDistanceTolerance/
  );
  expect(() => new GPUTrackSimilarity({...props, editDistanceGapPoint: [NaN, 0]})).toThrow(
    /editDistanceGapPoint/
  );
  expect(
    () =>
      new GPUTrackSimilarity({...props, output: {editDistanceOnRealSequence: view('float32', 3)}})
  ).toThrow(/pair capacity/);
});

it('GPUTrackSimilaritySearch builds the bound kernel of each new measure', () => {
  for (const metric of [
    'hausdorff',
    'edit-distance-on-real-sequence',
    'edit-distance-with-real-penalty',
    'symmetric-segment-path'
  ] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `view-${Math.random()}`, format, length);
    const nodes = new GPUTrackSimilaritySearch({
      metric,
      linesA: {kind: 'lines', positions: view('float32x2', 64), lineOffsets: view('uint32', 5)},
      pairA: view('uint32', 16),
      pairB: view('uint32', 16),
      threshold: view('float32', 1),
      output: {
        distance: view('float32', 16),
        lowerBound: view('float32', 16),
        matched: view('uint32', 16),
        status: view('uint32', 16)
      }
    }).getCommandNodes(graph);
    expect(
      nodes.map(node => node.id).some(id => id.endsWith('-bound')),
      metric
    ).toBe(true);
    const sources = getParsedKernelSources(nodes);
    expect(sources.get('track-similarity-search-bound'), metric).toContain(
      'bounds[boundsOffset + index]'
    );
  }
});
