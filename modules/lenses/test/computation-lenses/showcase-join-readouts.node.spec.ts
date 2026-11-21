// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';
import {
  decodeGPUOffsetCurveComponentsStatus,
  getGPUOffsetCurveRowsPerVertex
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import topologicalRelations from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/joins/topological-relations.scene';

// Execute the production readout closures without constructing deck.gl or a device.
function readClosure(path: string, start: string, end: string): string {
  const source = readFileSync(path, 'utf8');
  const first = source.indexOf(start);
  expect(first).toBeGreaterThan(0);
  const last = source.indexOf(end, first);
  expect(last).toBeGreaterThan(first);
  return transpileModule(source.slice(first, last), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
}

it.each([
  'candidate',
  'final',
  'both'
])('storm readout withholds dependent results on %s loss and recovers', cause => {
  const readouts = new Map<string, unknown>();
  const charts = new Map<string, {values?: ArrayLike<number>}>();
  const names = ['ctx', 'formatInteger', 'formatCount', 'quantile'];
  const factory = new Function(
    ...names,
    `
    const reportCount = 2, warningCount = 2, chainCount = 2;
    const kindNames = ['hail'], KIND_LABELS = {hail: 'hail'};
    const reportKinds = [0, 0], warningId = [0, 1];
    const PAIR_CAPACITY = 2, CANDIDATE_CAPACITY = 8;
    let results = null, resultsIncomplete = false;
    let verifiedReportCount = 0, mismatchReportCount = 0, missedReportCount = 0;
    const verifiedReportIds = {write() {}}, mismatchReportIds = {write() {}}, missedReportIds = {write() {}};
    function updateTimelineChart() {} function describeSelection() {}
    ${readClosure('examples/deck/spatial-analysis-showcase/src/scenes/earth/storm-warning-verification.compute.ts', '  function handleSummary(', '  const statusReader =')}
    return {handleSummary, state() {return {results, verifiedReportCount, mismatchReportCount, missedReportCount};}};
  `
  );
  const reader = factory(
    {
      setReadout: (key: string, value: unknown) => readouts.set(key, value),
      setChart: (key: string, value: {values?: ArrayLike<number>}) => charts.set(key, value),
      requestLayers() {}
    },
    String,
    String,
    (values: number[]) => values[0]
  );
  // Two reports, each with a compatible warning; report 0 has a retained 10-minute lead.
  // An omitted earlier warning could increase it. A missing report-1 pair cannot prove a miss.
  function summary(loss: boolean) {
    const bytes = new ArrayBuffer((5 + 8 + 2) * 4);
    const words = new Uint32Array(bytes);
    words.set([
      loss && cause === 'candidate' ? 1 : 2,
      Number(loss && cause !== 'candidate'),
      loss && cause !== 'candidate' ? 3 : loss ? 1 : 2,
      loss && cause !== 'candidate' ? 3 : 2,
      Number(loss && cause !== 'final')
    ]);
    new Float32Array(bytes).set([2, 600, 0, 1, loss ? 0 : 2, 1200, 1, 1], 5);
    words.set([1, loss ? 0 : 1], 13);
    return bytes;
  }
  reader.handleSummary(summary(false));
  expect(charts.get('leadChart')?.values?.length).toBeGreaterThan(0);
  reader.handleSummary(summary(true));
  if (cause === 'both') expect(readouts.get('pairs')).toMatch(/candidates, pairs/);
  expect(readouts.get('pairs')).toMatch(/OVERFLOW:.*incomplete/);
  expect(readouts.get('verifiedShare')).toMatch(/at least 1/);
  expect(readouts.get('verifiedShare')).not.toMatch(/%/);
  expect(readouts.get('unwarned')).toMatch(/unavailable/);
  expect(readouts.get('medianLead')).toMatch(/unavailable/);
  expect(reader.state().results).toBeNull();
  expect(reader.state().missedReportCount).toBe(0);
  expect(reader.state().mismatchReportCount).toBe(0);
  expect(charts.has('leadChart')).toBe(true);
  expect(charts.get('leadChart')?.values).toHaveLength(0);
  reader.handleSummary(summary(false));
  expect(readouts.get('verifiedShare')).toBe('2 of 2 reports (100%)');
  expect(reader.state().results).not.toBeNull();
  expect(reader.state().verifiedReportCount).toBe(2);
});

it('topological production decoder treats candidate-only loss as lower bounds, with an exact control', () => {
  const readouts = new Map<string, unknown>();
  const factory = new Function(
    'ctx',
    `
    const SUMMARY_WORDS = 10, maximumLeft = 2;
    let leftCounts, antiFlags, pairs, maximumCount, pairResultsIncomplete, antiResultsIncomplete;
    const formatInteger = String;
    const formatGPUSpatialRelate = String;
    const describePredicate = () => 'intersects';
    const describeOverflow = words => words[7] ? 'OVERFLOW: candidates' : 'no overflow';
    function refreshSelection() {} function refreshStatus() {}
    ${readClosure('examples/deck/spatial-analysis-showcase/src/scenes/joins/topological-relations.compute.ts', '  function handleSummary(', '  function describePredicate(')}
    return handleSummary;
  `
  );
  const decode = factory({
    options: {},
    setReadout: (key: string, value: unknown) => readouts.set(key, value)
  });
  // Closed form: two coincident left points x two coincident right points = four true pairs.
  const source = {
    pairCapacity: 4,
    leftCount: 2,
    rightCount: 2,
    combo: 'tracts-self',
    hasMatrix: false,
    withWeights: false,
    prepared: {encodedBuildCount: 1},
    frames: 1
  };
  const words = new Uint32Array(29);
  words.set([1, 0, 1, 0, 4, 0, 0, 1, 0, 0]);
  decode(source, words.buffer);
  expect(readouts.get('pairs')).toMatch(/at least 1/);
  expect(readouts.get('candidates')).not.toMatch(/%/);
  expect(readouts.get('candidates')).not.toMatch(/4 retained/);
  expect(readouts.get('anti')).toBe('Unknown due to incomplete results (0 of 2 unmatched)');
  words.set([4, 0, 4, 0, 4, 0, 0, 0, 0, 0]);
  words.set([0, 0, 1, 1], 10);
  words.set([0, 1, 0, 1], 14);
  decode(source, words.buffer);
  expect(readouts.get('pairs')).toBe('4 pairs: 2 of 2 left, 2 of 2 right');
  expect(readouts.get('candidates')).toMatch(/100%/);
  expect(readouts.get('anti')).toMatch(/exactly the complement/);
  for (const uncertaintyWord of [3, 9]) {
    words[uncertaintyWord] = 1;
    decode(source, words.buffer);
    expect(readouts.get('anti')).toBe('Unknown due to uncertain predicates');
    words[uncertaintyWord] = 0;
  }
});

it('explorer compares the anti join only with certified inner and anti results', () => {
  const readouts = new Map<string, string>();
  const pairsReadout = Object.fromEntries(
    ['pairs', 'anti', 'candidates', 'matrices', 'flags', 'engine', 'reuse'].map(key => [
      key,
      {setValue: (value: string) => readouts.set(key, value)}
    ])
  );
  const decode = new Function(
    'pairsReadout',
    `
    const destroyed = false, SUMMARY_WORDS = 10, zipCount = 1, predicate = 'intersects';
    const formatCount = String, formatGPUSpatialRelate = String, describePredicate = () => predicate;
    const invalidateEveryFrame = false;
    const view = {write() {}};
    const active = {
      pairCapacity: 1, scene: {shapes: [0, 1]}, matchedFlags: view, unmatchedFlags: view,
      rightCounts: view, prepared: {encodedBuildCount: 1}, frames: 1, engineText: ''
    };
    ${readClosure('examples/deck/spatial-analysis-explorer/modes/relate-mode.ts', '    const handleSummary =', '    /** Times the active graph')}
    return bytes => handleSummary(active, bytes);
    `
  )(pairsReadout);
  // The single inner pair uses left row 0. The anti join reports the other row, 1.
  const words = Uint32Array.of(1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0);
  decode(words.buffer);
  expect(readouts.get('anti')).toBe('1 of 2 · equals the complement of the pairs');
  for (const uncertaintyWord of [3, 9]) {
    words[uncertaintyWord] = 1;
    words[5] = 0;
    decode(words.buffer);
    expect(readouts.get('anti')).toBe('Unknown due to uncertain predicates');
    words[uncertaintyWord] = 0;
  }
});

it.each([
  'within',
  'relate',
  'dwithin'
] as const)('topological %s snippet parses and uses separate inner and anti status destinations', predicate => {
  const snippet = topologicalRelations.snippet;
  if (typeof snippet !== 'function') throw new Error('Expected a parameterized example');
  const source = snippet({
    predicate,
    combo: 'tracts-self',
    pattern: 'queen',
    distance: 100,
    engine: 'auto',
    matrix: true,
    show: 'anti',
    ramp: 'viridis',
    opacity: 1,
    showRight: true
  });
  const {diagnostics} = transpileModule(source, {reportDiagnostics: true});
  expect(diagnostics).toEqual([]);
  expect(source).toContain("...common,\n    id: 'inner'");
  expect(source).toContain("...common,\n    id: 'anti',\n    how: 'anti'");
  expect(source).toContain('status: {count: pairCount, overflow: pairOverflow');
  expect(source).toContain(
    'status: {count: antiCount, overflow: antiOverflow, candidateOverflow: antiCandidateOverflow}'
  );
  expect(source).toContain('status: {uncertainCount: antiUncertainCount}');
});

it('footprint provenance shows an edge only for complete, nonempty components', () => {
  const decode = new Function(
    'decodeGPUOffsetCurveComponentsStatus',
    'getGPUOffsetCurveRowsPerVertex',
    `
    const outputPositions = 0, topology = 1, statusIndex = 2;
    const byteSizes = [16, 16, 32, 8, 4, 8];
    const ownedBuffers = ['selected-component-points', 'selected-components', 'component-status',
      'raw-offset-curve', 'offset-row-counts', 'selected-source-edge-ids'].map(id => ({id}));
    const options = {join: 'round', quadSegments: 8, offsetDistance: 1, curveSide: 'left'};
    const footprint = {ringOffsets: [0, 2], projection: {unproject: (x, y) => [x, y]}};
    return bytes => {
      const variant = {};
      ${readClosure('examples/deck/spatial-analysis-showcase/src/scenes/geometry/footprint-shape-lab.compute.ts', '              const data = new Float32Array(bytes);\n              const indexes =', "            } else if (options.footprint === 'convex')")}
      return variant.result;
    };
    `
  )(decodeGPUOffsetCurveComponentsStatus, getGPUOffsetCurveRowsPerVertex);
  const words = new Uint32Array(21);
  new Float32Array(words.buffer).set([1, 2, 3, 4]);
  words.set([0, 2, 0, 0], 4);
  words[19] = 77; // Leave populated provenance behind for both withheld and empty frames.
  words.set([1, 2, 1, 2, 0, 0, 0, 0], 8);
  expect(decode(words.buffer).detail).toContain('first selected source-edge ID 77');
  words.set([0, 0, 1, 2, 0, 0, 1, 0], 8);
  const withheld = decode(words.buffer);
  expect(withheld.status).toMatch(/Withheld.*iterationLimit 1/);
  expect(withheld.detail).not.toContain('source-edge ID');
  words.fill(0, 8, 16);
  const empty = decode(words.buffer);
  expect(empty.status).toMatch(/0 written/);
  expect(empty.detail).not.toContain('source-edge ID');
});

it('RT-137 catalog crossing bound closes each ring without connecting separate rings', () => {
  const source = readClosure(
    'modules/kepler/src/analysis/catalog-platform4-raster.ts',
    '    let crossingCapacity = 64;',
    '    const totals ='
  );
  const estimate = new Function('joinInputs', 'cellSize', `${source}; return crossingCapacity;`);
  // Two vertical sides cross 100 rows each. Four rounding allowances plus the base 64.
  expect(estimate({positions: [0, 0, 1, 0, 1, 100, 0, 100], ringOffsets: [0, 4]}, 1)).toBe(268);
  // Separate 1x1 squares 1000 rows apart must not acquire artificial inter-ring edges.
  expect(
    estimate(
      {
        positions: [0, 0, 1, 0, 1, 1, 0, 1, 0, 1000, 1, 1000, 1, 1001, 0, 1001],
        ringOffsets: [0, 4, 8]
      },
      1
    )
  ).toBe(76);
});

it('RT-137 actual showcase publisher withholds failed joins, distinguishes complete empty, and recovers', () => {
  const readouts = new Map<string, unknown>();
  // Dependencies captured by the production closure.
  const body = readClosure(
    'examples/deck/spatial-analysis-showcase/src/scenes/joins/raster-join.compute.ts',
    '  function handleSummary(',
    '  /** Makes a built variant'
  );
  const run = new Function(
    'ctx',
    `const zoneCount=1, pointCount=1; let rasterCounts, exactCounts, valueRange; const formatInteger=String; const zoneValues={write(){}}; ${body}; return handleSummary;`
  )({
    options: {metric: 'count'},
    setReadout: (key: string, value: unknown) => readouts.set(key, value),
    setLegendExtent() {},
    requestLayers() {}
  });
  // Four zone columns, seven status words, then mismatch, boundary, exact ID and extent membership.
  const words = new Uint32Array(19);
  words.set([0, 0, 0xffffffff, 1], 11); // The point is in the raster but matches no zone.
  new Float32Array(words.buffer).set([0, 0, 1, 1], 15);
  words[6] = 1;
  words[9] = 1;
  run(words.buffer, 1, 1, 1);
  for (const key of ['joined', 'mismatch', 'explained', 'bound', 'difference', 'boundaryJoined'])
    expect(readouts.get(key), key).toMatch(/unavailable/);
  words[6] = 0;
  words[9] = 0;
  run(words.buffer, 1, 1, 1);
  expect(readouts.get('joined')).toMatch(/^0 by raster, 0 exact/);
  expect(readouts.get('bound')).toMatch(/^1 of 1/);
  words[0] = 1;
  words[13] = 0;
  words[2] = 1;
  run(words.buffer, 1, 1, 1);
  expect(readouts.get('joined')).toMatch(/^1 by raster, 1 exact/);
});

it('RT-137 explorer actual callback suppresses guarantee claims on loss and restores complete empty', () => {
  const values = new Map<string, string>();
  const readout = (key: string) => ({setValue: (value: string) => values.set(key, value)});
  const names = [
    'crossing',
    'inside',
    'boundary',
    'exact',
    'boundHolds',
    'worstShare',
    'worstDifference'
  ];
  const body = readClosure(
    'examples/deck/spatial-analysis-explorer/modes/raster-join-mode.ts',
    '    function refreshReadouts(',
    '    const readSummary ='
  );
  const run = new Function(
    ...names.map(name => `${name}Readout`),
    `const zoneCount=1, pointCount=1, RASTER_WIDTH=2, RASTER_HEIGHT=100, CROSSING_CAPACITY=167, GPU_POLYGON_RASTERIZATION_NO_ZONE=0xffffffff; const parking={positions: [0.5,50.5]}, formatCount=String, getZoneName=()=>'zone'; ${body}; return refreshReadouts;`
  )(...names.map(readout));
  const source = {
    extent: [0, 0, 1, 1],
    overflow: 1,
    incomplete: 1,
    counts: [0xffffffff],
    boundaryCounts: [0],
    exactZones: [0xffffffff],
    pointZones: [0xffffffff],
    pointBoundaryMask: [0],
    unassignedBoundaryCount: 0,
    outsideCount: 0,
    crossingCount: 200
  };
  run(source);
  for (const name of names.slice(1)) expect(values.get(name), name).toMatch(/unavailable/);
  expect(values.get('exact')).not.toMatch(/VIOLATES/);
  source.overflow = 0;
  source.incomplete = 0;
  source.counts = [0];
  run(source);
  expect(values.get('exact')).toMatch(/0 points differ.*as guaranteed/);
  expect(values.get('boundHolds')).toMatch(/^1 of 1/);
});

it('RT-137 loss bindings and copy lists include full, counts-only, explorer and lens routes', () => {
  const paths = [
    'examples/deck/spatial-analysis-showcase/src/scenes/joins/raster-join.compute.ts',
    'examples/deck/spatial-analysis-explorer/modes/raster-join-mode.ts',
    'modules/lenses/src/computation-lenses/scenarios/gpu-raster/raster-join.example.ts',
    'modules/kepler/src/analysis/catalog-platform4-raster.ts'
  ];
  for (const path of paths) {
    const source = readFileSync(path, 'utf8');
    expect(source).toContain('producerOverflow:');
    expect(source).toMatch(/status:\s*\{\s*incomplete[}:]/);
  }
  const showcase = readFileSync(paths[0], 'utf8');
  expect(showcase.match(/new GPURasterJoin\(/g)).toHaveLength(2);
  expect(showcase.match(/producerOverflow:/g)).toHaveLength(3);
  expect(showcase).toContain('{buffer: incomplete, size: 4}');
  const explorer = readFileSync(paths[1], 'utf8');
  expect(explorer).toContain('[incomplete, 1]');
  expect(explorer).toContain('incomplete: take(1)[0]');
});
