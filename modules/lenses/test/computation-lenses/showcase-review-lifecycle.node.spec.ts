// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';

const root = 'examples/deck/spatial-analysis-showcase/src/scenes/';
function code(path: string, start: string, end: string): string {
  const source = readFileSync(path, 'utf8');
  const first = source.indexOf(start),
    last = source.indexOf(end, first);
  expect(first).toBeGreaterThan(0);
  expect(last).toBeGreaterThan(first);
  return transpileModule(source.slice(first, last), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
}
function scoreProbe() {
  const path = `${root}networks/job-accessibility.compute.ts`;
  const holder = code(path, '  function createScoreHolder()', '  // ---- Timing');
  const commit = code(path, '  function commitGraphs(next:', '    built = next;');
  return new Function(`
    let built = {}, snapReader, companionReader, destroyed = false;
    let comparisonGeneration = 1, activeScoreGeneration = -1, encodedScoreGeneration = 1;
    let scoresComplete = 'certificate-A', snapOverflow = 'snap-A';
    const cumulativeBuffer = {}, gravityBuffer = {}, twoStepBuffer = {}, resources = {};
    const nodeCount = 2, roadNodeCount = 2; let latestScores, scoresReady;
    const comparisonSnapshots = new Map(); const comparisonKey = () => 'current';
    const ctx = {options: {measure: 'cumulative'}, setReadout() {}, requestLayers() {}};
    const publishComparisons = () => {}, formatInteger = String, getScoreBuffer = () => cumulativeBuffer;
    const copies = []; let instance;
    class SummaryReader {
      isPending = false; stale = false;
      constructor(resources, id, sources, callback) {this.sources = sources; this.callback = callback; instance = this;}
      request() {this.stale = true; this.flush();}
      flush() {if (!this.isPending && this.stale) {copies.push({sources: this.sources, generation: activeScoreGeneration}); this.isPending = true; this.stale = false;}}
      stop() {this.stale = false;}
    }
    ${holder}
    const scoreRing = createScoreHolder();
    ${commit}
    return {copies, request: () => scoreRing.request({}), flush: () => scoreRing.flush({}),
      finish() {const bytes = new ArrayBuffer(32); new Float32Array(bytes).fill(1, 0, 6); new Uint32Array(bytes)[6] = 1;
        instance.callback(bytes); instance.isPending = false;},
      rebuild() {commitGraphs({}); scoresComplete = 'certificate-B'; snapOverflow = 'snap-B';},
      generation(current, encoded) {comparisonGeneration = current; encodedScoreGeneration = encoded;}};
  `)();
}
it('B1 primary score reader follows certificate buffers across two complete builds', () => {
  const probe = scoreProbe();
  probe.request();
  probe.finish();
  probe.rebuild();
  probe.request();
  probe.finish();
  expect(probe.copies[1].sources[3].buffer).toBe('certificate-B');
  expect(probe.copies[1].sources[4].buffer).toBe('snap-B');
});
it('B2 queued score B cannot acquire generation C before C is encoded', () => {
  const probe = scoreProbe();
  probe.request();
  probe.generation(2, 2);
  probe.request();
  probe.generation(3, 2);
  probe.finish();
  probe.flush();
  expect(probe.copies).toHaveLength(1);
  probe.generation(3, 3);
  probe.request();
  expect(probe.copies.at(-1).generation).toBe(3);
});
it('B3 comparison snapshots exclude stop nodes from intersection counts', () => {
  const body = code(
    `${root}networks/job-accessibility.compute.ts`,
    '    const process = (bytes: ArrayBuffer)',
    '\n    return {'
  );
  const snapshots = new Map();
  const run = new Function(
    'comparisonSnapshots',
    `
    const destroyed = false, activeScoreGeneration = 1, comparisonGeneration = 1, nodeCount = 3, roadNodeCount = 2;
    let scoresReady, latestScores; const cumulativeBuffer = {}, twoStepBuffer = {};
    const ctx = {options: {}, requestLayers() {}, setReadout() {}};
    const comparisonKey = () => 'current', publishComparisons = () => {}, getScoreBuffer = () => cumulativeBuffer, formatInteger = String;
    ${body} return process;
  `
  )(snapshots);
  const bytes = new ArrayBuffer(44);
  new Float32Array(bytes).set([1, 2, 999999, 1, 2, 999999, 1, 2, 999999]);
  new Uint32Array(bytes)[9] = 1;
  run(bytes);
  expect(Array.from(snapshots.get('current').cumulative)).toEqual([1, 2]);
  expect(Array.from(snapshots.get('current').twoStep)).toEqual([1, 2]);
});
it('S3 rail planning decoder uses the actual readback source order and distinct completion neighbors', () => {
  const source = readFileSync(`${root}networks/rail-support-planning.compute.ts`, 'utf8');
  const start = source.indexOf('  const reader = new SummaryReader('),
    end = source.indexOf('\n  );', start);
  const body = transpileModule(source.slice(start, end + 5), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, unknown>();
  const reader = new Function(
    'ctx',
    `
    const resources = {}, destroyed = false, pendingGeneration = 1, generation = 1, nodeCount = 3, HUB_NAMES = ['Hub'];
    let outputReady, lastAssignments, lastCosts, lastWeightedReach;
    const assignmentBuffer = Uint32Array.of(0,0,0), serviceCostBuffer = Float32Array.of(10,20,30);
    const serviceStatusBuffer = Uint32Array.of(1), countsBuffer = Uint32Array.of(3), matrixStatusBuffer = Uint32Array.of(1);
    const weightOffsetsBuffer = Uint32Array.of(0,1,2,3), weightStatusBuffer = Uint32Array.of(0), weightCompleteBuffer = Uint32Array.of(0);
    const weightedReachBuffer = Float32Array.of(1,2,3);
    class SummaryReader {constructor(resources, name, sources, callback) {this.sources = sources; this.callback = callback;}}
    ${body} return {reader, complete() {weightCompleteBuffer[0] = 1;}};
  `
  )({
    options: {direction: 'inbound'},
    setReadout: (key: string, value: unknown) => readouts.set(key, value),
    requestLayers() {}
  });
  const decode = () => {
    const sources: {buffer: Uint32Array | Float32Array; size: number}[] = reader.reader.sources;
    const bytes = new Uint8Array(sources.reduce((size, source) => size + source.size, 0));
    let offset = 0;
    for (const source of sources) {
      bytes.set(
        new Uint8Array(source.buffer.buffer, source.buffer.byteOffset, source.size),
        offset
      );
      offset += source.size;
    }
    reader.reader.callback(bytes.buffer);
  };
  decode();
  expect(readouts.get('weightedPairs')).toBe('withheld');
  reader.complete();
  decode();
  expect(readouts.get('weightedPairs')).toBe('3 station-desk pairs · total weight 6.0');
});
it('B4 allocation publishers distinguish incomplete routing from infeasibility', () => {
  const body = code(
    `${root}networks/rail-support-allocation.compute.ts`,
    '  function publishPlacement()',
    '  function publishRational()'
  );
  const scenario = code(
    `${root}networks/rail-support-allocation.compute.ts`,
    '  function publishScenario()',
    '  const summaryBuffers ='
  );
  const values = new Map();
  const probe = new Function(
    'ctx',
    `
    const status = Uint32Array.of(1,1,1,8);
    const placementResults = {costsConverged: false, pmedianStatus: status, backupStatus: status, dispersionStatus: status,
      pmedianObjective: 1, backupObjective: 1, dispersionObjective: 1};
    const lastPmedianSelected = [1,1], lastBackupSelected = [1,1], lastDispersionSelected = [1,1], HUB_NAMES = ['A','B'], demandCount = 3;
    const parsedScenarioResults = Array.from({length: 3}, () => ({costs: new Float32Array(3), facilities: new Uint32Array(3), covered: 3, uncovered: 0, invalid: 0}));
    let outputReady, lastScenarioCosts, lastScenarioFacilities;
    ${body} ${scenario} return {publishPlacement, publishScenario, complete(feasible) {placementResults.costsConverged = true; status[1] = Number(feasible);}};
  `
  )({
    options: {objective: 'pmedian'},
    setReadout: (key: string, value: unknown) => values.set(key, value)
  });
  probe.publishPlacement();
  probe.publishScenario();
  expect(values.get('optimizerStatus')).toMatch(/incomplete|withheld/);
  expect(values.get('scenarioSummary')).toMatch(/incomplete|withheld/);
  probe.complete(false);
  probe.publishPlacement();
  probe.publishScenario();
  expect(values.get('optimizerStatus')).toMatch(/no feasible incumbent/);
  expect(values.get('scenarioSummary')).toBe('no feasible selected pair');
  probe.complete(true);
  probe.publishPlacement();
  probe.publishScenario();
  expect(values.get('optimizerStatus')).toMatch(/feasible incumbent/);
  expect(values.get('scenarioSummary')).toMatch(/3 units within/);
});
it.each([
  'snapshot-first',
  'summary-first'
])('B7 pairs weights results from one submission (%s)', order => {
  const body = code(
    `${root}weights/spatial-weights.compute.ts`,
    '  const requestReadbackIfNeeded =',
    '  const copyCsr ='
  );
  const probe = new Function(`
    let kitCallback, snapshotCallback; const publications = [];
    const space = {isReading: () => false, requestReadback(encoder, onKit, onSnapshot) {kitCallback = onKit; snapshotCallback = onSnapshot;}};
    const world = {space, kitSummary: {producerCandidateOverflow: true}, snapshot: null};
    const ctx = {options: {}, requestLayers() {}};
    let readbackStale = true, focusStale = false, referenceStale = false, optionRevision = 1;
    const updateReadouts = () => {}, publishAnnotations = () => {};
    function handleSnapshot(target, snapshot) {publications.push([target.kitSummary.producerCandidateOverflow, snapshot.id]);}
    ${body}
    return {request() {readbackStale = true; requestReadbackIfNeeded({});}, publications,
      kit(loss) {kitCallback({producerCandidateOverflow: loss});}, snapshot(id) {snapshotCallback({id});}};
  `)();
  for (const loss of [true, false]) {
    probe.request();
    const count = probe.publications.length;
    if (order === 'snapshot-first') {
      probe.snapshot(loss ? 'lost' : 'complete');
      expect(probe.publications).toHaveLength(count);
      probe.kit(loss);
    } else {
      probe.kit(loss);
      expect(probe.publications).toHaveLength(count);
      probe.snapshot(loss ? 'lost' : 'complete');
    }
    expect(probe.publications.at(-1)).toEqual([loss, loss ? 'lost' : 'complete']);
  }
});
it('B5 old storm readback of the same variant cannot restore a newer verification', () => {
  const path = `${root}earth/storm-warning-verification.compute.ts`,
    source = readFileSync(path, 'utf8');
  const marker = source.indexOf('built.reader = new SummaryReader');
  const first = source.indexOf('bytes => {', marker),
    end = source.indexOf('\n    });', first);
  const body = transpileModule(`const callback = ${source.slice(first, end)};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const probe = new Function(`
    const built = {key: 'intersects', readRevision: 1}, current = built, destroyed = false, verificationRevision = 2;
    const ctx = {options: {spatialTest: 'intersects'}}; let published = 0;
    const handleSummary = () => published++;
    ${body} return {callback, count() {return published;}};
  `)();
  probe.callback(new ArrayBuffer(0));
  expect(probe.count()).toBe(0);
});
it('B5 storm invalidation clears prior classifications before a new GPU submission', () => {
  const path = `${root}earth/storm-warning-verification.compute.ts`,
    source = readFileSync(path, 'utf8');
  const invalidator = source.includes('  function invalidateVerification()')
    ? code(path, '  function invalidateVerification()', '  /** Makes a compiled join variant')
    : '';
  const commit = code(path, '  function commitVariant(', '  const graphSwap =');
  const probe = new Function(`
    let current, verifyDirty = false, verificationRevision = 0, resultsIncomplete = false, results = {};
    let verifiedReportCount = 1, mismatchReportCount = 1, missedReportCount = 1;
    const ctx = {setReadout() {}, setChart() {}, requestLayers() {}};
    function updateTimelineChart() {} function describeSelection() {}
    ${invalidator} ${commit} commitVariant({});
    return {resultsIncomplete, results, verifiedReportCount, mismatchReportCount, missedReportCount};
  `)();
  expect(probe.resultsIncomplete).toBe(true);
  expect(probe.results).toBeNull();
  expect(probe.missedReportCount).toBe(0);
});
it('B2 companion flush also waits for the generation actually encoded', () => {
  const source = readFileSync(`${root}networks/job-accessibility.compute.ts`, 'utf8');
  const marker = source.indexOf('      snapReader?.flush(commandEncoder);');
  const end = source.indexOf('\n    },', marker);
  const body = transpileModule(source.slice(marker, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const probe = new Function(`
    const commandEncoder = {}, snapReader = {flush() {}}, scoreRing = {flush() {}};
    let encodedCompanionGeneration = 2, comparisonGeneration = 3, companionScoreGeneration = 1;
    const copies = []; const companionReader = {isPending: false, flush() {copies.push(companionScoreGeneration);}};
    function flush() {${body}} return {flush, copies, encode() {encodedCompanionGeneration = 3;}};
  `)();
  probe.flush();
  expect(probe.copies).toEqual([]);
  probe.encode();
  probe.flush();
  expect(probe.copies).toEqual([3]);
});
it('B4 rational tooltip cannot expose an uncertified placement cost', () => {
  const source = readFileSync(`${root}networks/rail-support-allocation.compute.ts`, 'utf8');
  const marker = source.indexOf("label: 'Facility scenario nearest cost'");
  const start = source.indexOf('value:', marker),
    end = source.indexOf('\n          },', start);
  const value = source.slice(start + 6, end).trim();
  expect(
    new Function(
      `const outputReady = false, lastScenarioCosts = [7], nearest = 0; return (${value});`
    )()
  ).toMatch(/withheld/);
});
it('B2 explorer invalidation clears both numerical captions while the current scores are pending', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/accessibility-mode.ts',
    'utf8'
  );
  const marker = source.indexOf('const writeScoring =');
  const start = source.indexOf('      scoring.write(', marker),
    end = source.indexOf('\n    };', start);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const values = new Map<string, string>([
    ['reached', '10'],
    ['range', '100']
  ]);
  new Function(
    'reachedReadout',
    'scoreRangeReadout',
    `
    const scoring = {write() {}}, getGPUNetworkAccessibilityParameterValues = () => [], settings = {};
    let scoresReady = true, scoreGeneration = 1, scoreDirty = false, scoreReadbackWanted = false;
    const context = {updateLayers() {}}, scheduleTiming = () => {};
    ${body}
  `
  )(
    {setValue: (value: string) => values.set('reached', value)},
    {setValue: (value: string) => values.set('range', value)}
  );
  expect(values.get('reached')).toBe('pending');
  expect(values.get('range')).toBe('pending');
});
