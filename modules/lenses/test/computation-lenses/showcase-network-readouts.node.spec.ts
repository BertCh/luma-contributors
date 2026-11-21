// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';
import {
  createApplicationContributor,
  createApplicationScorePublisher,
  readApplicationScorePacket
} from './showcase-accessibility-fixture';

const sceneRoot = 'examples/deck/spatial-analysis-showcase/src/scenes/networks/';

it('job score decoder withholds unfinished or short-horizon scores, clears comparisons, and recovers', () => {
  const source = readFileSync(`${sceneRoot}job-accessibility.compute.ts`, 'utf8');
  const start = source.indexOf('    const process = (bytes: ArrayBuffer) => {');
  const end = source.indexOf('\n    return {', start);
  expect(start).toBeGreaterThan(0);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, unknown>();
  const factory = new Function(
    'ctx',
    `
    let destroyed = false, activeScoreGeneration = 1, comparisonGeneration = 1;
    const nodeCount = 5, roadNodeCount = 5;
    let latestScores = null, scoresReady = true;
    const comparisonSnapshots = new Map([['current', {cumulative: new Float32Array(5)}]]);
    function comparisonKey() {return 'current';} function publishComparisons() {}
    const cumulativeBuffer = {}, twoStepBuffer = {}; function getScoreBuffer() {return cumulativeBuffer;}
    const formatInteger = String;
    ${body}
    return {process, state() {return {latestScores, scoresReady, size: comparisonSnapshots.size};}};
  `
  );
  const reader = factory({
    options: {measure: 'cumulative', transit: false},
    setReadout: (key: string, value: unknown) => readouts.set(key, value),
    requestLayers() {}
  });
  // Hand-worked five-node chain with unit costs: horizon 1 retains [0,1]; request 4 needs all five.
  // The GPU consumer supplies complete=0 for either an unfinished search or an insufficient horizon.
  for (const complete of [0, 0, 1]) {
    const bytes = new ArrayBuffer((15 + 2) * 4);
    new Float32Array(bytes).fill(complete ? 5 : 2, 0, 15);
    new Uint32Array(bytes).set([complete, 0], 15);
    reader.process(bytes);
    if (!complete) {
      expect(reader.state().latestScores).toBeNull();
      expect(reader.state().scoresReady).toBe(false);
      expect(reader.state().size).toBe(0);
      expect(readouts.get('median')).toMatch(/withheld/);
    } else {
      expect(reader.state().scoresReady).toBe(true);
      expect(reader.state().size).toBe(1);
      expect(readouts.get('median')).toBe('5');
    }
  }
});

it('both job matrices and explorer carry their own convergence and horizons into score certificates', () => {
  const job = readFileSync(`${sceneRoot}job-accessibility.compute.ts`, 'utf8');
  expect(job).toMatch(/costsComplete: view\(matrixConverged/);
  expect(job).toMatch(/costsComplete: view\(comparisonConverged/);
  expect(job).toMatch(/costLimit: costLimitParameter.importToGraph\(scoreGraph\)/);
  expect(job).toMatch(/costLimit: costLimitParameter.importToGraph\(comparisonScoreGraph\)/);
  expect(job).not.toContain('Math.min(options.thresholdMinutes, options.matrixLimitMinutes)');
  const explorer = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/accessibility-mode.ts',
    'utf8'
  );
  expect(explorer).toMatch(/costsComplete: importGraphBuffer\(\s*scoreGraph,\s*'matrix-converged'/);
  expect(explorer).toMatch(/costLimit: costLimit.importToGraph\(scoreGraph\)/);
  expect(explorer).toContain('sourceBuffer: scoresComplete');
  expect(explorer).toContain('instanceCount: scoresReady ? segmentCount : 0');
});

it('rail planning reads the weights completion word and supplies the unbounded matrix horizon in minutes', () => {
  const source = readFileSync(`${sceneRoot}rail-support-planning.compute.ts`, 'utf8');
  expect(source).toMatch(/costsComplete: view\(matrixStatusBuffer/);
  expect(source).toMatch(/costLimit: view\(matrixHorizonMinutesBuffer/);
  expect(source).toMatch(/complete: view\(weightCompleteBuffer/);
  expect(source).toContain('const weightsComplete = words[weightStart + nodeCount + 2] === 1');
  expect(source).toContain('&& weightsComplete');
});

it('job companion score readout removes uncertified comparison data and recovers', () => {
  const source = readFileSync(`${sceneRoot}job-accessibility.compute.ts`, 'utf8');
  const marker = source.indexOf('`access-comparison-${rowCount}');
  const start = source.indexOf('      bytes => {', marker);
  const end = source.indexOf('\n    );', start);
  expect(start).toBeGreaterThan(0);
  const body = transpileModule(`const callback = ${source.slice(start, end)};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const reader = new Function(`
    const destroyed = false, companionScoreGeneration = 1, comparisonGeneration = 1;
    const nodeCount = 5, roadNodeCount = 2, companionMode = true;
    const comparisonSnapshots = new Map([[true, {}]]);
    const comparisonKey = value => value;
    function publishComparisons() {}
    ${body}
    return {callback, size() {return comparisonSnapshots.size;}, scores() {return comparisonSnapshots.get(true)?.cumulative;}};
  `)();
  const bytes = new ArrayBuffer(28);
  new Float32Array(bytes).fill(2, 0, 5);
  reader.callback(bytes);
  expect(reader.size()).toBe(0);
  new Uint32Array(bytes)[5] = 1;
  reader.callback(bytes);
  expect(reader.size()).toBe(1);
  expect(Array.from(reader.scores())).toEqual([2, 2]);
});

it('job production comparison publisher clears the prior competition chart when certification is lost', () => {
  const source = readFileSync(`${sceneRoot}job-accessibility.compute.ts`, 'utf8');
  const start = source.indexOf('  const publishComparisons =');
  const end = source.indexOf('  const writeWeights =', start);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const charts = new Map<string, {values?: number[]; charts?: unknown[]}>();
  const reader = new Function(
    'ctx',
    `
    const comparisonSnapshots = new Map([['current', {cumulative: new Float32Array(5), twoStep: new Float32Array(5)}]]);
    const comparisonKey = value => value === undefined ? 'current' : value;
    const ACCESS_CLASSES = {cumulative: {}, twoStep: {}};
    const countClasses = () => [5], cumulativeClassLabels = () => ['jobs'];
    ${body}
    return {publishComparisons, clear() {comparisonSnapshots.clear();}};
  `
  )({
    setChart: (key: string, value: {values?: number[]; charts?: unknown[]}) =>
      charts.set(key, value)
  });
  reader.publishComparisons();
  expect(charts.get('competitionComparison')?.charts).toHaveLength(2);
  reader.clear();
  reader.publishComparisons();
  expect(charts.get('competitionComparison')?.values).toEqual([]);
});

it('NET-101 actual application constructor and publisher preserve the incumbent certificate seam', () => {
  for (const route of ['job', 'explorer'] as const) {
    const source = readFileSync(
      route === 'job'
        ? `${sceneRoot}job-accessibility.compute.ts`
        : 'examples/deck/spatial-analysis-explorer/modes/accessibility-mode.ts',
      'utf8'
    );
    const view = (buffer: string) => buffer;
    const instance = createApplicationContributor(source, {
      GPUNetworkAccessibility: class {
        constructor(readonly props: Record<string, unknown>) {}
      },
      view,
      importGraphBuffer: (_graph: unknown, _name: unknown, buffer: unknown) => buffer,
      scoreGraph: {},
      nodeCount: 2,
      rowCount: 1,
      matrixLength: 2,
      matrix: 'costs',
      matrixBuffer: 'costs',
      matrixConverged: 'converged',
      costLimit: {importToGraph: () => 'horizon'},
      costLimitParameter: {importToGraph: () => 'horizon'},
      scoring: {importToGraph: () => 'parameters'},
      scoresComplete: 'complete',
      opportunityWeights: 'supply',
      demandBuffer: 'demand',
      cumulative: 'cumulative',
      cumulativeBuffer: 'cumulative',
      gravity: 'gravity',
      gravityBuffer: 'gravity',
      twoStep: 'two-step',
      twoStepBuffer: 'two-step',
      facilityRatios: 'ratios',
      ratiosBuffer: 'ratios'
    });
    expect(instance.props['costsComplete']).toBe('converged');
    expect(instance.props['status']).toEqual({complete: 'complete'});
    expect(instance.props['catchment']).toEqual({
      demand: 'demand',
      output: 'two-step',
      ratios: 'ratios'
    });
    const publisher = createApplicationScorePublisher(route, source);
    for (const complete of [0, 1, 0, 1]) {
      const bytes = new ArrayBuffer(route === 'job' ? 32 : 12);
      // Closed form for retained supply 12 and reached demand 3: ratio 4; score [4,0].
      new Float32Array(bytes).set(route === 'job' ? [12, 0, 12, 0, 4, 0] : [4, 0]);
      new Uint32Array(bytes)[route === 'job' ? 6 : 2] = complete;
      publisher.publish(route === 'job' ? bytes : new Uint8Array(bytes));
      expect(publisher.ready()).toBe(Boolean(complete));
      if (!complete) expect(publisher.values.get('reached')).toMatch(/withheld/);
      else
        expect(publisher.values.get('reached')).toBe(
          route === 'job' ? '50% of intersections' : '1 of 2'
        );
    }
  }
});

it('B8 actual application copy sources restore the two-step magnitude and reject a gravity substitution', async () => {
  for (const route of ['job', 'explorer'] as const) {
    const source = readFileSync(
      route === 'job'
        ? `${sceneRoot}job-accessibility.compute.ts`
        : 'examples/deck/spatial-analysis-explorer/modes/accessibility-mode.ts',
      'utf8'
    );
    const buffers = {
      nodeCount: 2,
      cumulativeBuffer: Float32Array.of(12, 0),
      gravityBuffer: Float32Array.of(9, 0),
      twoStepBuffer: Float32Array.of(4, 0),
      scoresComplete: Uint32Array.of(0),
      snapOverflow: Uint32Array.of(0),
      cumulative: Float32Array.of(12, 0),
      gravity: Float32Array.of(9, 0),
      twoStep: Float32Array.of(4, 0)
    };
    const read = async (buffer: unknown, size: number) => {
      const view = buffer as ArrayBufferView;
      return new Uint8Array(view.buffer, view.byteOffset, size).slice();
    };
    const publisher = createApplicationScorePublisher(route, source);
    for (const complete of [0, 1, 0, 1]) {
      buffers.scoresComplete[0] = complete;
      const bytes = await readApplicationScorePacket(route, source, buffers, read);
      publisher.publish(route === 'job' ? bytes : new Uint8Array(bytes));
      expect(publisher.ready()).toBe(Boolean(complete));
      if (complete) {
        if (route === 'job')
          expect(publisher.values.get('top')).toBe('4000.0 maximum (fixed comparison classes)');
        else expect(publisher.values.get('range')).toBe('4.00 / 4.00');
      } else expect(publisher.values.get('reached')).toMatch(/withheld/);
    }
    // A source-list mutation must change the decoded magnitude, even though reached count agrees.
    const mutated =
      route === 'job'
        ? source.replace(
            '{buffer: twoStepBuffer, size: nodeCount * 4}',
            '{buffer: gravityBuffer, size: nodeCount * 4}'
          )
        : source.replace('sourceBuffer: getScoreBuffer(),', 'sourceBuffer: gravity,');
    expect(mutated).not.toBe(source);
    const wrongBytes = await readApplicationScorePacket(route, mutated, buffers, read);
    publisher.publish(route === 'job' ? wrongBytes : new Uint8Array(wrongBytes));
    expect(publisher.values.get(route === 'job' ? 'top' : 'range')).toBe(
      route === 'job' ? '9000.0 maximum (fixed comparison classes)' : '9.00 / 9.00'
    );
  }
});
