// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';

it('explorer neighborhood range withholds search loss with clear median overflow and recovers', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/spatial-weights-mode.ts',
    'utf8'
  );
  const start = source.indexOf('        const nextResultLinks =');
  const end = source.indexOf('\n      } catch', start);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const readouts: string[] = [];
  const reader = new Function(
    'publish',
    `
    const floats = [2, 7], words = new Uint32Array(5), SUMMARY_EXTENT = 0, SUMMARY_TOOLKIT_FLAGS = 0;
    const resultSlots = 2, resultCapacity = 4, maximumCardinality = 1;
    const neighborhoodReadout = {setValue: publish}, formatNumber = String;
    const context = {updateLayers() {}};
    let resultLinkCount = 0, cardinalityMaximum = 0, neighborhoodRange = [0, 1];
    return (searchOverflow, medianCapacityLost = false, neighborhoodStatistic = 'mean') => {words[4] = Number(medianCapacityLost); ${body} return [...neighborhoodRange];};
  `
  )((value: string) => readouts.push(value));
  // Intermediate search loss alone makes the range unavailable; final median output has room.
  expect(reader(true)).toEqual([0, 1]);
  expect(readouts.at(-1)).toMatch(/unavailable.*incomplete/i);
  expect(reader(false, true)).toEqual([2, 7]);
  expect(readouts.at(-1)).toBe('2 to 7');
  expect(reader(false, true, 'median')).toEqual([2, 7]);
  expect(readouts.at(-1)).toMatch(/unavailable.*median overflow/i);
  expect(reader(false)).toEqual([2, 7]);
  expect(readouts.at(-1)).toBe('2 to 7');
});

it('polygon hot-spots production readout withholds candidate-only loss and recovers', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/hot-spots.compute.ts',
    'utf8'
  );
  const marker = source.indexOf('getReaderSources(buffers, rows, core.overflow');
  expect(marker).toBeGreaterThan(0);
  const start = source.indexOf('      snapshot => {', marker);
  const end = source.indexOf('\n    );', start);
  const callback = transpileModule(`const callback = ${source.slice(start, end)};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, unknown>();
  const reader = new Function(
    'ctx',
    `
    const created = {snapshot: null}; const world = created; const switching = false;
    const graphSwap = {pending: false}; let published = 0;
    const publishPending = message => ctx.setReadout('analysisStatus', message);
    const updateReadouts = () => {published++; ctx.setReadout('analysisStatus', 'Ready');};
    ${callback}
    return {callback, state() {return {snapshot: created.snapshot, published};}};
  `
  )({
    setReadout: (key: string, value: unknown) => readouts.set(key, value),
    setAnnotations() {},
    setLegendData() {},
    requestLayers() {}
  });
  // Candidate-only loss: final CSR capacity has room, but the raw contiguity pairs lost entries.
  reader.callback({weightsOverflow: 0, weightsCandidateOverflow: 1});
  expect(reader.state().snapshot).toBeNull();
  expect(reader.state().published).toBe(0);
  expect(readouts.get('analysisStatus')).toMatch(/incomplete/i);
  reader.callback({weightsOverflow: 0, weightsCandidateOverflow: 0});
  expect(reader.state().snapshot).not.toBeNull();
  expect(reader.state().published).toBe(1);
});

it('polygon hot-spots production byte decoder reads the appended candidate-loss word', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/hot-spots.compute.ts',
    'utf8'
  );
  const start = source.indexOf('  const parseSnapshot =');
  const end = source.indexOf('  const getReaderSources =', start);
  expect(start).toBeGreaterThan(0);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const parse = new Function(`const GI_BIN_COUNT = 7; ${body} return parseSnapshot;`)();
  // Seven bins, four statistics, permutation flag, two counts, extent, final loss;
  // then five per-row arrays and the polygon candidate-loss word.
  const words = new Uint32Array(17 + 5 * 4 + 1);
  words[words.length - 1] = 1;
  expect(parse(words.buffer, 4, false).weightsOverflow).toBe(0);
  expect(parse(words.buffer, 4, false).weightsCandidateOverflow).toBe(1);
  words[words.length - 1] = 0;
  expect(parse(words.buffer, 4, false).weightsCandidateOverflow).toBe(0);
});

it('showcase weights production captions withhold candidate-only loss and recover', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/spatial-weights.compute.ts',
    'utf8'
  );
  const start = source.indexOf('  function updateReadouts()');
  const end = source.indexOf('  /** The standing sample line', start);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, unknown>();
  const reader = new Function(
    'ctx',
    `
    const kit = {producerCandidateOverflow: true, producerOverflow: false, slots: 2, isolates: 0,
      asymmetricSlots: 0, s0: 2, s1: 4, s2: 8, lagMinimum: 1, lagMaximum: 2,
      unionOverflow: false, unionSlots: 2, focus: {weights: {degree: 1, weightSum: 1}}};
    const world = {geography: {count: 2, medianSpacing: 1, analysisFrame: {name: 'local', maxScaleError: 0}},
      kitSummary: kit, snapshot: null, cornerLinks: null, core: {getBandMeters: () => 1}};
    let counts = {}; const publishLegendData = () => {};
    const formatCount = String, formatDistance = String;
    const getFocusName = () => 'selected', getConfig = () => ({}), getVariableInfo = () => ({digits: 1});
    ${body} return {updateReadouts, recover() {kit.producerCandidateOverflow = false;}};
  `
  )({
    options: {matrix: 'weights'},
    setReadout: (key: string, value: unknown) => readouts.set(key, value),
    setChart() {},
    setLegendData() {},
    requestLayers() {},
    setCost() {}
  });
  reader.updateReadouts();
  expect(readouts.get('islands')).toMatch(/unavailable/);
  expect(readouts.get('meanNeighbors')).toMatch(/unavailable/);
  reader.recover();
  reader.updateReadouts();
  expect(readouts.get('links')).toBe('2');
  expect(readouts.get('capacity')).toBe('ok');
});

it('explorer weights production captions withhold aggregate candidate loss and recover', async () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/spatial-weights-mode.ts',
    'utf8'
  );
  const start = source.indexOf('    const readKitSummary =');
  const end = source.indexOf('    const setFocus =', start);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const values = new Map<string, unknown>();
  const handles = [
    'sourceRowsReadout',
    'sourceDegreeReadout',
    'sourceIslandsReadout',
    'sourceMomentsReadout',
    'sourceAsymmetryReadout',
    'sourceUnionReadout',
    'sourceLagReadout',
    'sourceFocusReadout'
  ];
  let loss = true;
  const getSummary = async () => ({
    producerOverflow: loss,
    slots: 2,
    isolates: 0,
    minimumCardinality: 1,
    maximumCardinality: 1,
    s0: 2,
    s1: 4,
    s2: 8,
    unionSlots: 2,
    transposeAsymmetricSlots: 0,
    unionOverflow: false,
    lagMinimum: 1,
    lagMaximum: 2,
    focus: {weights: {degree: 1, weightSum: 1}}
  });
  const reader = new Function(
    'getSummary',
    'handles',
    `
    const {${handles.join(', ')}} = handles;
    const active = {kit: {readSummary: getSummary}, linkCapacity: 4, rowCount: 2, label: 'block', focusRow: 0};
    const getActiveSource = () => active, getActiveMatrix = () => 'weights';
    const destroyed = false, source = 'blocks', capacity = 4, cellCount = 2, UNION_SLOTS_PER_ROW = 2;
    let kitSummary, kitReadbackPending, kitNeedsReadback, transposeLinkCount, unionLinkCount;
    const formatCount = String, formatNumber = String, context = {updateLayers() {}};
    ${body} return readKitSummary;
  `
  )(
    getSummary,
    Object.fromEntries(
      handles.map(name => [name, {setValue: (value: unknown) => values.set(name, value)}])
    )
  );
  await reader({});
  expect(values.get('sourceIslandsReadout')).toMatch(/unavailable/);
  expect(values.get('sourceAsymmetryReadout')).toMatch(/unavailable/);
  loss = false;
  await reader({});
  expect(values.get('sourceIslandsReadout')).toBe('0');
});

it('queen-versus-rook production reference decoder withholds comparisons on either candidate-loss word', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/spatial-weights.compute.ts',
    'utf8'
  );
  const marker = source.indexOf("'weights-reference',");
  const start = source.indexOf('      bytes => {', marker);
  const end = source.indexOf('\n    );', start);
  const body = transpileModule(`const callback = ${source.slice(start, end)};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const reader = new Function(`
    const target = {reference: {data: null}, cornerRows: [], cornerLinks: null};
    const world = target, rows = 2, queen = {slots: 2}, rook = {slots: 2}; let publications = 0;
    const ctx = {setReadout() {}, requestLayers() {}};
    function updateCorners() {publications++;} function publishAnnotations() {}
    ${body} return {callback, state() {return {data: target.reference.data, publications};}};
  `)();
  // Two CSR offset arrays (3 words each), two neighbor arrays (2 words each), four loss words.
  for (const index of [11, 13]) {
    const words = new Uint32Array(14);
    words[index] = 1;
    reader.callback(words.buffer);
    expect(reader.state().data).toBeNull();
    expect(reader.state().publications).toBe(0);
  }
  reader.callback(new Uint32Array(14).buffer);
  expect(reader.state().data).not.toBeNull();
  expect(reader.state().publications).toBe(1);
});
