// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';
import {createPlaybackClock} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/playback';
import {createApplicationContributor} from './showcase-accessibility-fixture';
import {getGPUTimeWindowFilterParameterValues} from '@luma-contributors/analysis/dataframe-analysis';

const showcase = 'examples/deck/spatial-analysis-showcase/src/scenes/';
const explorer = 'examples/deck/spatial-analysis-explorer/modes/';

function extract(path: string, start: string, end: string, marker = '') {
  const source = readFileSync(path, 'utf8');
  const offset = marker ? source.indexOf(marker) : 0;
  expect(offset).toBeGreaterThanOrEqual(0);
  const first = source.indexOf(start, offset);
  const last = source.indexOf(end, first);
  expect(first).toBeGreaterThan(0);
  expect(last).toBeGreaterThan(first);
  return source.slice(first, last);
}

function evaluate(source: string, bindings: Record<string, unknown>, expression: string) {
  const body = transpileModule(source, {compilerOptions: {target: ScriptTarget.ES2022}}).outputText;
  return new Function(...Object.keys(bindings), `${body}\nreturn ${expression};`)(
    ...Object.values(bindings)
  );
}

const quantizeSatellitePlaybackTime = evaluate(
  extract(
    `${showcase}earth/satellite-playback-core.ts`,
    'export function quantizeSatellitePlaybackTime(',
    '/** Readback of the playhead graph:'
  ).replace('export ', ''),
  {},
  'quantizeSatellitePlaybackTime'
) as (seconds: number) => number;

it('PTS-130 showcase separates known reached, unreached and unknown demand and recovers', () => {
  const values = new Map<string, unknown>();
  const state = evaluate(
    `const callback = ${extract(`${showcase}regression/trade-areas.compute.ts`, '    get => {', '\n  );', "    'trade-areas',")};`,
    {
      ctx: {
        options: {demand: 'people', facility: 'clinic'},
        setReadout: (key: string, value: unknown) => values.set(key, value),
        setLegendExtent() {},
        requestLayers() {}
      },
      measures: {people: [2, 3, 5]},
      demandCount: 3,
      poverty: [0, 0, 0],
      latest: null,
      expectedRange: 0,
      ratioRange: 0,
      accessRange: {},
      getQuantile: () => 1,
      formatNumber: String,
      formatInteger: String,
      kindTotals: [1],
      FACILITY_KINDS: ['clinic'],
      selectedSupplyTotal: 12,
      tradeAreaColors: {write() {}},
      assignTradeColors() {},
      adjacency: []
    },
    '({callback, getLatest: () => latest})'
  );
  const callback = state.callback;
  const counts = Uint32Array.of(0, 1, 0xffffffff);
  const producerOverflow = Uint32Array.of(0);
  const get = (name: string) => ({
    f32: Float32Array.of(1, 1, 1),
    u32: name === 'reachable' ? counts : name === 'overflows' ? producerOverflow : Uint32Array.of(0)
  });
  callback(get);
  // Hand-worked partition: count 0 -> 2 unreached; count 1 -> 3 reached; sentinel -> 5 unknown.
  expect(values.get('unreached')).toMatch(/2.*unreached.*3.*reached.*5.*unknown/);
  expect(values.get('captured')).toMatch(/unavailable|n\/a/);
  expect(values.get('meanAccess')).toMatch(/unavailable|n\/a/);
  producerOverflow[0] = 1;
  callback(get);
  for (const key of ['captured', 'conservation', 'unreached'])
    expect(values.get(key)).toBe('n/a: incomplete weights');
  expect(state.getLatest().forwardWeightsIncomplete).toBe(true);
  producerOverflow[0] = 0;
  counts[2] = 1;
  callback(get);
  expect(values.get('unreached')).toMatch(/20%.*1 tracts/);
  expect(values.get('captured')).toMatch(/3 of 8/);
  expect(state.getLatest().forwardWeightsIncomplete).toBe(false);
});

it('PTS-130 explorer separates the sentinel before totals and recovers', () => {
  const values = new Map<string, string>();
  const readout = (name: string) => ({setValue: (value: string) => values.set(name, value)});
  const callback = evaluate(
    `const callback = ${extract(`${explorer}trade-areas-mode.ts`, '      bytes => {', '\n    );', "      'trade-areas',")};`,
    {
      demandCount: 3,
      facilityCount: 1,
      demandValues: [2, 3, 5],
      accessibilityRange: 0,
      expectedRange: 0,
      accessibilityRanges: new Map(),
      demandColor: '2sfca',
      getPositiveQuantile: () => 1,
      formatCount: String,
      formatDecimal: String,
      facilitySupply: 12,
      overflowReadout: readout('overflow'),
      unreachedReadout: readout('unreached'),
      conservationReadout: readout('conservation'),
      capturedReadout: readout('captured'),
      context: {updateLayers() {}}
    },
    'callback'
  );
  const bytes = new ArrayBuffer(56);
  new Float32Array(bytes).set([1, 1, 1, 1, 1, 1, 8], 4);
  const counts = new Uint32Array(bytes, 44, 3);
  counts.set([0, 1, 0xffffffff]);
  callback(bytes);
  expect(values.get('unreached')).toMatch(/2.*unreached.*3.*reached.*5.*unknown/);
  expect(values.get('captured')).toMatch(/unavailable|n\/a/);
  new Uint32Array(bytes)[0] = 1;
  callback(bytes);
  for (const key of ['captured', 'conservation', 'unreached'])
    expect(values.get(key)).toBe('n/a: incomplete weights');
  new Uint32Array(bytes)[0] = 0;
  counts[2] = 1;
  callback(bytes);
  expect(values.get('unreached')).toBe('1 of 3 cells');
  expect(values.get('captured')).toBe('8 of 8 (busiest 8)');
});

it('GEO-118 direct hull readouts distinguish group and vertex loss; recipe uses aggregate evidence', () => {
  for (const [group, vertex] of [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
  ]) {
    const values = new Map<string, string>();
    const source = extract(
      `${explorer}clusters-mode.ts`,
      '        const hullOverflow =',
      '        if (largestCluster >= 0)'
    );
    evaluate(
      source,
      {
        words: Uint32Array.of(0, 0, 0, 0, group || vertex, 0, 0, group, vertex),
        GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW: 1,
        GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW: 2,
        hullTotal: 1,
        hullLargest: 3,
        formatCount: String,
        hullReadout: {setValue: (value: string) => values.set('explorer', value)}
      },
      'undefined'
    );
    for (const isRecipe of [false, true]) {
      const caption = extract(
        `${showcase}points/nature-clusters.compute.ts`,
        '    const hullFlags =',
        '    const winner ='
      );
      evaluate(
        caption,
        {
          isRecipe,
          detail: {
            u32: (name: string) => [
              name === 'hullGroupOverflow'
                ? group
                : name === 'hullVertexOverflow'
                  ? vertex
                  : group || vertex
            ]
          },
          analysis: {hulls: [{}]},
          formatCount: String,
          GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW: 1,
          GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW: 2,
          ctx: {
            setReadout: (_key: string, value: string) =>
              values.set(isRecipe ? 'recipe' : 'direct', value)
          }
        },
        'undefined'
      );
    }
    for (const route of ['explorer', 'direct']) {
      const value = values.get(route)!;
      expect(value.includes('over 256')).toBe(Boolean(group));
      expect(value.includes('total capacity')).toBe(Boolean(vertex));
      expect(value.includes('OVERFLOW')).toBe(Boolean(group || vertex));
    }
    expect(values.get('recipe')!.includes('OVERFLOW')).toBe(Boolean(group || vertex));
    expect(values.get('recipe')).not.toMatch(/over 256|total capacity/);
  }
});

it('GEO-118 hull diagnostic bindings, copies and header offsets travel together', () => {
  const source = readFileSync(`${explorer}clusters-mode.ts`, 'utf8');
  expect(source).toContain('const SUMMARY_HEADER_WORDS = 9;');
  for (const name of ['hullGroupOverflow', 'hullVertexOverflow']) {
    expect(source).toContain(`copy(set.${name}, 4)`);
    const scene = readFileSync(`${showcase}points/nature-clusters.compute.ts`, 'utf8');
    expect(scene).toContain(`{name: '${name}', buffer: ${name}, size: 4}`);
  }
  expect(source).toMatch(/groupOverflow: importGraphBuffer/);
  expect(source).toMatch(/vertexOverflow: importGraphBuffer/);
});

it('DPC-108 consecutive explorer frames upload exact 1/1024-second bounds and display those bounds', () => {
  const uploads: Float32Array[] = [];
  const readouts = new Map<string, unknown>();
  const method = extract(
    `${explorer}time-mode.ts`,
    '      encode(commandEncoder, frame) {',
    '      getLayers()'
  );
  const instance = evaluate(
    `let currentTime = 0; const instance = {${method}};`,
    {
      playing: true,
      speed: 1,
      loopStart: () => -10,
      loopEnd: () => 10,
      trailSeconds: 0.25,
      getGPUTimeWindowFilterParameterValues,
      quantizeSatellitePlaybackTime,
      windowBuffer: {write: (value: Float32Array) => uploads.push(value.slice())},
      compiled: {encode() {}},
      SCRUBBER_INTERVAL_FRAMES: 1,
      READBACK_INTERVAL_FRAMES: 1,
      readbackPending: true,
      scrubber: {setValue() {}},
      formatClock: String,
      clockReadout: {setValue: (value: unknown) => readouts.set('clock', value)},
      windowReadout: {setValue: (value: unknown) => readouts.set('window', value)}
    },
    'instance'
  );
  // floor(t * 1024) / 1024, independently for t and t - 1/4. These binary fractions are exact.
  const expected = [
    [-0.150390625, 0.099609375],
    [0.0498046875, 0.2998046875],
    [0.75, 1]
  ];
  for (const [index, deltaSeconds] of [0.1, 0.2, 0.7].entries()) {
    instance.encode({}, {deltaSeconds, frameIndex: index});
    const lanes = uploads[index];
    expect(uploads).toHaveLength(index + 1);
    expect(Array.from(lanes).every(Number.isFinite)).toBe(true);
    expect(Array.from(lanes)).toEqual([
      expected[index][0],
      0,
      expected[index][1],
      0,
      0.25,
      0,
      0,
      0
    ]);
    expect([lanes[0] + lanes[1], lanes[2] + lanes[3]]).toEqual(expected[index]);
    expect(readouts.get('window')).toBe(`${expected[index][0]} – ${expected[index][1]}`);
    expect(readouts.get('clock')).toBe(String(expected[index][1]));
  }
});

it('DPC-108 satellite trail updates upload exact bounds without rejecting fractional animation times', () => {
  const uploads: Float32Array[] = [];
  const method = extract(
    `${showcase}earth/satellite-playback-core.ts`,
    '    encodeTrails(commandEncoder, end, lengthSeconds, fadeSeconds) {',
    '    setGroupFilter('
  );
  const core = evaluate(
    `const core = {${method}};`,
    {
      getGPUTimeWindowFilterParameterValues,
      quantizeSatellitePlaybackTime,
      windowParameters: {write: (value: Float32Array) => uploads.push(value.slice())},
      trailCompiled: {encode() {}}
    },
    'core'
  );
  const expected = [
    [-0.150390625, 0.099609375],
    [0.0498046875, 0.2998046875],
    [0.75, 1]
  ];
  for (const [index, end] of [0.1, 0.3, 1].entries()) {
    core.encodeTrails({}, end, 0.25, 0.25);
    const lanes = uploads[index];
    expect(uploads).toHaveLength(index + 1);
    expect(Array.from(lanes).every(Number.isFinite)).toBe(true);
    expect(Array.from(lanes)).toEqual([
      expected[index][0],
      0,
      expected[index][1],
      0,
      0.25,
      0,
      0,
      0
    ]);
    expect([lanes[0] + lanes[1], lanes[2] + lanes[3]]).toEqual(expected[index]);
  }
});

it('DPC-108 satellite displayed minutes agree with uploaded endpoints across clock boundaries', () => {
  const clock = evaluate(
    extract(
      `${showcase}earth/satellite-tracks.ts`,
      'export function formatSatelliteClock(',
      '/** Group palette'
    ).replaceAll('export ', ''),
    {},
    '({formatSatelliteClock, formatElapsed})'
  );
  const uploads: Float32Array[] = [];
  const method = extract(
    `${showcase}earth/satellite-playback-core.ts`,
    '    encodeTrails(commandEncoder, end, lengthSeconds, fadeSeconds) {',
    '    setGroupFilter('
  );
  const core = evaluate(
    `const core = {${method}};`,
    {
      getGPUTimeWindowFilterParameterValues,
      quantizeSatellitePlaybackTime,
      windowParameters: {write: (value: Float32Array) => uploads.push(value.slice())},
      trailCompiled: {encode() {}}
    },
    'core'
  );
  const tracks = {timeOriginUnixSeconds: 1700000000};
  for (const end of [9.999999, 10, 29.999999, 30, 69.999999, 70, 89.999999, 90]) {
    core.encodeTrails({}, end, 0.25, 0.25);
    const lanes = uploads[uploads.length - 1];
    const uploadedEnd = lanes[2] + lanes[3];
    expect(clock.formatSatelliteClock(tracks, uploadedEnd)).toBe(
      clock.formatSatelliteClock(tracks, end)
    );
    expect(clock.formatElapsed(uploadedEnd)).toBe(clock.formatElapsed(end));
  }
});

it('DPC-108 explorer trail-length control echoes the representable scrubber time', () => {
  const body = extract(
    `${explorer}time-mode.ts`,
    '      onChange: value => {',
    '\n    });',
    "label: 'Trail length (per-frame window)'"
  );
  const values: number[] = [];
  const run = evaluate(
    `let trailSeconds = 1; const control = {${body}};`,
    {
      currentTime: 0.1,
      scrubber: {setValue: (value: number) => values.push(value)}
    },
    'control.onChange'
  );
  run(0.25);
  expect(values).toEqual([0.099609375]);
});

it('GEO-118 both direct application constructors request each diagnostic output', () => {
  for (const path of [
    `${explorer}clusters-mode.ts`,
    `${showcase}points/nature-clusters.compute.ts`
  ]) {
    const instance = createApplicationContributor(
      readFileSync(path, 'utf8'),
      {
        GPUGroupConvexHull: class {
          constructor(readonly props: Record<string, unknown>) {}
        },
        importGraphBuffer: (_graph: unknown, _name: unknown, buffer: unknown) => buffer,
        shapes: {},
        data: {frame: {spatialContext: {}}},
        projection: {spatialContext: {}},
        options: {prefilterLevels: 0},
        shapePositions: {},
        shapeLabels: {},
        groupCount: 2,
        HULL_MAXIMUM_VERTICES: 3,
        HULL_CAPACITY: 8,
        hullIndices: 'indices',
        hullPositions: 'positions',
        hullOffsets: 'offsets',
        hullCounts: 'counts',
        hullOverflow: 'aggregate',
        hullGroupOverflow: 'group',
        hullVertexOverflow: 'vertex'
      },
      'GPUGroupConvexHull'
    );
    expect(instance.props['output'].status).toEqual({
      overflow: 'aggregate',
      groupOverflow: 'group',
      vertexOverflow: 'vertex'
    });
  }
});

it('B2 catchment count capture preserves raw words and declares their missing-value meaning', () => {
  const captures: Record<string, {view: unknown; meaning?: string}> = {};
  const reachableFacilities = {values: Uint32Array.of(2, 0xffffffff, 6)};
  const block = extract(
    'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/catchment-accessibility.lens.ts',
    "            captures['reachableFacilities'] =",
    "            captures['facilityOverflow'] ="
  );
  evaluate(block, {captures, result: {reachableFacilities}}, 'undefined');
  expect(captures['reachableFacilities'].view).toBe(reachableFacilities);
  expect(captures['reachableFacilities'].meaning).toMatch(
    /0xffffffff.*unknown.*not a numeric count/
  );
  expect(Array.from(reachableFacilities.values)).toEqual([2, 4294967295, 6]);
});

it('B3 Huff tooltip distinguishes forward loss from known empty catchments and recovers', () => {
  const latest = {
    access2: [NaN],
    access3: [NaN],
    reachable: [0xffffffff],
    tradeArea: [0xffffffff],
    probability: [NaN],
    forwardWeightsIncomplete: true
  };
  const method = extract(
    `${showcase}regression/trade-areas.compute.ts`,
    '    getTooltip(event) {',
    '    destroy()'
  );
  const handler = evaluate(
    `const handler = {${method}};`,
    {
      latest,
      locate: () => 0,
      geoids: ['one'],
      ctx: {options: {demand: 'people'}},
      measures: {people: [2]},
      poverty: [0],
      formatInteger: String,
      GPU_HUFF_NO_TRADE_AREA: 0xffffffff
    },
    'handler'
  );
  const event = {coordinate: [0, 0]};
  expect(handler.getTooltip(event)).toMatch(/Huff.*unavailable.*incomplete weights/);
  expect(handler.getTooltip(event)).not.toMatch(/No Huff trade area/);
  latest.forwardWeightsIncomplete = false;
  latest.reachable[0] = 0;
  expect(handler.getTooltip(event)).toMatch(/No Huff trade area/);
  latest.tradeArea[0] = 1;
  latest.probability[0] = 1;
  expect(handler.getTooltip(event)).toMatch(/Huff: site 1, probability 100%/);
});

it('B4 dependent recipe hull and winner readouts distinguish loss, degeneracy and recovery', () => {
  const values = new Map<string, unknown>();
  let hull: {areaSquareMeters: number} | undefined;
  const detailWords = {count: 0, overflow: 1};
  const detail = {
    u32: (name: string) =>
      name === 'hullCounts' ? [detailWords.count, 4] : [detailWords.overflow],
    f32: (name: string) =>
      name === 'areas' ? [0, 9] : name === 'perimeters' ? [0, 12] : [0, 0, 0, 0]
  };
  const set = {analysis: {ranked: [0, 1], sizes: [10, 5]}, config: {method: 'dbscan'}, detail};
  const bindings = {
    ctx: {options: {}, setReadout: (key: string, value: unknown) => values.set(key, value)},
    set,
    analysis: set.analysis,
    detail,
    isRecipe: true,
    formatCount: String,
    formatArea: String,
    formatDistance: String,
    getRankedHull: () => hull,
    formatRate: String,
    pendingFit: false
  };
  const publishRecipe = () =>
    evaluate(
      extract(
        `${showcase}points/nature-clusters.compute.ts`,
        '    const winner = analysis.ranked[0];',
        '    // Distribution of the whole selected pattern.'
      ),
      bindings,
      'undefined'
    );
  const publishWinner = evaluate(
    extract(
      `${showcase}points/nature-clusters.compute.ts`,
      '  function publishWinner(',
      '  /** Geographic bounds'
    ),
    bindings,
    'publishWinner'
  );
  publishRecipe();
  publishWinner(set);
  expect(values.get('recipeHulls')).toMatch(/largest unavailable.*retained.*9/);
  expect(values.get('hullArea')).toMatch(/unavailable/);
  expect(values.get('hullArea')).not.toMatch(/vertex cap/);
  detailWords.count = 2;
  detailWords.overflow = 0;
  publishRecipe();
  publishWinner(set);
  expect(values.get('hullArea')).toMatch(/degenerate.*2/);
  expect(values.get('recipeHulls')).toMatch(/degenerate/);
  detailWords.count = 4;
  hull = {areaSquareMeters: 9};
  publishWinner(set);
  expect(values.get('hullArea')).toBe('9');
});

it('S1 explorer hull diagnostics decode the actual copy ticket including the first group', async () => {
  const source = extract(
    `${explorer}clusters-mode.ts`,
    '    const readSummary = async (',
    '    /**\n     * Times DBSCAN'
  );
  const values = new Map<string, unknown>();
  const headerWords = 9;
  const ticket = {
    buffer: new Uint8Array(headerWords * 4 + 32),
    markEncoded() {},
    read() {
      return this.buffer;
    }
  };
  const set = {
    groupCount: 1,
    pointCount: 37,
    config: {method: 'dbscan'},
    destroyed: false,
    summaryByteLength: ticket.buffer.length,
    readbackRing: {tryAcquire: () => ticket},
    clusterCount: Uint32Array.of(1),
    clusterStoredCount: Uint32Array.of(99),
    drawCommands: {
      buffer: Uint32Array.of(6, 1, 0, 0),
      getInstanceCountByteOffset(index: number) {
        expect(index).toBe(0);
        return Uint32Array.BYTES_PER_ELEMENT;
      }
    },
    clusterOverflow: Uint32Array.of(0),
    geometryOverflow: Uint32Array.of(0),
    hullOverflow: Uint32Array.of(1),
    hullGroupOverflow: Uint32Array.of(0),
    hullVertexOverflow: Uint32Array.of(1),
    iterationCount: null,
    converged: null,
    sizes: Uint32Array.of(37),
    geometryCounts: Uint32Array.of(37),
    hullCounts: Uint32Array.of(2),
    meanCenters: new Float32Array(2),
    weightedCenters: new Float32Array(2),
    medoids: Uint32Array.of(0xffffffff)
  };
  const bindings: Record<string, unknown> = {
    destroyed: false,
    readbackPending: false,
    epsilon: 1,
    minimumSamples: 2,
    scheduleReadback: (read: () => unknown) => read(),
    CLUSTER_CAPACITY: 2048,
    SUMMARY_HEADER_WORDS: headerWords,
    KMEANS_ITERATIONS: 48,
    MEDOID_MAXIMUM_GROUP_SIZE: 1024,
    formatCount: String
  };
  for (const name of [
    'clusters',
    'clustered',
    'largest',
    'mean',
    'parameters',
    'geometry',
    'convergence',
    'hull',
    'center',
    'medoid'
  ]) {
    bindings[`${name}Readout`] = {setValue: (value: unknown) => values.set(name, value)};
  }
  const run = evaluate(source, bindings, 'readSummary');
  const encoder = {
    copyBufferToBuffer({
      sourceBuffer,
      sourceOffset,
      destinationBuffer,
      destinationOffset,
      size
    }: {
      sourceBuffer: ArrayBufferView;
      sourceOffset: number;
      destinationBuffer: Uint8Array;
      destinationOffset: number;
      size: number;
    }) {
      destinationBuffer.set(
        new Uint8Array(sourceBuffer.buffer, sourceBuffer.byteOffset + sourceOffset, size),
        destinationOffset
      );
    }
  };
  for (const [group, vertex] of [
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
  ]) {
    set.hullGroupOverflow[0] = group;
    set.hullVertexOverflow[0] = vertex;
    set.hullOverflow[0] = group || vertex;
    await run(set, encoder);
    // Instance-count word (1) differs from vertex count (6) and the standalone count (99).
    expect(new Uint32Array(ticket.buffer.buffer)[1]).toBe(1);
    expect(String(values.get('hull')).includes('over 256')).toBe(Boolean(group));
    expect(String(values.get('hull')).includes('total capacity')).toBe(Boolean(vertex));
    expect(values.get('largest')).toBe('37');
    expect(values.get('geometry')).toBe('equal for every cluster');
    expect(new Uint32Array(ticket.buffer.buffer)[headerWords]).toBe(37);
  }
});

it('B5 accumulated satellite frames use the same representable endpoint for upload and display', () => {
  const clock = evaluate(
    extract(
      `${showcase}earth/satellite-tracks.ts`,
      'export function formatSatelliteClock(',
      '/** Group palette'
    ).replaceAll('export ', ''),
    {},
    '({formatSatelliteClock, formatElapsed})'
  );
  let playhead = 0;
  for (let frame = 0; frame < 360; frame++) playhead += (1 / 90) * 30;
  // Shipped CelesTrak epoch; the expected grid tick is floor(120*1024 - epsilon)/1024.
  const tracks = {timeOriginUnixSeconds: 1791331200};
  const displayedPlayhead = quantizeSatellitePlaybackTime(playhead);
  const uploads: Float32Array[] = [];
  const method = extract(
    `${showcase}earth/satellite-playback-core.ts`,
    '    encodeTrails(commandEncoder, end, lengthSeconds, fadeSeconds) {',
    '    setGroupFilter('
  );
  const core = evaluate(
    `const core = {${method}};`,
    {
      getGPUTimeWindowFilterParameterValues,
      quantizeSatellitePlaybackTime,
      windowParameters: {write: (value: Float32Array) => uploads.push(value.slice())},
      trailCompiled: {encode() {}}
    },
    'core'
  );
  core.encodeTrails({}, displayedPlayhead, 60, 60);
  const uploaded = uploads[0][2] + uploads[0][3];
  expect(uploaded).toBe(119.9990234375);
  expect(clock.formatSatelliteClock(tracks, displayedPlayhead)).toBe(
    clock.formatSatelliteClock(tracks, uploaded)
  );
});

it('S3 S2 guide names the measured equal-area denominator and excludes the special rows', () => {
  // Measurement fixture: 10,240 total rows minus 64 special cases = 10,176 equal-area rows.
  const guide = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/docs/guides/cells.md',
    'utf8'
  );
  expect(guide).toMatch(/10,176-point equal-area sample \(excluding 64 special-case rows\)/);
  expect(guide).toMatch(/6\.6% mismatches at level 20/);
});

it.each([
  'satellite-playback',
  'satellite-constellations'
])('DPC-108 %s actual frame callback formats the uploaded endpoint', scene => {
  const clockContext = {
    options: {time: 0, play: true, speed: 30, loop: false},
    setOptions(updates: Partial<{time: number; play: boolean; speed: number; loop: boolean}>) {
      Object.assign(this.options, updates);
    }
  };
  const clock = createPlaybackClock(
    clockContext,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, 3600], rate: 1, step: 1}
  );
  const formatters = evaluate(
    extract(
      `${showcase}earth/satellite-tracks.ts`,
      'export function formatSatelliteClock(',
      '/** Group palette'
    ).replaceAll('export ', ''),
    {},
    '({formatSatelliteClock, formatElapsed})'
  );
  let uploaded = 0;
  let trailEnd = 0;
  const readouts = new Map<string, unknown>();
  const method = extract(
    `${showcase}earth/${scene}.compute.ts`,
    '    encode(commandEncoder, frame) {',
    '    getLayers()'
  );
  const instance = evaluate(
    `const instance = {${method}};`,
    {
      ctx: {
        options: {showTrails: true, trailMinutes: 1, tailFade: 1},
        setReadout: (key: string, value: unknown) => readouts.set(key, value)
      },
      clock,
      quantizeSatellitePlaybackTime,
      tracks: {timeOriginUnixSeconds: 1791331200},
      ...formatters,
      active: {dirty: false, reader: {flush() {}}},
      core: {
        encodePlayhead: (_encoder: unknown, time: number) => (uploaded = time),
        encodeTrails: (_encoder: unknown, end: number) => (trailEnd = end),
        reader: {markStale() {}, flush() {}}
      }
    },
    'instance'
  );
  for (let frameIndex = 0; frameIndex < 360; frameIndex++)
    instance.encode({}, {frameIndex, timeSeconds: frameIndex / 90, deltaSeconds: 1 / 90});
  // floor((360 * 30/90 accumulated in binary64) * 1024)/1024.
  expect(uploaded).toBe(119.9990234375);
  expect(trailEnd).toBe(uploaded);
  expect(readouts.get('clock')).toMatch(/^00:01/);
  expect(clock.time).toBeGreaterThan(uploaded);
});
