// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';
import {getGPUPermutationParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {
  GPUTerrainCurvature,
  type GPUTerrainCurvatureKind
} from '../../../analysis/src/gpu-terrain/terrain-curvature/gpu-terrain-curvature';
import {makeGPURasterBand} from '../../../analysis/src/raster-analysis/raster-grid/gpu-raster-grid';
import {createSingleCurvatureOutput} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/single-curvature-output';
const root = 'examples/deck/spatial-analysis-showcase/src/scenes/';
function snippet(path: string, start: string, end: string, marker = '') {
  const source = readFileSync(path, 'utf8');
  const offset = marker ? source.indexOf(marker) : 0;
  const first = source.indexOf(start, offset),
    last = source.indexOf(end, first);
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
function context() {
  const values = new Map<string, unknown>(),
    charts = new Map<string, unknown>();
  return {
    values,
    charts,
    ctx: {
      options: {},
      setReadout: (key: string, value: unknown) => values.set(key, value),
      setChart: (key: string, value: unknown) => charts.set(key, value),
      requestLayers() {},
      setStatus() {},
      setLegendExtent() {},
      setLegendData() {}
    }
  };
}
it('B8 queued hotspot fixture supplies a valid production permutation packer input', () => {
  const source = readFileSync(
    'modules/lenses/test/computation-lenses/showcase-weights-loss-readouts.spec.ts',
    'utf8'
  );
  const argument = source.match(/getGPUPermutationParameterValues\((\{[\s\S]*?\})\)/)![1];
  expect(() =>
    new Function('pack', `return pack(${argument});`)(getGPUPermutationParameterValues)
  ).not.toThrow();
});
it('B11 coloring does not publish suppression zeros as an exact coloring', () => {
  const test = context();
  const publish = evaluate(
    snippet(
      `${root}geometry/polygon-measures.compute.ts`,
      '  function publishColoring(',
      '  function updateSelectedReadout('
    ),
    {ctx: test.ctx},
    'publishColoring'
  );
  publish({
    coloringTable: {
      overflow: 1,
      colorCount: 0,
      conflictCount: 0,
      roundCount: 0,
      converged: 0,
      adjacencies: 2
    }
  });
  expect(test.values.get('coloring')).toMatch(/unavailable|incomplete/);
  expect(test.values.get('adjacencies')).toMatch(/at least 2/);
  expect(test.charts.get('coloringRounds')).toBeNull();
  publish({
    coloringTable: {
      overflow: 0,
      colorCount: 2,
      conflictCount: 0,
      roundCount: 3,
      converged: 1,
      adjacencies: 2
    }
  });
  expect(test.values.get('coloring')).toBe('2 colors, 0 conflicts, 3 rounds');
});
it('B9 observations do not classify an unseen match as outside on aggregate loss', () => {
  const test = context();
  const callback = evaluate(
    `const callback = ${snippet(`${root}joins/observations-by-tract.compute.ts`, '    bytes => {', '\n  );', "'observations-assign',")};`,
    {
      ctx: test.ctx,
      destroyed: false,
      pointCount: 3,
      featureCount: 2,
      GPU_SPATIAL_JOIN_NO_FEATURE: 0xffffffff,
      observationTract: [0, 1, 1],
      positions: new Float32Array(6),
      edgeIndex: {getDistance: () => 1},
      formatInteger: String,
      assignReader: {},
      assignmentComplete: false
    },
    'callback'
  );
  const words = Uint32Array.of(0, 0xffffffff, 1, 1, 1, 1, 3, 0);
  callback(words.buffer);
  expect(test.values.get('joined')).toMatch(/at least 2/);
  expect(test.values.get('outside')).toMatch(/unavailable/);
  words[5] = 0;
  callback(words.buffer);
  expect(test.values.get('outside')).toBe('1 (33.33%)');
});
it('B9 near-transit retained positives are lower bounds without a composition claim', () => {
  const test = context();
  const callback = evaluate(
    `const callback = ${snippet(`${root}joins/near-transit.compute.ts`, '      bytes => {', '\n    );', '`summary-${key}`')};`,
    {
      ctx: test.ctx,
      destroyed: false,
      variant: {key: 'key'},
      key: 'key',
      count: 3,
      formatInteger: String,
      source: {categoryNames: ['kind'], categories: [0, 0, 0]},
      selectionComplete: false
    },
    'callback'
  );
  callback(Uint32Array.of(1, 0, 1, 1, 0, 0).buffer);
  expect(test.values.get('selected')).toMatch(/at least 1/);
  expect(test.values.get('selected')).not.toMatch(/%/);
  expect(test.values.get('composition')).toMatch(/unavailable/);
  callback(Uint32Array.of(1, 0, 0, 1, 0, 0).buffer);
  expect(test.values.get('selected')).toBe('1 of 3 (33.3%)');
});
it('B10 ring join refuses exact population coverage and gap when the join lost candidates', () => {
  const test = context();
  const sizes = [4, 4, 4, 4, 4, 8, 12, 12];
  const callback = evaluate(
    `const callback = ${snippet(`${root}networks/isochrones.compute.ts`, '    bytes => {', '\n  );', "'isochrones-rings',")};`,
    {
      ctx: test.ctx,
      destroyed: false,
      RING_CAPACITY: 2,
      demandSizes: sizes,
      demand: {count: 3, population: [10, 10, 10]},
      totalPopulation: 30,
      GPU_SPATIAL_JOIN_NO_FEATURE: 0xffffffff,
      NO_BAND: 0xffffffff,
      formatInteger: String,
      cpuDemandBands: null,
      ringsComplete: false,
      sliceSections: (bytes: ArrayBuffer, sizes: number[]) => {
        let offset = 0;
        return sizes.map(size => {
          const result = bytes.slice(offset, offset + size);
          offset += size;
          return result;
        });
      }
    },
    'callback'
  );
  callback(Uint32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 0xffffffff, 0, 0, 0, 0).buffer);
  expect(test.values.get('ringJoin')).toMatch(/unavailable/);
  expect(test.values.get('ringJoinGap')).toMatch(/unavailable/);
  expect(test.charts.get('ringComparison')).toBeNull();
});
it('B10 nearest-facility refuses exact distance summaries and catchment rankings on loss', () => {
  const test = context();
  test.ctx.options = {radius: 300};
  const run = evaluate(
    snippet(
      `${root}joins/nearest-facility.compute.ts`,
      '  function handleSummary(',
      '  function writeMask('
    ),
    {
      ctx: test.ctx,
      placeCount: 3,
      HIDDEN_VALUE: -1,
      formatInteger: String,
      displayDistances: null,
      catchmentCounts: null,
      maximumCatchment: 1,
      nearestComplete: false
    },
    'handleSummary'
  );
  const bytes = new ArrayBuffer(28);
  new Float32Array(bytes).set([100, 3.4e38, 200]);
  new Uint32Array(bytes).set([1, 1, 1, 3], 3);
  run(
    {mode: 'nearest', set: {kind: 'points', count: 2, label: 'facilities', names: () => 'A'}},
    bytes
  );
  expect(test.values.get('median')).toMatch(/unavailable/);
  expect(test.values.get('catchment')).toMatch(/unavailable/);
  new Uint32Array(bytes)[5] = 0;
  run(
    {mode: 'nearest', set: {kind: 'points', count: 2, label: 'facilities', names: () => 'A'}},
    bytes
  );
  expect(test.values.get('median')).toMatch(/200 m median/);
});
it('NOTE satellite gaps and exact entry series are withheld on entry loss', () => {
  const test = context();
  test.charts.set('passChart', {series: [1]});
  test.charts.set('beltChart', {series: [1]});
  const block = snippet(
    `${root}earth/satellite-ground-tracks.compute.ts`,
    '    const band = bandIndex(bandCenter);',
    '    ctx.refreshTooltip();',
    '  function applyResult()'
  );
  evaluate(
    block,
    {
      ctx: test.ctx,
      areEntriesIncomplete: true,
      bandCenter: 0,
      bandIndex: () => 0,
      result: {enters: [0]},
      hours: 3,
      formatLatitude: String,
      BAND_MIN_LATITUDE: 0,
      BAND_HEIGHT: 5,
      active: {satelliteCount: 1},
      formatGap: String
    },
    'undefined'
  );
  expect(test.values.get('bandGap')).toMatch(/unavailable/);
  expect(test.charts.get('passChart')).toBeNull();
  expect(test.charts.get('beltChart')).toBeNull();
});
it('UNFINISHED zone dwell refuses durations when an omitted crossing can change a stay', () => {
  const test = context();
  test.ctx.options = {stopMinutes: 1};
  const primary = {
    flags: {candidateOverflow: 1},
    stays: {count: 1, zones: [0], seconds: [120], tracks: [0]}
  };
  const run = evaluate(
    snippet(
      `${root}movement/zone-dwell.compute.ts`,
      '  function updateStayReadouts()',
      '  function updateEventReadouts()'
    ),
    {
      ctx: test.ctx,
      getPrimary: () => primary,
      zoneCount: 1,
      groups: {groupOfZone: [0]},
      isShown: () => true,
      describeStay: () => '120 seconds',
      formatPercent: String,
      formatDwell: String
    },
    'updateStayReadouts'
  );
  run();
  expect(test.values.get('longestVisit')).toMatch(/unavailable/);
  expect(test.values.get('meanVisit')).toMatch(/unavailable/);
  primary.flags.candidateOverflow = 0;
  run();
  expect(test.values.get('longestVisit')).toBe('120 seconds');
});
it('UNFINISHED hurricane strongest landfall is unknown during candidate loss', () => {
  const test = context();
  // Execute the beginning of the actual publisher; no retained landfalls is not proof of none.
  const source = snippet(
    `${root}earth/hurricane-landfall.compute.ts`,
    '  function updateLandfallViews()',
    '    // Charts.'
  );
  const run = evaluate(
    source,
    {
      ctx: test.ctx,
      eventSnapshot: {candidateOverflow: 1},
      landfalls: [],
      zoneCount: 1,
      trackCount: 2,
      stateCounts: null,
      stateStrongest: null,
      stateStrongestStorm: null,
      stateDisplayBuffer: {write() {}},
      landfallMaximum: 1,
      eventCapacity: 2,
      dotPositions: {write() {}},
      dotCategories: {write() {}},
      dotCount: 0,
      formatCount: String
    },
    'updateLandfallViews'
  );
  run();
  expect(test.values.get('strongest')).toMatch(/unavailable/);
});
it('B6 topological selection labels zero retained matches as a lower bound', () => {
  const test = context();
  const run = evaluate(
    snippet(
      `${root}joins/topological-relations.compute.ts`,
      '  function refreshSelection()',
      '  /** Makes a built variant'
    ),
    {
      ctx: test.ctx,
      selected: 2,
      variant: {combo: 'tracts-self', hasMatrix: false},
      pairs: [],
      pairResultsIncomplete: true,
      getLeftName: () => 'left 2',
      getRightName: String
    },
    'refreshSelection'
  );
  run();
  expect(test.values.get('selection')).toMatch(/at least 0 matches.*incomplete/);
});
it('B9 rates decoder refuses to learn populations from a truncated assignment', () => {
  const test = context();
  const flags = {setValue: (value: unknown) => test.values.set('flags', value)};
  const source = snippet(
    'examples/deck/spatial-analysis-explorer/modes/rates-mode.ts',
    '      bytes => {',
    '        if (!eventsKnown)',
    "'rates',"
  );
  const run = evaluate(
    `let latest = null, eventsKnown = false, ratesComplete = false; const callback = ${source};`,
    {
      context: {updateLayers() {}},
      featureCount: 2,
      layout: {
        flags: 0,
        raw: 4,
        smoothed: 4,
        spatial: 4,
        standardized: 4,
        events: 4,
        populations: 4,
        pseudoP: 4,
        classes: 4
      },
      pooledReadout: flags,
      varianceReadout: flags,
      priorReadout: flags,
      smallStdReadout: flags,
      extremeReadout: flags,
      moranReadout: flags,
      classReadout: flags,
      selectedReadout: flags,
      flagsReadout: flags
    },
    '({run: callback, latest: () => latest, known: () => eventsKnown})'
  );
  const words = new Uint32Array(6);
  words[3] = 1;
  run.run(words.buffer);
  expect(run.latest()).toBeNull();
  expect(run.known()).toBe(false);
});
it('B10 explorer polygon join withholds the top-five ranking during aggregate loss', () => {
  const test = context();
  const source = snippet(
    'examples/deck/spatial-analysis-explorer/modes/polygon-join-mode.ts',
    '        const words = new Uint32Array(bytes.buffer',
    '        const maximum =',
    'const readSummary'
  );
  const callback = evaluate(
    `function callback(bytes) {${source}}`,
    {
      summaryWords: 4,
      featureCount: 2,
      zips: {featureNames: ['A', 'B'], featureIds: [0, 1]},
      formatCount: String,
      joinComplete: false,
      topReadout: {setValue: (v: unknown) => test.values.set('top', v)},
      overflowReadout: {setValue() {}},
      candidateReadout: {setValue() {}},
      context: {updateLayers() {}}
    },
    'callback'
  );
  callback(new Uint8Array(Uint32Array.of(1, 0, 1, 2).buffer));
  expect(test.values.get('top')).toMatch(/unavailable/);
});
it('B10 explorer nearest assignment labels the retained matched count as a lower bound', () => {
  const test = context();
  const source = snippet(
    'examples/deck/spatial-analysis-explorer/modes/nearest-join-mode.ts',
    '        const words = new Uint32Array(bytes.buffer',
    '\n      } catch',
    'const readSummary'
  );
  const callback = evaluate(
    `function callback(bytes) {${source}}`,
    {
      pointCount: 3,
      formatCount: String,
      matchedReadout: {setValue: (v: unknown) => test.values.set('matched', v)},
      overflowReadout: {setValue() {}},
      candidateReadout: {setValue() {}}
    },
    'callback'
  );
  callback(new Uint8Array(Uint32Array.of(1, 1, 2).buffer));
  expect(test.values.get('matched')).toMatch(/at least 1/);
});
it('UNFINISHED explorer zone events does not publish dwell or rankings after candidate loss', () => {
  const test = context();
  const handles = {setValue: (v: unknown) => test.values.set('visits', v)};
  const source = snippet(
    'examples/deck/spatial-analysis-explorer/modes/zone-events-mode.ts',
    '      bytes => {',
    '\n    );',
    'const summary = new SummaryReader('
  );
  const callback = evaluate(
    `let table; const callback = ${source};`,
    {
      destroyed: false,
      ZONE_COUNT: 2,
      visitsComplete: false,
      metricMaximum: {},
      writeChoropleth() {},
      writeSelection() {},
      eventReadout: {setValue() {}},
      visitReadout: handles,
      selectedReadout: {setValue() {}},
      context: {updateLayers() {}},
      formatCount: String,
      formatDuration: String,
      formatZoneEventsStatus: () => 'candidate loss',
      maxEventsPerTrack: 2
    },
    'callback'
  );
  const words = new Uint32Array(15);
  words[2] = 1;
  words[6] = 1;
  callback(words.buffer);
  expect(test.values.get('visits')).toMatch(/unavailable/);
  const rankSource = snippet(
    'examples/deck/spatial-analysis-explorer/modes/zone-events-mode.ts',
    '      bytes => {',
    '\n    );',
    "'zone-events-visits',"
  );
  const ranking = evaluate(
    `const callback = ${rankSource};`,
    {
      destroyed: false,
      tableCapacity: 1,
      TOP_DWELL_COUNT: 1,
      cellCount: 4,
      formatCount: String,
      formatDuration: String,
      visitTableReadout: {setValue() {}},
      topDwellReadouts: [{setValue: (v: unknown) => test.values.set('rank', v)}]
    },
    'callback'
  );
  const ranked = new Uint32Array(9);
  ranked[0] = 1;
  ranked[8] = 1;
  new Float32Array(ranked.buffer)[4] = 120;
  ranking(ranked.buffer);
  expect(test.values.get('rank')).toMatch(/unavailable/);
});

it('TRJ-105 zone event caption never promises exact statistics after candidate loss and recovers', () => {
  const test = context();
  const flags = {
    candidateOverflow: 0,
    trackOverflow: 0,
    events: 2,
    candidates: 2,
    eventOverflow: 0,
    listOverflow: 0
  };
  const run = evaluate(
    snippet(
      `${root}movement/zone-dwell.compute.ts`,
      '  function updateEventReadouts()',
      '  /** The note annotations'
    ),
    {
      ctx: test.ctx,
      inside: {flags},
      stopped: {flags: {}},
      eventTable: null,
      formatCount: String,
      EVENT_CANDIDATE_CAPACITY: 10
    },
    'updateEventReadouts'
  );
  for (const [candidate, track] of [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
  ]) {
    flags.candidateOverflow = candidate;
    flags.trackOverflow = track;
    run();
    const caption = String(test.values.get('overflow'));
    if (candidate) {
      expect(caption).toMatch(/candidate scratch.*statistics incomplete/);
      expect(caption).not.toMatch(/statistics stay exact/);
    } else if (track) {
      expect(caption).toMatch(/per-vessel.*statistics stay exact/);
    } else expect(caption).toBe('none');
  }
});

it('TRJ-107 harbor labels nonfinite RTS correction fallback independently of invalid state', () => {
  const test = context();
  // The reviewed nonfinite-correction fixture has det(prediction)=1 and finite states
  // [-3e38, 3e38]. Subtracting these overflows f32 after inversion; its row status is 10 (prediction-only 2 plus fallback 8).
  // A second fallback and one invalid row make swapped 4/8 masks distinguishable.
  const summary = {
    fullKalmanStatus: Uint32Array.of(1, 10, 8, 4),
    kalmanStatus: Uint32Array.of(0, 0, 0)
  };
  test.ctx.options = {covarianceModel: 'full'};
  const block = snippet(
    `${root}movement/harbor-track-quality.compute.ts`,
    '      const useFullCovariance =',
    '      const bins ='
  );
  evaluate(block, {ctx: test.ctx, summary, observationCount: 4}, 'undefined');
  expect(test.values.get('kalmanValidity')).toBe(
    '3 valid · 1 invalid states · 2 RTS fallback steps'
  );
  const reference = readFileSync(
    'examples/deck/spatial-analysis-showcase/scripts/check-harbor-quality-reference.mjs',
    'utf8'
  );
  expect(reference.includes('265 valid · 0 invalid states · 0 RTS fallback steps')).toBe(true);
});

it('B6 dense zone statistics use their own producer loss, independently of visit lists', () => {
  const source = readFileSync(`${root}movement/zone-dwell.compute.ts`, 'utf8');
  const start = source.indexOf('  function hasEventLoss(');
  const computeEnd = source.indexOf('    const frozen =', start);
  const rankStart = source.indexOf('    const rankValues =', computeEnd);
  const rankEnd = source.indexOf("    ctx.setReadout(\n      'zonesUsed'", rankStart);
  const flags: Record<string, number> = {};
  const stats = {counts: [2], sums: [120]};
  let ranked: unknown;
  const run = evaluate(
    `${source.slice(start, computeEnd)}${source.slice(rankStart, rankEnd)}}`,
    {
      destroyed: false,
      ctx: {options: {compareVariants: true}, setReadout() {}, setChart() {}},
      inside: {stats, flags},
      stopped: {stats, flags},
      groups: {},
      insideGroups: null,
      stoppedGroups: null,
      insideValues: null,
      stoppedValues: null,
      reduceToGroups: (value: unknown) => value,
      getGroupValuesFor: (value: unknown) => value,
      writeDisplay() {},
      getPrimary: () => ({flags}),
      updateRanking: (value: unknown) => {
        ranked = value;
      }
    },
    '({refresh, state: () => ({insideGroups, stoppedGroups})})'
  );
  for (const name of [
    'trackOverflow',
    'eventOverflow',
    'listOverflow',
    'candidateOverflow',
    'stopOverflow',
    'joinOverflow',
    'none'
  ]) {
    for (const key of Object.keys(flags)) flags[key] = 0;
    if (name !== 'none') flags[name] = 1;
    run.refresh();
    const state = run.state();
    expect(state.insideGroups, name).toBe(name === 'candidateOverflow' ? null : stats);
    expect(state.stoppedGroups, name).toBe(
      name === 'stopOverflow' || name === 'joinOverflow' ? null : stats
    );
    expect(ranked, name).toBe(name === 'candidateOverflow' ? null : stats);
  }
});

it('B7 event and stop readouts qualify retained crossings and suppress incomplete geographic breakdowns', () => {
  const test = context();
  const flags = {
    candidateOverflow: 1,
    trackOverflow: 1,
    events: 2,
    candidates: 2,
    eventOverflow: 0,
    listOverflow: 0
  };
  const stopped = {
    flags: {joinOverflow: 1, stopOverflow: 0},
    stays: {overflow: true, count: 2, zones: [0, 99]}
  };
  const events = evaluate(
    snippet(
      `${root}movement/zone-dwell.compute.ts`,
      '  function updateEventReadouts()',
      '  /** The note annotations'
    ),
    {
      ctx: test.ctx,
      inside: {flags},
      stopped,
      eventTable: {count: 2, zones: [0, 0]},
      groups: {groupOfZone: [0]},
      isShown: () => true,
      formatCount: String,
      EVENT_CANDIDATE_CAPACITY: 10
    },
    'updateEventReadouts'
  );
  const stops = evaluate(
    `${snippet(`${root}movement/zone-dwell.compute.ts`, '  function updateDefinitionReadouts()', '    const insideHours =')} }`,
    {
      ctx: test.ctx,
      stopped,
      zoneCount: 1,
      formatCount: String
    },
    'updateDefinitionReadouts'
  );
  events();
  stops();
  expect(test.values.get('events')).toMatch(/at least 2 retained crossings/);
  expect(test.values.get('stopCount')).toMatch(/unavailable.*incomplete/);
  flags.candidateOverflow = 0;
  flags.trackOverflow = 0;
  stopped.flags.joinOverflow = 0;
  stopped.stays.overflow = false;
  events();
  stops();
  expect(test.values.get('events')).toBe('2 crossings in the zones in view');
  expect(test.values.get('stopCount')).toBe('1 stops in the zones, 1 elsewhere');
});

it('B6 stopped visit lists and dense tooltips stay independent of event-list loss', () => {
  const test = context();
  test.ctx.options = {stopMinutes: 1};
  const primary = {
    variant: 'stopped',
    flags: {candidateOverflow: 1, trackOverflow: 1, stopOverflow: 0, joinOverflow: 0},
    stays: {count: 1, zones: [0], seconds: [120], tracks: [0], overflow: false}
  };
  const run = evaluate(
    snippet(
      `${root}movement/zone-dwell.compute.ts`,
      '  function updateStayReadouts()',
      '  function updateEventReadouts()'
    ),
    {
      ctx: test.ctx,
      getPrimary: () => primary,
      zoneCount: 1,
      groups: {groupOfZone: [0]},
      isShown: () => true,
      describeStay: () => '120 seconds',
      formatPercent: String,
      formatDwell: String
    },
    'updateStayReadouts'
  );
  run();
  expect(test.values.get('meanVisit')).toBe('120');
  primary.flags.stopOverflow = 1;
  run();
  expect(test.values.get('meanVisit')).toMatch(/unavailable/);
  primary.flags.stopOverflow = 0;
  const hasEventLoss = evaluate(
    snippet(
      `${root}movement/zone-dwell.compute.ts`,
      '  function hasEventLoss(',
      '  function refresh()'
    ),
    {},
    'hasEventLoss'
  );
  const handler = evaluate(
    `const handler = {${snippet(`${root}movement/zone-dwell.compute.ts`, '    getTooltip(event) {', '    onClick(event) {')}};`,
    {
      hasEventLoss,
      getPrimary: () => primary,
      getPrimaryValues: () => [120],
      vessels: {project: () => [0, 0]},
      findZone: () => 0,
      zones: {},
      groups: {groupOfZone: [0]},
      describeGroup: () => '120 dense seconds'
    },
    'handler'
  );
  expect(handler.getTooltip({coordinate: [0, 0]})).toBe('120 dense seconds');
});

it('W3-03 actual wildfire publisher suppresses all zonal readouts and restores each reduction', () => {
  const test = context();
  const incomplete = new Uint32Array(5);
  // Two cells per region: slopes 10/20 degrees, elevations 100/200 m.
  const table = {
    incomplete,
    slopeCells: [2, 2],
    slopeCount: [2, 2],
    slopeSum: [20, 40],
    slopeMax: [10, 20],
    southCount: [2, 2],
    southSum: [1, 0],
    steepCount: [2, 2],
    steepSum: [1, 0],
    elevationCount: [2, 2],
    elevationSum: [200, 400],
    rose: new Uint32Array(16).fill(1)
  };
  const run = evaluate(
    snippet(
      `${root}raster/wildfire-terrain.compute.ts`,
      '  const analyze = () => {',
      '  const stageReader ='
    ) +
      snippet(
        `${root}raster/wildfire-terrain.compute.ts`,
        '  const describeSelected =',
        '  const selectFire ='
      ) +
      snippet(
        `${root}raster/wildfire-terrain.compute.ts`,
        '  const stageReader =',
        '  const rasterReader ='
      ),
    {
      clamp: (value: number, minimum: number, maximum: number) =>
        Math.max(minimum, Math.min(maximum, value)),
      ctx: test.ctx,
      table,
      fireCount: 1,
      selected: 0,
      resources: {},
      zonalSources: [],
      destroyed: false,
      parseTable: () => table,
      SummaryReader: class {
        callback: (bytes: ArrayBuffer) => void;
        constructor(
          _resources: unknown,
          _name: string,
          _sources: unknown,
          callback: (bytes: ArrayBuffer) => void
        ) {
          this.callback = callback;
        }
      },
      describeFire: () => 'fixture fire',
      formatCount: String,
      visibleFires: [true],
      inDem: [1],
      MINIMUM_CELLS: 1,
      data: {names: ['fire']},
      OCTANTS: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
      median: (values: number[]) => values[0]
    },
    '() => stageReader.callback(new ArrayBuffer(0))'
  );
  test.ctx.options = {steepSlope: 15};
  const keys = [
    'scope',
    'comparable',
    'cellsInside',
    'cellsRing',
    'slopeMean',
    'steepShare',
    'southShare',
    'southness',
    'elevation',
    'steeperFires',
    'medianDifference'
  ];
  for (let producer = 0; producer < 5; producer++) {
    incomplete[producer] = 1;
    run();
    for (const key of [...keys, 'selected'])
      expect(test.values.get(key), key).toMatch(/unavailable/);
    for (const key of [
      'slopeChart',
      'steepChart',
      'differenceChart',
      'aspectChart',
      'aspectDifferenceChart'
    ])
      expect(test.charts.get(key), key).toBeNull();
    incomplete[producer] = 0;
    run();
    expect(test.values.get('slopeMean')).toBe('10.0 vs 20.0 deg (-10.0)');
    expect(test.values.get('elevation')).toBe('100 vs 200 m');
    expect(test.values.get('steepShare')).toBe('50.0% vs 0.0%');
    expect(test.values.get('southness')).toBe('0.50 vs 0.00');
    expect(test.values.get('selected')).toBe(
      'fixture fire\ncells inside 2, ring 2\nslope 10.0 vs 20.0 deg (max 10)\nelevation 100 vs 200 m'
    );
    expect(test.charts.get('aspectChart')).toMatchObject({
      series: [{y: new Array(9).fill(12.5)}, {y: new Array(9).fill(12.5)}]
    });
    expect(test.charts.get('aspectDifferenceChart')).toMatchObject({values: new Array(8).fill(0)});
    expect(test.charts.get('differenceChart')).toMatchObject({
      values: [1, ...new Array(15).fill(0)]
    });
    expect(test.charts.get('steepChart')).toMatchObject({values: [50, 0]});
    expect(test.charts.get('slopeChart')).toMatchObject({values: [10, 20]});
  }
});

it('W3-04 actual storm diagnostics distinguish static uncertainty from capacity loss and recover', () => {
  const test = context();
  const state = evaluate(
    `let staticOverflowed = false, exposureOverflowed = false, staticUncertainCount = 0, staticFlashOverflow = false; ${snippet(`${root}earth/storm-outage-exposure.compute.ts`, '  function updateOverflowReadout()', '  function updateValueRange()')}`,
    {ctx: test.ctx},
    '({run: updateOverflowReadout, set: (uncertain, capacity) => {staticUncertainCount = uncertain; staticOverflowed = capacity;}})'
  );
  state.set(1, false);
  state.run();
  expect(test.values.get('overflow')).toMatch(/1 uncertain.*not exact/);
  expect(test.values.get('overflow')).toMatch(/no capacity overflowed/);
  state.set(1, true);
  state.run();
  expect(test.values.get('overflow')).toMatch(/capacity.*overflowed.*1 uncertain/);
  state.set(0, false);
  state.run();
  expect(test.values.get('overflow')).toMatch(/no capacity.*no uncertain/);
  const source = readFileSync(`${root}earth/storm-outage-exposure.compute.ts`, 'utf8');
  expect(source).toContain('uncertainCount: importGraphBuffer(');
  expect(source).toContain('{buffer: zonalUncertainCount, size: 4}');
  expect(source).toContain('staticUncertainCount = flags[3]');
});

it('W3-03 every wildfire reduction guards the raster producer and publishes five loss words', () => {
  const source = readFileSync(`${root}raster/wildfire-terrain.compute.ts`, 'utf8');
  for (const [index, name] of ['slope', 'south', 'steep', 'elevation', 'rose'].entries()) {
    const block = source.slice(source.indexOf(`id: 'zonal-${name}'`)).split('      })')[0];
    expect(block).toContain(`status: status(${index})`);
    expect(block).toContain(name === 'rose' ? 'producerOverflow:' : '...common');
  }
  expect(source).toContain('{buffer: zonalIncomplete, size: 20}');
  expect(source).toContain('incomplete: take(Uint32Array, 5)');
});

it('W5-28 both quake readers correct reciprocal candidates against signed original-frame edges', () => {
  // Float32 reciprocal multiplication at the exact first 1312 m edge rounds down.
  expect(Math.fround(1312 * Math.fround(1 / 1312))).toBe(0.9999999403953552);
  const source = readFileSync(`${root}time/quake-plate-boundaries.compute.ts`, 'utf8');
  expect(source.match(/declarations: QUAKE_DISTANCE_CELL_WGSL/g)).toHaveLength(2);
  expect(source).toContain('body: QUAKE_EVENT_DISTANCE_WGSL');
  expect(source).toContain('body: QUAKE_LINKS_WGSL');
});

it('DEF-09 both curvature consumers build a selected nonempty map after narrowing the ring route', () => {
  for (const path of [
    'examples/deck/spatial-analysis-explorer/modes/geomorphometry-mode.ts',
    `${root}terrain/landforms-products.ts`
  ]) {
    const source = readFileSync(path, 'utf8');
    expect(source).toContain('createSingleCurvatureOutput(');
    expect(source).not.toMatch(
      /curvatures: \{\[(?:kind|curvatureChoice) as GPUTerrainCurvatureKind\]/
    );
    expect(source).toContain('ringCurvature:');
  }
});

it('DEF-09 selected curvature maps have exactly one requested leaf and genuine empty outputs still reject', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output = createTransientView(graph, 'curvature', 'float32', 9);
    const elevation = createTransientView(graph, 'elevation', 'float32', 9);
    const parameters = createTransientView(graph, 'settings', 'float32', 12);
    const grid: Parameters<typeof makeGPURasterBand>[1] = {
      width: 3,
      height: 3,
      affine: [1, 0, 0, 0, 1, 0],
      pixelInterpretation: 'area'
    };
    const common = {grid, elevation: makeGPURasterBand(elevation, grid), parameters};
    for (const kind of [
      'profile',
      'plan',
      'tangential',
      'mean',
      'gaussian',
      'minimal',
      'maximal',
      'unsphericity',
      'difference',
      'horizontal-excess',
      'vertical-excess',
      'accumulation',
      'ring',
      'rotor',
      'laplacian'
    ] as const) {
      const selected = createSingleCurvatureOutput(kind, output);
      expect(Object.keys(selected.curvatures!)).toEqual([kind]);
      expect(selected.curvatures![kind]).toBe(output);
      expect(() => new GPUTerrainCurvature({...common, output: selected})).not.toThrow();
    }
    expect(
      () => new GPUTerrainCurvature({...common, output: {ringCurvature: output}, ringRadii: [1, 2]})
    ).not.toThrow();
    // Deliberate JavaScript misuse: keep the constructor's empty-selection rejection.
    // @ts-expect-error empty output is not a permitted terminal selection
    expect(() => new GPUTerrainCurvature({...common, output: {}})).toThrow();
    expect(() =>
      createSingleCurvatureOutput('ring-multi-radius' as GPUTerrainCurvatureKind, output)
    ).toThrow(/kind must be one of/);
  } finally {
    device.destroy();
  }
});
