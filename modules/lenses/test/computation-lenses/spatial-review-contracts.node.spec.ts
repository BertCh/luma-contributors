// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it} from 'vitest';
import {
  getGPUNetworkSubgraphFilterParameterValues,
  computeNetworkLouvainOnCPU
} from '@luma-contributors/analysis/gpu-network';
import {getGPUStreamlinesParameterValues} from '@luma-contributors/analysis/raster-analysis';
import {fromRecordedLensFile} from '../../src/computation-lenses/index';
import {expectStraightSkeleton} from './straight-skeleton-expectations';
import {extractLensExample} from '../../src/computation-lenses/lens-example';

const source = (path: string) =>
  readFileSync(new URL(`../../../../${path}`, import.meta.url), 'utf8');
const scenes = 'examples/deck/spatial-analysis-showcase/src/scenes/';

it.each([
  'directed',
  'symmetric'
])('aviation %s airport tooltip qualifies provisional membership using its own solver (B6)', communityModel => {
  const compute = source(scenes + 'flows/aviation-network-lab.compute.ts');
  const method = compute
    .slice(compute.indexOf('    getTooltip(event) {'), compute.indexOf('    onThemeChange() {'))
    .trim()
    .replace(/,$/, '');
  const javascript = transpileModule(`const instance = {${method}};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const selected = Uint32Array.of(1, 0, 1, 1),
    other = Uint32Array.of(1, 1, 2, 4);
  const tooltip = new Function(
    'ctx',
    'lastAnalysis',
    `const nodeCount=1, region='west'; const regionData={lonLat:[0,0],iata:['A'],names:['Airport'],offsets:[0,1],airportFlights:[2]}; ${javascript}; return instance.getTooltip;`
  )(
    {
      options: {measure: 'communities', communityModel},
      getViewport: () => ({project: () => [0, 0]})
    },
    {
      communities: [7],
      symmetricCommunities: [9],
      communitySummary: communityModel === 'directed' ? selected : other,
      symmetricCommunitySummary: communityModel === 'symmetric' ? selected : other
    }
  );
  const score = () => tooltip({pixel: [0, 0]}).rows[2].value as string;
  expect(score()).toMatch(/provisional/i);
  expect(score()).toContain(`${communityModel} community ${communityModel === 'directed' ? 7 : 9}`);
  selected[1] = 1;
  expect(score()).not.toMatch(/provisional/i);
  selected[0] = 0;
  expect(score()).toBe('partition invalid');
});

it.each([
  'directed',
  'symmetric'
])('aviation %s quotient keeps iteration-limited labels provisional', communityModel => {
  const compute = source(scenes + 'flows/aviation-network-lab.compute.ts');
  const summaryBuffer =
    communityModel === 'directed' ? 'communitySummaryBuffer' : 'symmetricCommunitySummaryBuffer';
  expect(compute).toMatch(
    new RegExp(
      `labelsComplete: graph\\.createDataView\\(\\s*view\\(${summaryBuffer}, 'uint32', 4\\)\\.buffer,\\s*\\{\\s*format: 'uint32',\\s*length: 1,\\s*byteOffset: 4`
    )
  );
  const body = compute.slice(
    compute.indexOf('  function publishSummary()'),
    compute.indexOf('  function update()')
  );
  const javascript = transpileModule(body, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const readouts: Record<string, string> = {};
  const charts: Record<string, unknown> = {};
  // Hand-built valid partition, stopped after one sweep: validity=1, convergence=0.
  // Two opposite links of weight 2 merge into one community. Symmetric coarsening
  // counts the unordered pair once, carrying its doubled weight to preserve the total.
  const summary = Uint32Array.of(1, 0, 1, 1);
  const countedEdges = communityModel === 'directed' ? 2 : 1;
  const quotient = {
    summary: Uint32Array.of(2, 0, countedEdges, 0, countedEdges, 0, 1, 0),
    intraWeights: [4, 0],
    edgeWeights: [],
    overflow: 0
  };
  expect(quotient.summary[0]).toBe(2);
  const otherSummary = Uint32Array.of(1, 1, 2, 4);
  const otherEdges = communityModel === 'directed' ? 1 : 2;
  const otherQuotient = {
    summary: Uint32Array.of(2, 0, otherEdges, 0, 0, otherEdges, 2, otherEdges),
    intraWeights: [0, 0],
    edgeWeights: otherEdges === 1 ? [4] : [2, 2],
    overflow: 0
  };
  const publish = new Function(
    'ctx',
    'lastAnalysis',
    'regionData',
    'setReadout',
    'setChart',
    `
    const nodeCount = 2, requestLayers = () => {}, setAnnotations = () => {};
    ${javascript}; return publishSummary;
  `
  )(
    {options: {communityModel, measure: 'communities'}},
    {
      distance: [0, 0],
      hops: [0, 0],
      components: [0, 0],
      communities: communityModel === 'directed' ? [0, 0] : [0, 1],
      symmetricCommunities: communityModel === 'symmetric' ? [0, 0] : [0, 1],
      communitySummary: communityModel === 'directed' ? summary : otherSummary,
      symmetricCommunitySummary: communityModel === 'symmetric' ? summary : otherSummary,
      coarsening: {
        directed: communityModel === 'directed' ? quotient : otherQuotient,
        symmetric: communityModel === 'symmetric' ? quotient : otherQuotient
      },
      converged: false
    },
    {iata: ['A', 'B'], flights: [2, 2]},
    (key: string, value: string) => {
      readouts[key] = value;
    },
    (key: string, value: unknown) => {
      charts[key] = value;
    }
  );
  publish();
  expect(readouts['communityMembers']).toMatch(/provisional/i);
  expect(readouts['communityCoarsening']).toMatch(/provisional|iteration limit/i);
  expect(charts['communityCoarseningChart']).toBeNull();
  summary[1] = 1;
  publish();
  expect(readouts['communityCoarsening']).toContain('conserved');
  expect(readouts['communityCoarsening']).toContain(
    `1 groups · 0 inter-group pairs / ${countedEdges} within-group pairs`
  );
  expect(readouts['communityMembers']).toBe('0: A,B');
  expect(charts['communityCoarseningChart']).not.toBeNull();
});

it.each([
  true,
  false
])('the reciprocal two-node solver fixture merges in one limited sweep (directed=%s, S1)', directed => {
  // For gamma=1 the all-in-one partition has Q=0, improving on the singleton partition Q=-1/2.
  const result = computeNetworkLouvainOnCPU(
    {weights: Float32Array.of(0, 2, 2, 0), parameters: Float32Array.of(1, 0.0001)},
    {nodeCount: 2, directed, seed: 17, maximumLevels: 1, maximumIterations: 1}
  );
  expect(Array.from(result.communities)).toEqual([0, 0]);
  expect(Array.from(result.summary)).toEqual([1, 0, 1, 1]);
});

it('streamline placement spellings encode one finite twelve-word packet', () => {
  // Origins and steps are supplied directly; inverse seed steps are 1/2, stepLength is 1.
  const expected = [100, 220, 10, -10, 120, 220, 2, 2, 0.5, 0.5, 1, 0];
  expect(
    Array.from(
      getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [100, 220], cellSize: [10, -10]},
        gridPlacement: {origin: [120, 220], cellSize: [2, 2]},
        stepLength: 1
      })
    )
  ).toEqual(expected);
  expect(
    Array.from(
      getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [100, 220], cellSize: [10, -10]},
        gridPlacement: {origin: [120, 220], cellSize: [2, 2]},
        stepLength: 1
      })
    )
  ).toEqual(expected);
});

it('capture validates the serialized skeleton being written', () => {
  const capture = source('modules/lenses/test/computation-lenses/capture-records.spec.ts');
  expect(capture).toContain('fromRecordedLensFile(recorded)');
  expect(capture).not.toContain('scenario.load()');
  expect(capture.indexOf('fromRecordedLensFile(recorded)')).toBeLessThan(
    capture.indexOf('commands.writeFile')
  );
});

it('the provenance oracle rejects stale serialized edges even with green status', () => {
  const decoded = fromRecordedLensFile(
    JSON.parse(source('docs/lens-records/gpu-spatial-analysis.vector-straight-skeleton.json'))
  );
  const snapshots = decoded.record.captures?.snapshots ?? [];
  const values = (name: string) =>
    Array.from(snapshots.find(snapshot => snapshot.name === name)?.values ?? [], Number);
  expect(values('status')).toEqual([12, 12, 8, 0]);
  const sourceEdges = values('source-edges');
  // Deliberately corrupt arc 8, regardless of whether the archive is later recaptured.
  sourceEdges[16] = 6;
  sourceEdges[17] = 7;
  expect(() =>
    expectStraightSkeleton(name => (name === 'source-edges' ? sourceEdges : values(name)))
  ).toThrow();
});

it('prepared controls describe refresh requests without promising equal cost', () => {
  const scene = source(scenes + 'joins/observations-by-tract.scene.ts');
  expect(scene).not.toContain('forces a rebuild');
  expect(scene).not.toContain('costs what the unprepared one does');
  const compute = source(scenes + 'joins/observations-by-tract.compute.ts');
  expect(compute).toContain("ctx.setReadout('indexBuilds', 'not applicable (no prepared handle)')");
});

it('wind snippet defines field, viewport and camera-derived seed placements', () => {
  const snippet = source(scenes + 'raster/wind-flow.scene.ts').split('snippet: state =>')[1];
  expect(snippet).toContain('const fieldPlacement =');
  expect(snippet).toContain('const viewportPlacement =');
  expect(snippet).toContain('const cellY = cellX * Math.cos(camera.latitude * Math.PI / 180)');
  expect(snippet).toContain('cellSize: [cellX, cellY]');
});

it('feature legend does not certify clipped halo support from feature fit status', () => {
  const scene = source(scenes + 'terrain/lidar-ground.scene.ts');
  expect(scene).not.toContain('pale green points have complete support');
  expect(scene).not.toContain("label: 'Complete neighbourhood'");
  expect(scene).toContain('Halo completeness is not certified');
});

it('nearest control describes feature rows independently of legacy wire spelling', () => {
  const explorer = source('examples/deck/spatial-analysis-explorer/modes/nearest-join-mode.ts');
  expect(explorer).not.toContain('lowest-id cuts ties at k');
  expect(explorer).toContain("label: 'lowest feature row (cut at k)'");
});

it('facility tooltips remain available in weights and candidates modes', () => {
  const compute = source(scenes + 'joins/nearest-facility.compute.ts');
  const prefix = compute.split('getTooltip(event) {')[1].split('if (!event.coordinate)')[0];
  const isVisible = new Function('variant', 'nearestComplete', prefix + 'return true;');
  expect(isVisible({mode: 'weights'}, false)).toBe(true);
  expect(isVisible({mode: 'candidates'}, false)).toBe(true);
  expect(isVisible({mode: 'nearest'}, false)).toBe(null);
  expect(isVisible({mode: 'neighbors'}, false)).toBe(null);
  expect(isVisible({mode: 'nearest'}, true)).toBe(true);
});

it('the displayed travel time inequality includes zero and excludes exactly six', () => {
  const packet = getGPUNetworkSubgraphFilterParameterValues(
    {nodeColumnCount: 0, edgeColumnCount: 1, hasEdgeTimes: false},
    {edgeAttributeRanges: [[0, 6]]}
  );
  expect(Array.from(packet)).toEqual([0, 6, 1, 0]);
  const values = [-1, 0, 5.999999523162842, 6, 7];
  expect(values.map(value => packet[0] <= value && value < packet[1])).toEqual([
    false,
    true,
    true,
    false,
    false
  ]);
});

it('the canonical half-open network brush includes float32 below decimal 1.3 and excludes exact six', () => {
  const layout = {nodeColumnCount: 0, edgeColumnCount: 1, hasEdgeTimes: false};
  const packet = getGPUNetworkSubgraphFilterParameterValues(layout, {
    edgeAttributeRanges: [[0, 1.3]]
  });
  // Nearest float32 is 1.2999999523162842 < the decimal upper endpoint 1.3.
  const below = Math.fround(1.3);
  expect(Array.from(packet)).toEqual([0, 1.3000000715255737, 1, 0]);
  expect(below).toBe(1.2999999523162842);
  expect(packet[0] <= below && below < packet[1]).toBe(true);
  const six = getGPUNetworkSubgraphFilterParameterValues(layout, {edgeAttributeRanges: [[0, 6]]});
  expect(six[0] <= 6 && 6 < six[1]).toBe(false);
  const lens = source(
    'modules/lenses/src/computation-lenses/scenarios/gpu-network/network-subgraph-filter.lens.ts'
  );
  expect(lens).toContain('edgeAttributeRanges: [[0, MAXIMUM_SECONDS]]');
});

it('the visibility recipe displays the qualified public props type', () => {
  const recipe = source(
    'modules/lenses/src/computation-lenses/scenarios/gpu-network/visibility-graph-recipe.example.ts'
  );
  expect(recipe).toContain('type GPUNetworkVisibilityGraphCSRProps');
  expect(extractLensExample(recipe).code).toContain('props: GPUNetworkVisibilityGraphCSRProps');
});
