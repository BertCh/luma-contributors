// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test, vi} from 'vitest';
import type {GPUComputationNodeMetrics} from '@luma-contributors/analysis/computation-record';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import * as lensRegistry from '../../../src/computation-lenses/registry-implementation';
import {getRasterViewPack} from '../../../src/gpu-raster/computation-pack/pack';
import {
  ViewRegistry,
  createViewContext
} from '../../../src/computation-views/registry/view-registry';
import {buildLensTwin} from '../../../src/computation-views/twin/static-twin';
import {
  RESULT_VIEW,
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {getRasterLensGeometry} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-harness';
import {loadDixieDem} from '../../../src/computation-lenses/scenarios/gpu-raster/raster-lens-data';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';
import {buildDatumReceiptModel} from '../../../src/computation-views/views/datum-receipt';
import {TABLE_VIEW} from '../../../src/computation-views/views/table-view';
import {resolveValueScale} from '../../../src/computation-views/views/value-strip';
import {buildCompareViewModel} from '../../../src/computation-views/views/compare-view';
import {
  buildTimingViewModel,
  renderTimingView
} from '../../../src/computation-views/views/timing-view';
import {
  REALIZED_VIEW,
  buildRealizedViewModel
} from '../../../src/computation-views/views/realized-view';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';
import {makeContext, makeCapture} from '../fixtures/v4-record';
import {makeContext as makeLoopContext} from '../fixtures/v5-record';

test('B1 standalone real FillHoles archive resolves the unique resource writer', () => {
  const archive = JSON.parse(
    readFileSync(
      new URL(
        '../../../../../docs/lens-records/gpu-raster.raster-fill-holes.json',
        import.meta.url
      ),
      'utf8'
    )
  );
  const scenarioId = 'gpu-raster.raster-fill-holes';
  const previousScenario = lensRegistry.getLensScenario(scenarioId);
  // Exercise the shared-module state of the full suite without importing or leaking registrations.
  const dispose = previousScenario
    ? undefined
    : lensRegistry.registerLensScenario({
        id: scenarioId,
        entry: 'gpu-raster',
        element: 'GPURasterFillHoles',
        title: 'test fixture',
        audience: 'analyst',
        rungs: ['result'],
        defaultRung: 'result',
        load: async () => ({run: async () => fromRecordedLensFile(archive).record})
      });
  expect(lensRegistry.getLensScenario(scenarioId)).toBeDefined();
  const scenarioLookup = vi.spyOn(lensRegistry, 'getLensScenario').mockReturnValue(undefined);
  try {
    expect(lensRegistry.getLensScenario('gpu-raster.raster-fill-holes')).toBeUndefined();
    scenarioLookup.mockClear();
    const registry = new ViewRegistry();
    let outputPalette: string | undefined;
    registry.register({
      ...RESULT_VIEW,
      build: context => {
        const model = buildResultViewModel(context);
        outputPalette = model.output.palette;
        return model;
      }
    });
    registry.registerPack(getRasterViewPack());
    const twin = buildLensTwin(
      archive,
      {scenario: 'gpu-raster.raster-fill-holes', rung: 'result'},
      {registry}
    );
    expect(outputPalette).toBe('nom');
    expect(twin.figure.html).toContain('cv-nom-');
    const record = fromRecordedLensFile(archive).record;
    expect(
      buildResultViewModel(createViewContext(record, {packs: registry.getPacksFor(record)})).output
        .palette
    ).toBe('nom');
    expect(scenarioLookup).toHaveBeenCalledWith(scenarioId);
  } finally {
    scenarioLookup.mockRestore();
    dispose?.();
  }
  expect(lensRegistry.getLensScenario(scenarioId)).toBe(previousScenario);
});

test('B3 writer accepts the production loader CRS and canonicalizes PROJJSON identity', async () => {
  const data = await loadDixieDem();
  expect(data.crs).toBe('EPSG:32610');
  expect(getRasterLensGeometry(data)).toMatchObject({crs: 'EPSG:32610'});
  const explicit = getRasterLensGeometry(data, {
    kind: 'raster',
    shape: [1, 1],
    cellSize: 5,
    origin: [100, 200],
    coordinateModel: 'projected',
    unit: 'm'
  });
  expect(explicit).toMatchObject({
    shape: [1, 1],
    cellSize: 5,
    origin: [100, 200],
    crs: 'EPSG:32610'
  });
  expect(explicit).not.toHaveProperty('affine');
  const first = getRasterLensGeometry({
    ...data,
    crs: undefined,
    grid: {
      ...data.grid,
      coordinateReferenceSystem: {
        projectionJson: {type: 'ProjectedCRS', datum: {name: 'fixture', code: 1}}
      }
    }
  });
  const second = getRasterLensGeometry({
    ...data,
    crs: undefined,
    grid: {
      ...data.grid,
      coordinateReferenceSystem: {
        projectionJson: {datum: {code: 1, name: 'fixture'}, type: 'ProjectedCRS'}
      }
    }
  });
  expect(first).toEqual(second);
});

test('B4 rectangular raster first cell and ground-space companion point coincide', () => {
  const record = makeY2Record([
    makeY2Capture('ground', [12, 29], {
      role: 'context',
      components: 2,
      geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
    }),
    makeY2Capture('raster', [1, 2, 3, 4, 5, 6], {
      role: 'output',
      geometry: {
        kind: 'raster',
        shape: [3, 2],
        affine: [4, 0, 10, 0, -6, 32],
        pixelInterpretation: 'area',
        coordinateModel: 'projected',
        unit: 'm'
      }
    })
  ]);
  const context = createViewContext(record);
  const model = buildResultViewModel(context);
  // Inverse affine: (12-10)/4 = .5; (29-32)/-6 = .5. Pixel display y = 2-.5.
  expect(model.panels.find(panel => panel.capture.name === 'ground')!.points).toEqual([[0.5, 1.5]]);
  const tree = renderResultView(model, context);
  const image = findViewNodes(tree, node => node.tag === 'image')[0];
  const point = findViewNodes(tree, node => node.tag === 'circle')[0];
  expect(Number(point.attributes['cx'])).toBeCloseTo(
    Number(image.attributes['x']) + Number(image.attributes['width']) / 6,
    1
  );
  expect(Number(point.attributes['cy'])).toBeCloseTo(
    Number(image.attributes['y']) + Number(image.attributes['height']) / 4,
    1
  );
  expect(model.projection.statement).toContain('pixels');
});

function makeProducerContext() {
  const context = makeContext({bindings: {entity: 1}});
  const census = context.record.graphs[0];
  census.nodes.find(node => node.id === 'finish')!.op = 'CensusCount';
  const catchment = structuredClone(census);
  catchment.id = 'catchment';
  catchment.nodes.find(node => node.id === 'finish')!.op = 'GPUCatchmentAccessibility';
  context.record.graphs = [...context.record.graphs, catchment];
  context.record.captures!.snapshots = [
    makeCapture({
      name: 'reachableFacilities',
      graph: census.id,
      elementType: 'u32',
      values: [4, 0xffffffff, 12]
    }),
    makeCapture({
      id: 'catchment-counts',
      name: 'result.reachableFacilities',
      graph: catchment.id,
      elementType: 'u32',
      values: [4, 0xffffffff, 12],
      role: 'output'
    })
  ];
  return context;
}

test('B5 same-record census and catchment counts retain their separate meanings', () => {
  const context = makeProducerContext();
  const table = TABLE_VIEW.build(context).sections.find(
    section => section.id === 'captures'
  )!.table;
  const rows = table.rows.map(row =>
    Object.fromEntries(table.columns.map((column, index) => [column.name, row[index]]))
  );
  expect(rows[0]['max']).toBe('4294967295');
  expect(rows[0]['value at 1']).toBe('4294967295');
  expect(rows[1]['max']).toBe('12');
  expect(rows[1]['value at 1']).toBe('unknown');
});

test('B6 catchment unknown words do not enter scales, marks or comparison arithmetic', () => {
  const context = makeProducerContext();
  const capture = context.record.captures!.snapshots[1];
  expect(resolveValueScale(context, capture.name)!.scale.domain).toEqual([4, 12]);
  const stripContext = {...context, record: structuredClone(context.record)};
  stripContext.record.captures!.snapshots = [{...capture, keySpace: undefined}];
  const tree = renderResultView(buildResultViewModel(stripContext), stripContext);
  const unknown = findViewNodes(
    tree,
    node => node.attributes['data-key'] === 1 && node.attributes['data-role'] === 'output'
  );
  expect(JSON.stringify(unknown)).toContain('no-data');
  const other = structuredClone(context.record);
  other.captures!.snapshots[1].values = [5, 3, 15];
  context.bindings = {...context.bindings, buffer: capture.name};
  context.compare = {record: other, label: 'A', referenceLabel: 'B'};
  const comparison = buildCompareViewModel(context);
  expect(Number.isNaN(comparison.delta[1])).toBe(true);
  expect(comparison.delta.filter(Number.isFinite)).toEqual([1, 3]);
  expect(comparison.maxAbsDelta).toBe(3);
  expect(comparison.changedCount).toBe(2);
  expect(Array.from(capture.values)).toEqual([4, 0xffffffff, 12]);
});

test('B7 duration-only timing has no timestamp ticks', () => {
  const context = makeLoopContext();
  context.record.metrics![0].nodes = {'relax-0': {cpuEncodeMs: 0, gpuTimeMs: 0.75}};
  const tree = renderTimingView(buildTimingViewModel(context), context);
  expect(getViewNodeText(tree)).toContain('0.75 ms');
  expect(
    findViewNodes(tree, node => node.tag === 'svg').flatMap(svg =>
      findViewNodes(svg, node => node.tag === 'text')
    )
  ).toEqual([]);
});

test('B8 duration evidence enables realized view while partial and indirect iterations stay unresolved', () => {
  const context = makeLoopContext();
  context.record.captures = undefined;
  context.record.metrics![0].series = undefined;
  const measurements: Record<string, GPUComputationNodeMetrics> = {
    'relax-0': {cpuEncodeMs: 0, gpuTimeMs: 0}
  };
  context.record.metrics![0].nodes = measurements;
  expect(REALIZED_VIEW.satisfies(context)).toEqual({ok: true});
  let model = buildRealizedViewModel(context);
  expect(model.loops[0].executedIterations).toBeUndefined();
  expect(model.loops[0].executedIterationsLowerBound).toBe(1);
  expect(REALIZED_VIEW.describe(model, context).keyNumber.value).toBe('unknown');
  measurements['relax-1'] = {cpuEncodeMs: 0, outcome: 'indirect'};
  model = buildRealizedViewModel(context);
  expect(model.loops[0].executedIterations).toBeUndefined();
  measurements['relax-1'] = {cpuEncodeMs: 0, outcome: 'indirect', gpuTimeMs: 2};
  expect(buildRealizedViewModel(context).loops[0].executedIterations).toBe(2);
  measurements['relax-1'] = {cpuEncodeMs: 0, outcome: 'skipped'};
  expect(buildRealizedViewModel(context).loops[0].executedIterations).toBe(1);
});

test('Earlier ROAD-111 description separates captured frames from unknown executions', () => {
  const context = makeLoopContext();
  context.record.metrics = undefined;
  context.record.captures!.snapshots = [makeCapture({iteration: 14})];
  context.record.captures!.stamps = [];
  const description = REALIZED_VIEW.describe(buildRealizedViewModel(context), context);
  expect(description.keyNumber.value).toBe('unknown');
  expect(description.facts).toContain('captured key frames 1');
});

test('B6 receipt neighborhoods do not color catchment unknowns as numeric counts', () => {
  const context = makeProducerContext();
  context.record.captures!.snapshots = [context.record.captures!.snapshots[1]];
  context.packs = [{id: 'test', neighbourhoods: {cells: {kind: 'stencil-3x3', keySpace: 'cells'}}}];
  const model = buildDatumReceiptModel(context);
  expect(model.neighbourhoods[0].cells!.find(cell => cell.key === 1)!.value).toBeUndefined();
});
