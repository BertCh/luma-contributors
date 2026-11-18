// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  COMPARE_VIEW,
  buildCompareViewModel,
  renderCompareView
} from '../../../src/computation-views/views/compare-view';
import {getRunScaleRegistry} from '../../../src/computation-views/views/value-strip';
import {viewNodeToHtml, findViewNodes} from '../../../src/computation-views/view-node';
import {makeContext, makeRecord, makeCapture} from '../fixtures/v4-record';
import {
  buildBufferStripModel,
  renderBufferStrip
} from '../../../src/computation-views/views/buffer-strip';
import {buildDatumReceiptModel} from '../../../src/computation-views/views/datum-receipt';
import {
  assertValidLensRecord,
  assertValidLensSidecars
} from '../../../src/computation-lenses/validate-lens-record';

describe('CompareView', () => {
  test('H4 refuses incompatible raster geometry shapes on otherwise compatible validated records', () => {
    const context = makeContext({bindings: {buffer: 'result'}});
    const other = makeRecord();
    for (const [record, values, shape] of [
      [context.record, [1, 2, 3, 4], [2, 2]],
      [other, [10, 20, 30, 40], [4, 1]]
    ] as const) {
      record.axes.keySpaces = [{id: 'cells', label: 'cells', size: 4}];
      record.captures!.snapshots = [makeCapture({values, geometry: {kind: 'raster', shape}})];
      expect(() => assertValidLensRecord(record)).not.toThrow();
      expect(() => assertValidLensSidecars(record, record)).not.toThrow();
    }
    context.compare = {record: other, label: 'A', referenceLabel: 'B'};
    expect(COMPARE_VIEW.satisfies(context)).toMatchObject({ok: false});
    expect(() => buildCompareViewModel(context)).toThrow();
    other.captures!.snapshots[0].geometry = {kind: 'raster', shape: [2, 2]};
    expect(COMPARE_VIEW.satisfies(context)).toEqual({ok: true});
    expect(buildCompareViewModel(context).delta).toEqual([9, 18, 27, 36]);
    for (const geometry of [
      {rowDirection: 'north' as const},
      {cellSize: 2},
      {origin: [10, 0] as const},
      {coordinateModel: 'projected' as const},
      {unit: 'm'}
    ]) {
      other.captures!.snapshots[0].geometry = {kind: 'raster', shape: [2, 2], ...geometry};
      expect(COMPARE_VIEW.satisfies(context)).toMatchObject({ok: false});
    }
    other.captures!.snapshots[0].geometry = {
      kind: 'raster',
      shape: [2, 2],
      rowDirection: 'south',
      cellSize: 1,
      origin: [0, 2],
      coordinateModel: 'unknown'
    };
    expect(COMPARE_VIEW.satisfies(context)).toEqual({ok: true});
  });
  test('G2 compares every vector component and labels the scalar strip component', () => {
    const context = makeContext({bindings: {buffer: 'result'}});
    context.record.captures!.snapshots = [
      makeCapture({
        id: 'first',
        iteration: 0,
        meaning: 'first:8',
        components: 2,
        values: [10, 12, 11, 13]
      }),
      makeCapture({
        id: 'second',
        iteration: 1,
        meaning: 'first:8',
        components: 2,
        values: [10, 92, 11, 93]
      })
    ];
    const compare = buildCompareViewModel(context);
    const buffer = buildBufferStripModel(context).diff!;
    expect(compare).toMatchObject({changedCount: 2, equalCount: 0, maxAbsDelta: 80});
    expect(compare.changedCount).toBe(buffer.changedCount);
    expect(compare.maxAbsDelta).toBe(buffer.maxAbsDelta);
    expect(compare.delta).toEqual([0, 0]);
    expect(viewNodeToHtml(renderCompareView(compare, context))).toContain('component 0');
  });
  test('G9 refuses cross-run captures from unrelated domains and incompatible layouts', () => {
    const context = makeContext({bindings: {buffer: 'rainfall'}});
    context.record.captures!.snapshots = [
      makeCapture({name: 'rainfall', keySpace: 'stations', meaning: 'first:8', values: [1, 2]})
    ];
    context.record.axes.keySpaces = [{id: 'stations', label: 'stations', size: 10}];
    context.record.axes.axes[0].keySpace = 'stations';
    const other = makeRecord();
    other.runId = 'other-run';
    other.captures!.runId = other.runId;
    other.metrics = other.metrics?.map(metrics => ({...metrics, runId: other.runId}));
    other.axes.keySpaces = [{id: 'cells', label: 'cells', size: 10}];
    other.captures!.snapshots = [
      makeCapture({name: 'rainfall', keySpace: 'cells', meaning: 'first:8', values: [10, 20]})
    ];
    context.compare = {record: other, label: 'A', referenceLabel: 'B'};
    for (const record of [context.record, other]) {
      expect(() => assertValidLensRecord(record)).not.toThrow();
      expect(() => assertValidLensSidecars(record, record)).not.toThrow();
    }
    expect(COMPARE_VIEW.satisfies(context)).toMatchObject({ok: false});
    expect(() => buildCompareViewModel(context)).toThrow();
    other.axes.keySpaces = [...context.record.axes.keySpaces];
    other.axes.axes[0].keySpace = 'stations';
    other.captures!.snapshots[0].keySpace = 'stations';
    other.captures!.snapshots[0].components = 2;
    other.captures!.snapshots[0].values = [10, 12, 20, 22];
    expect(COMPARE_VIEW.satisfies(context)).toMatchObject({ok: false});
    other.captures!.snapshots[0].components = 1;
    other.captures!.snapshots[0].values = [10, 20];
    expect(COMPARE_VIEW.satisfies(context)).toEqual({ok: true});
    expect(buildCompareViewModel(context).delta).toEqual([9, 18]);
  });
  test('G9 buffer differences also refuse mismatched iteration domains or components', () => {
    for (const previous of [
      makeCapture({id: 'first', iteration: 0, keySpace: 'stations', values: [1, 2]}),
      makeCapture({id: 'first', iteration: 0, components: 2, values: [1, 2, 3, 4]})
    ]) {
      const context = makeContext({bindings: {buffer: 'result'}});
      context.record.captures!.snapshots = [previous, makeCapture({values: [10, 20]})];
      expect(COMPARE_VIEW.satisfies(context)).toMatchObject({ok: false});
      expect(buildBufferStripModel(context).diff).toBeUndefined();
      expect(buildBufferStripModel(context).envelope).toBeUndefined();
    }
  });
  for (const [firstA, firstB, expectedKeys, expectedDelta] of [
    [8, 8, [8, 9], [10, 10]],
    [8, 9, [9], [9]],
    [8, 10, [], []]
  ] as const) {
    test(`F1 compares absolute window intersection ${firstA}/${firstB}`, () => {
      const context = makeContext({bindings: {buffer: 'result'}});
      context.record.captures!.snapshots = [
        makeCapture({id: 'first', iteration: 0, meaning: `first:${firstA}`, values: [10, 11]}),
        makeCapture({id: 'second', iteration: 1, meaning: `first:${firstB}`, values: [20, 21]})
      ];
      const model = buildCompareViewModel(context);
      expect(model.delta).toEqual(expectedDelta);
      expect(model).toMatchObject({
        changedCount: expectedKeys.length,
        equalCount: 0,
        maxAbsDelta: expectedDelta.length ? Math.max(...expectedDelta) : 0
      });
      const keys = findViewNodes(
        renderCompareView(model, context),
        node => node.attributes['data-action'] === 'pick'
      ).map(node => Number(node.attributes['data-key']));
      expect(keys).toEqual(expectedKeys);
      expect(buildBufferStripModel(context).diff).toMatchObject({
        comparedCount: expectedKeys.length,
        changedCount: expectedKeys.length
      });
    });
  }
  test('F1 buffer and receipt use absolute keys for windowed vector captures', () => {
    const context = makeContext({bindings: {entity: 8, buffer: 'result'}});
    context.record.axes = {
      ...context.record.axes,
      keySpaces: [
        {...context.record.axes.keySpaces[0], size: 10, indexSpace: '1d', shape: undefined}
      ]
    };
    context.record.captures!.snapshots = [
      makeCapture({meaning: 'first:8', components: 2, values: [10, 12, 11, 13]})
    ];
    const model = buildBufferStripModel(context);
    expect(model.table.map(row => [row.key, row.value])).toEqual([
      [8, '10, 12'],
      [9, '11, 13']
    ]);
    expect(
      findViewNodes(
        renderBufferStrip(model, context),
        node => node.attributes['data-action'] === 'pick'
      ).map(node => Number(node.attributes['data-key']))
    ).toContain(8);
    expect(buildDatumReceiptModel(context).rows[0].value).toBe('10, 12');
    expect(
      findViewNodes(
        renderBufferStrip(model, context),
        node => node.attributes['data-action'] === 'pick'
      ).every(node => node.attributes['data-key-space'] === 'cells')
    ).toBe(true);
  });
  test('handles unchanged and nonfinite references without inventing finite deltas', () => {
    const context = makeContext();
    context.record.captures!.snapshots = [
      makeCapture({id: 'first', iteration: 0, values: [2, NaN, Infinity]}),
      makeCapture({values: [2, NaN, Infinity]})
    ];
    const model = buildCompareViewModel(context);
    expect(model).toMatchObject({changedCount: 0, equalCount: 3, maxAbsDelta: 0});
    expect(model.delta[0]).toBe(0);
    expect(model.delta[1]).toBeNaN();
    expect(model.delta[2]).toBeNaN();
    expect(viewNodeToHtml(renderCompareView(model, context))).toContain('midpoint 0 = unchanged');
  });
  test('selects last two iterations, computes B minus A and honours explicit iterations', () => {
    const context = makeContext();
    expect(COMPARE_VIEW.satisfies(context)).toEqual({ok: true});
    const model = buildCompareViewModel(context);
    expect(model).toMatchObject({
      name: 'result',
      labelA: 'iteration 0',
      labelB: 'iteration 1',
      mode: 'delta',
      count: 9,
      changedCount: 9,
      equalCount: 0,
      maxAbsDelta: 1,
      midpoint: 0,
      provenance: 'derived'
    });
    expect(model.delta).toEqual(new Array(9).fill(1));
    expect(
      buildCompareViewModel({...context, options: {iterationA: 1, iterationB: 0}}).delta
    ).toEqual(new Array(9).fill(-1));
    expect(COMPARE_VIEW.describe(model, context).keyNumber).toEqual({
      label: 'keys changed',
      value: 9
    });
    context.record.captures!.snapshots = [makeCapture()];
    expect(COMPARE_VIEW.satisfies(context)).toEqual({
      ok: false,
      reason: 'needs two references (k vs k′, run A vs B, or GPU vs replay)'
    });
  });
  test('prefers external references, then replay versus readback', () => {
    const context = makeContext({bindings: {buffer: 'result'}});
    context.record.captures!.replays = [
      makeCapture({id: 'replay', provenance: 'gpu-replay', values: new Array(9).fill(0)})
    ];
    expect(buildCompareViewModel(context)).toMatchObject({labelA: 'GPU replay', labelB: 'run'});
    const reference = makeRecord();
    reference.captures!.snapshots = [makeCapture({values: new Array(9).fill(100)})];
    expect(
      buildCompareViewModel({
        ...context,
        compare: {record: reference, label: 'A', referenceLabel: 'B'}
      })
    ).toMatchObject({labelA: 'A', labelB: 'B', delta: new Array(9).fill(100)});
  });
  test('registers a diverging zero-centred scale and renders deterministic derived marks', () => {
    const context = makeContext();
    const model = buildCompareViewModel(context);
    const html = viewNodeToHtml(renderCompareView(model, context));
    const scale = getRunScaleRegistry(context)
      .list()
      .find(candidate => candidate.variable === 'delta:result');
    expect(scale).toMatchObject({kind: 'diverging', midpoint: 0, ramp: 'rdbu', domain: [-1, 1]});
    expect(html).toContain('midpoint 0 = unchanged');
    expect(html).toContain('data-provenance="derived"');
    expect(html).toContain('data-action="pick"');
    expect(html).not.toContain('<td></td>');
    expect(html).toBe(viewNodeToHtml(renderCompareView(model, context)));
  });
  test('juxtaposes references on one shared value scale', () => {
    const context = makeContext({options: {mode: 'juxtapose'}});
    const model = buildCompareViewModel(context);
    const tree = renderCompareView(model, context);
    expect(
      findViewNodes(tree, node => node.attributes['class'] === 'lens-value-strip')
    ).toHaveLength(2);
    expect(
      getRunScaleRegistry(context)
        .list()
        .map(scale => scale.variable)
    ).toEqual(['result']);
  });
});
