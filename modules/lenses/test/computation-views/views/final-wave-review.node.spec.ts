// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';
import {fromRecordedLensFile, toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {resolveValueScale} from '../../../src/computation-views/views/value-strip';
import {
  buildDatumReceiptModel,
  DATUM_RECEIPT_VIEW
} from '../../../src/computation-views/views/datum-receipt';
import {getSymbolValueAt} from '../../../src/computation-views/views/formula-tex';
import {
  buildBufferStripModel,
  BUFFER_STRIP_VIEW
} from '../../../src/computation-views/views/buffer-strip';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';
import {D8_STENCIL_VIEW} from '../../../src/gpu-terrain/computation-pack/views/d8-stencil-view';
import {coarsenCaptureText} from '../../../src/computation-views/encoding/crispness';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';
import {makeRecord, makeCapture} from '../fixtures/v4-record';

function capture(name: string, values: number[], metadata: Partial<GPUComputationCapture> = {}) {
  return makeY2Capture(name, values, {
    id: `A-${name}-3`,
    graph: 'A',
    afterNode: 'step',
    iteration: 3,
    keySpace: 'rowsA',
    ...metadata
  });
}

function scopedRecord(snapshots: GPUComputationCapture[]) {
  const record = makeY2Record(snapshots);
  record.graphs = ['A', 'B'].map(id => ({
    id,
    form: 'compiled',
    domain: 'gpu-core',
    nodes: [
      {
        id: 'step',
        namespace: `${id}/step`,
        label: id,
        kind: 'compute',
        op: 'Calculation',
        reads: [],
        writes: ['sum', 'count', 'scores', ...(id === 'A' ? ['soil'] : [])],
        attrs: {}
      }
    ],
    edges: [],
    resources: ['sum', 'count', 'scores', ...(id === 'A' ? ['soil'] : [])].map(resource => ({
      id: resource,
      kind: 'buffer',
      bytes: 16,
      bytesKind: 'logical',
      ownership: 'imported'
    }))
  }));
  record.axes.axes[0].keySpace = 'rowsA';
  record.axes.keySpaces = ['rowsA', 'rowsB'].map(id => ({id, label: id, size: 9}));
  return record;
}

function decode(record: ReturnType<typeof makeY2Record>) {
  return fromRecordedLensFile(
    JSON.parse(
      JSON.stringify(
        toRecordedLensFile(record, {
          scenarioId: 'gpu-core.cards',
          recordedAt: record.createdAt,
          adapterLabel: 'fixture'
        })
      )
    )
  ).record;
}

function points(metadata: Partial<GPUComputationCapture> = {}) {
  return capture('points', [0, 0, 1, 1], {
    resource: 'soil',
    role: 'output',
    components: 2,
    geometry: {
      kind: 'points',
      coordinateModel: 'projected',
      countCapture: 'count',
      valuesCapture: 'scores'
    },
    ...metadata
  });
}

test('B3/S2 named companions are invariant under snapshot order within graph and frame', () => {
  const snapshots = [
    points(),
    capture('count', [2]),
    capture('scores', [14, 22]),
    capture('count', [1], {id: 'B-count', graph: 'B', keySpace: 'rowsB'}),
    capture('scores', [90, 91], {id: 'B-scores', graph: 'B', keySpace: 'rowsB'}),
    capture('count', [1], {id: 'old-count', iteration: 1})
  ];
  for (const ordered of [snapshots, [...snapshots].reverse()]) {
    const record = decode(scopedRecord(ordered));
    const model = buildResultViewModel(createViewContext(record));
    expect(model.output.count).toBe(2);
    expect(model.output.points).toEqual([
      [0, 0],
      [1, 1]
    ]);
    expect(Array.from(model.output.values!.values)).toEqual([14, 22]);
    expect(model.output.scale?.domain).toEqual([14, 22]);
    expect(model.output.scale?.variable).toBe('scores');
  }
  const inferred = scopedRecord([
    points({graph: undefined}),
    capture('count', [2]),
    capture('scores', [14, 22])
  ]);
  expect(buildResultViewModel(createViewContext(inferred)).output.points).toEqual([
    [0, 0],
    [1, 1]
  ]);
});

test('B3 stale frames, cross-graph explicit IDs and ambiguous observations cannot certify counts', () => {
  for (const variant of ['stale', 'explicit', 'ambiguous'] as const) {
    const point = points();
    const count = capture('count', [1]);
    if (variant === 'stale') count.iteration = 1;
    if (variant === 'explicit') {
      count.graph = 'B';
      point.geometry = {kind: 'points', coordinateModel: 'projected', countCapture: count.id};
    }
    if (variant === 'ambiguous') {
      point.graph = undefined;
      point.resource = undefined;
      point.afterNode = undefined;
    }
    expect(
      buildResultViewModel(createViewContext(scopedRecord([point, count]))).output.count,
      variant
    ).toBe('unknown');
  }
});

test('B4 point panels and automatic scales share logical capacity and active windows', () => {
  for (const keySpace of [undefined, 'rowsA']) {
    const record = scopedRecord([
      points({graph: undefined, keySpace}),
      capture('count', [3]),
      capture('scores', [14, 22])
    ]);
    const context = createViewContext(record);
    const model = buildResultViewModel(context);
    expect(model.output.count).toBe('unknown');
    expect(
      findViewNodes(renderResultView(model, context), node => node.tag === 'circle')
    ).toHaveLength(0);
    expect(resolveValueScale(context, 'scores')).toBeUndefined();
    record.captures!.snapshots[1].values = [2];
    expect(resolveValueScale(context, 'scores')?.scale.domain).toEqual([14, 22]);
    record.captures!.snapshots[1].values = [3];
    expect(resolveValueScale(context, 'scores')).toBeUndefined();
  }
});

test('B3/S2 simultaneous raster join graph/frame certificates survive reordering', () => {
  const record = scopedRecord([
    points({
      geometry: {kind: 'points', coordinateModel: 'projected', valuesCapture: 'joinPointZones'}
    }),
    capture('joinPointZones', [3, 4], {resource: 'sum', elementType: 'u32'}),
    capture('joinPointZones', [8, 9], {
      id: 'old-zones',
      resource: 'sum',
      elementType: 'u32',
      iteration: 1
    }),
    capture('joinPointZones', [90, 91], {
      id: 'other-zones',
      resource: 'sum',
      elementType: 'u32',
      graph: 'B'
    }),
    capture('joinIncomplete', [0], {resource: 'certificate', elementType: 'u32'}),
    capture('joinIncomplete', [1], {
      id: 'old-cert',
      resource: 'certificate',
      elementType: 'u32',
      iteration: 1
    }),
    capture('joinIncomplete', [1], {
      id: 'other-cert',
      resource: 'certificate',
      elementType: 'u32',
      graph: 'B'
    })
  ]);
  for (const graph of record.graphs) {
    graph.nodes[0].op = 'GPURasterJoin';
    graph.nodes[0].bindings = [{name: 'pointZones', resource: 'sum', access: 'write'}];
    graph.resources = [
      ...(graph.resources ?? []),
      {id: 'certificate', kind: 'buffer', bytes: 4, ownership: 'imported'}
    ];
    graph.nodes = [
      ...graph.nodes,
      {
        id: 'step-producer-incomplete',
        label: 'certificate',
        namespace: '',
        kind: 'compute',
        op: 'RasterProducerCompleteness',
        attrs: {},
        reads: [],
        writes: ['certificate'],
        bindings: [{name: 'incomplete', resource: 'certificate', access: 'write'}]
      }
    ];
  }
  for (const ordered of [record.captures!.snapshots, [...record.captures!.snapshots].reverse()]) {
    const loaded = decode({...record, captures: {...record.captures!, snapshots: ordered}});
    expect(
      Array.from(buildResultViewModel(createViewContext(loaded)).output.values!.values)
    ).toEqual([3, 4]);
  }
});

test('B5 worked receipt and symbols keep the observed graph and frame', () => {
  const record = scopedRecord([
    capture('sumA', [2], {resource: 'sum'}),
    capture('sumB', [90], {id: 'B-sum', graph: 'B', keySpace: 'rowsB', resource: 'sum'}),
    capture('sumA', [7], {id: 'old-sum', iteration: 1, resource: 'sum'})
  ]);
  record.graphs = [...record.graphs].reverse();
  for (const graph of record.graphs) graph.nodes[0].formula = `formula${graph.id}`;
  record.refs.formulas = ['A', 'B'].map(id => ({
    id: `formula${id}`,
    tex: '\\term{sum}{s}',
    ascii: 's',
    derivation: 'curated',
    symbols: [{id: 'sum', role: 'output', binding: 'resource:sum'}]
  }));
  const loaded = decode(record);
  const context = createViewContext(loaded, {bindings: {entity: 0}, options: {keySpace: 'rowsA'}});
  const receipt = buildDatumReceiptModel(context);
  expect(receipt.formula?.formulaId).toBe('formulaA');
  expect(receipt.formula?.workedTex).toContain('2');
  expect(receipt.formula?.workedTex).not.toContain('90');
  expect(
    getSymbolValueAt(loaded, {id: 'sum', role: 'output', binding: 'resource:sum'}, 0)
  ).toBeUndefined();
});

function archive(name: string) {
  return fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../../docs/lens-records/${name}.json`, import.meta.url),
        'utf8'
      )
    )
  ).record;
}

test('B6 renamed real raster join ports still require a certificate; unrelated names stay numeric', () => {
  const record = archive('gpu-raster.raster-join');
  const assignment = record.captures!.snapshots.find(
    candidate => candidate.name === 'joinPointZones'
  )!;
  assignment.name = 'assignedZones';
  const point = record.captures!.snapshots.find(
    candidate => candidate.geometry?.kind === 'points' && candidate.role === 'output'
  )!;
  if (!point.geometry || point.geometry.kind !== 'points')
    throw new Error('Expected point geometry');
  point.geometry = {...point.geometry, valuesCapture: assignment.id};
  expect(buildResultViewModel(createViewContext(record)).output.values).toBeDefined();
  record.captures!.snapshots = record.captures!.snapshots.filter(
    candidate => candidate.name !== 'joinIncomplete'
  );
  expect(buildResultViewModel(createViewContext(record)).output.values).toBeUndefined();
  const receipt = buildDatumReceiptModel(
    createViewContext(record, {bindings: {entity: 0}, options: {keySpace: assignment.keySpace!}})
  );
  expect(receipt.rows.find(row => row.name === 'assignedZones')!.value).toBe('unknown');
  const unrelated = scopedRecord([
    points(),
    capture('joinPointZones', [2, 6], {resource: 'sum', elementType: 'u32'})
  ]);
  unrelated.graphs[0].nodes[0].op = 'CensusCount';
  unrelated.captures!.snapshots[0].geometry = {
    kind: 'points',
    coordinateModel: 'projected',
    valuesCapture: 'joinPointZones'
  };
  expect(
    Array.from(buildResultViewModel(createViewContext(unrelated)).output.values!.values)
  ).toEqual([2, 6]);
});

test('B7 join counts obey loss and recovery without hiding certified zeros', () => {
  const record = archive('gpu-raster.raster-join');
  const count = record.captures!.snapshots.find(candidate => candidate.name === 'joinZoneCounts')!;
  const outside = record.captures!.snapshots.find(
    candidate => candidate.name === 'joinOutsideCount'
  )!;
  const certificate = record.captures!.snapshots.find(
    candidate => candidate.name === 'joinIncomplete'
  )!;
  certificate.name = 'guardFlag'; // Certificate identity follows its producer binding, not this label.
  for (const incomplete of [1, 0, 1, 0]) {
    certificate.values = [incomplete];
    count.values = [incomplete ? 0xffffffff : 0];
    outside.values = [incomplete ? 0xffffffff : 0];
    const context = createViewContext(record, {
      bindings: {entity: 0, buffer: outside.id},
      options: {keySpace: count.keySpace!}
    });
    expect(buildDatumReceiptModel(context).rows.find(row => row.name === count.name)!.value).toBe(
      incomplete ? 'unknown' : '0'
    );
    const buffer = buildBufferStripModel(context);
    expect(buffer.stats.finiteCount).toBe(incomplete ? 0 : 1);
    expect(buffer.table[0].value).toBe(incomplete ? 'unknown' : 0);
  }
});

function catchmentFrames() {
  const record = makeRecord();
  record.graphs[0].nodes[1].op = 'GPUCatchmentAccessibility';
  record.captures!.snapshots = [0, 1].map(iteration =>
    makeCapture({
      id: `counts-${iteration}`,
      name: 'reachableFacilities',
      iteration,
      elementType: 'u32',
      values: [0xffffffff, 0xffffffff]
    })
  );
  return record;
}

test('B8 provisional unknowns survive Buffer, Datum and exported receipts', () => {
  const record = catchmentFrames();
  const context = createViewContext(record, {bindings: {entity: 0, buffer: 'counts-0'}});
  const buffer = buildBufferStripModel(context);
  const bufferTree = BUFFER_STRIP_VIEW.render(buffer, context);
  const bufferRows = findViewNodes(
    bufferTree,
    node => node.tag === 'tr' && node.attributes['data-provisional'] === 'provisional'
  );
  expect(getViewNodeText(bufferRows[0])).toContain('unknown');
  expect(getViewNodeText(bufferRows[0])).not.toContain('NaN');
  expect(BUFFER_STRIP_VIEW.describe(buffer, context).keyNumber.value).toBe('unavailable');
  expect(getViewNodeText(bufferTree)).not.toContain('max NaN');
  const datum = buildDatumReceiptModel(context);
  const datumRows = findViewNodes(
    DATUM_RECEIPT_VIEW.render(datum, context),
    node => node.tag === 'tr' && node.attributes['data-provisional'] === 'provisional'
  );
  expect(getViewNodeText(datumRows[0])).toContain('unknown');
  expect(DATUM_RECEIPT_VIEW.receipt!(datum, context)!.rows[0][2]).toBe('unknown');
  expect(coarsenCaptureText('unknown, 1234', 'provisional')).toBe('unknown, 1200');
});

test('B9 unknown pairs do not become numerical changes or a zero delta', () => {
  const record = catchmentFrames();
  record.captures!.snapshots[1].values = [8, 0xffffffff];
  const context = createViewContext(record, {bindings: {buffer: 'counts-1'}});
  const model = buildBufferStripModel(context);
  expect(model.diff?.comparedCount).toBe(0);
  expect(model.diff?.changedCount).toBe(0);
  expect(Number.isNaN(model.diff!.maxAbsDelta)).toBe(true);
  expect(getViewNodeText(BUFFER_STRIP_VIEW.render(model, context))).toContain(
    'no comparable numeric keys'
  );
});

test('B10 actual terrain-flow registration admits its numbered snapshot and receiver', () => {
  const record = archive('gpu-terrain.terrain-flow');
  const context = createViewContext(record, {bindings: {entity: 1826}});
  expect(D8_STENCIL_VIEW.isActive!(context)).toBe(true);
  expect(D8_STENCIL_VIEW.satisfies(context)).toEqual({ok: true});
  expect(D8_STENCIL_VIEW.build(context).chosenDirection).toBe(1); // ESRI receiver 2 = southeast.
  const rasters = record.captures!.snapshots.filter(capture => capture.geometry?.kind === 'raster');
  const next = rasters.map(capture => ({
    ...capture,
    id: `${capture.id}-next`,
    iteration: 1,
    values: Array.from(capture.values)
  }));
  next.find(capture => capture.meaning?.startsWith('D8'))!.values[1826] = 4;
  record.captures!.snapshots = [...record.captures!.snapshots, ...next];
  expect(D8_STENCIL_VIEW.build(context).chosenDirection).toBe(2);
  expect(D8_STENCIL_VIEW.build({...context, options: {iteration: 0}}).chosenDirection).toBe(1);
});

test('S3 interpretation metadata work is independent of the number of buffer lanes', () => {
  const reads: number[] = [];
  for (const length of [2, 2048]) {
    const record = catchmentFrames();
    record.captures!.snapshots = [record.captures!.snapshots[0]];
    record.captures!.snapshots[0].values = new Uint32Array(length).fill(8);
    let operatorReads = 0;
    Object.defineProperty(record.graphs[0].nodes[1], 'op', {
      get: () => {
        operatorReads++;
        return 'GPUCatchmentAccessibility';
      }
    });
    const context = createViewContext(record, {bindings: {buffer: 'counts-0'}});
    const model = buildBufferStripModel(context);
    BUFFER_STRIP_VIEW.render(model, context);
    reads.push(operatorReads);
  }
  expect(reads[1]).toBe(reads[0]);
  expect(reads[1]).toBeLessThan(20);
});
