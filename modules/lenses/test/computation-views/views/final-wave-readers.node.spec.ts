// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {buildDispatchViewModel} from '../../../src/computation-views/views/dispatch-view';
import {buildCommandGraphViewModel} from '../../../src/computation-views/views/command-graph-view';
import {buildDatumReceiptModel} from '../../../src/computation-views/views/datum-receipt';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {findViewNodes} from '../../../src/computation-views/view-node';
import {makeRecord, makeCapture} from '../fixtures/v4-record';
import {fromRecordedLensFile, toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {buildBufferStripModel} from '../../../src/computation-views/views/buffer-strip';
import {getSymbolValueAt} from '../../../src/computation-views/views/formula-tex';

function makeSharedNodeRecord() {
  const record = makeRecord();
  const original = record.graphs[0];
  const node = {
    ...original.nodes[1],
    id: 'step',
    namespace: 'A/step',
    reads: [],
    writes: ['soil'],
    dispatch: {workgroups: [1, 1, 1] as [number, number, number]},
    workgroupSize: [1, 1, 1] as [number, number, number]
  };
  record.graphs = [
    {
      ...original,
      id: 'A',
      nodes: [node],
      edges: [],
      resources: [
        {id: 'soil', kind: 'buffer', bytes: 16, bytesKind: 'logical', ownership: 'imported'}
      ]
    },
    {
      ...original,
      id: 'B',
      form: 'lowered',
      nodes: [{...node, namespace: 'B/step', writes: []}],
      edges: [],
      resources: []
    }
  ];
  record.captures!.snapshots = [
    makeCapture({
      graph: undefined,
      afterNode: 'step',
      resource: 'soil',
      iteration: undefined,
      name: 'id-probe',
      values: [7],
      keySpace: 'cells'
    })
  ];
  return record;
}

test('W5-21 inferred captures belong only to their unique graph; contradictions stay unattributed', () => {
  const record = makeSharedNodeRecord();
  const context = createViewContext(record, {bindings: {entity: 0}});
  expect(buildDispatchViewModel(context).idProbe).toHaveProperty('captureId');
  expect(buildCommandGraphViewModel(context).previews.some(preview => preview.hasCaptures)).toBe(
    true
  );
  expect(buildDatumReceiptModel(context).rows[0].stage).toBe('A');
  const other = {...context, options: {form: 'logical'}, bindings: {node: 'B:step'}};
  expect(buildDispatchViewModel(other).idProbe).toEqual({status: 'not-recorded'});
  expect(buildCommandGraphViewModel(other).previews.every(preview => !preview.hasCaptures)).toBe(
    true
  );
  record.captures!.snapshots[0].graph = 'B';
  expect(buildDispatchViewModel(other).idProbe).toEqual({status: 'not-recorded'});
  record.captures!.snapshots[0].graph = undefined;
  record.captures!.snapshots[0].resource = undefined;
  expect(buildDispatchViewModel(context).idProbe).toEqual({status: 'not-recorded'});
});

test('W5-21 inferred point captures enforce logical resource capacity', () => {
  const record = makeSharedNodeRecord();
  record.captures!.snapshots = [
    makeCapture({
      graph: undefined,
      afterNode: 'step',
      resource: 'soil',
      iteration: undefined,
      role: 'output',
      components: 2,
      keySpace: undefined,
      values: [0, 0, 1, 1],
      geometry: {kind: 'points', coordinateModel: 'projected', countCapture: 'count'}
    }),
    makeCapture({
      id: 'count',
      name: 'count',
      resource: undefined,
      graph: 'A',
      afterNode: 'step',
      iteration: undefined,
      keySpace: undefined,
      values: [3]
    })
  ];
  const panel = buildResultViewModel(createViewContext(record)).output;
  // 16 logical bytes / (2 components * 4 bytes) = 2 points; count 3 is invalid.
  expect(panel.count).toBe('unknown');
  expect(panel.points).toEqual([]);
  record.captures!.snapshots[0].keySpace = 'cells'; // Its nine-row domain cannot override two logical slots.
  expect(buildResultViewModel(createViewContext(record)).output.count).toBe('unknown');
  record.captures!.snapshots[0].graph = 'B'; // Contradicts resource soil, which exists only in A.
  expect(buildResultViewModel(createViewContext(record)).output.count).toBe('unknown');
});

test('RT-137 loaded raster join assignments require a certificate in their graph and frame', () => {
  const record = makeRecord();
  record.graphs = [...record.graphs, {...structuredClone(record.graphs[0]), id: 'other-graph'}];
  const graph = record.graphs[0];
  graph.resources = [
    ...(graph.resources ?? []),
    {id: 'point-zones', kind: 'buffer', bytes: 8, ownership: 'imported'},
    {id: 'join-certificate', kind: 'buffer', bytes: 4, ownership: 'imported'}
  ];
  graph.nodes[1].op = 'GPURasterJoin';
  graph.nodes[1].writes = [...graph.nodes[1].writes, 'point-zones'];
  graph.nodes[1].bindings = [{name: 'pointZones', resource: 'point-zones', access: 'write'}];
  graph.nodes = [
    ...graph.nodes,
    {
      id: 'finish-producer-incomplete',
      label: 'certificate',
      namespace: '',
      kind: 'compute',
      op: 'RasterProducerCompleteness',
      attrs: {},
      reads: [],
      writes: ['join-certificate'],
      bindings: [{name: 'incomplete', resource: 'join-certificate', access: 'write'}]
    }
  ];
  record.captures!.snapshots = [
    makeCapture({
      id: 'points',
      name: 'joinedPoints',
      role: 'output',
      components: 2,
      values: [0, 0, 1, 1],
      geometry: {kind: 'points', coordinateModel: 'projected', valuesCapture: 'joinPointZones'}
    }),
    makeCapture({
      id: 'zones',
      name: 'joinPointZones',
      resource: 'point-zones',
      values: [0, 1],
      elementType: 'u32'
    }),
    makeCapture({
      id: 'certificate',
      name: 'joinIncomplete',
      resource: 'join-certificate',
      afterNode: 'finish-producer-incomplete',
      values: [0],
      keySpace: undefined,
      elementType: 'u32'
    }),
    makeCapture({id: 'other', name: 'temperature', values: [8, 9], role: 'input'})
  ];
  const envelope = toRecordedLensFile(record, {
    scenarioId: 'gpu-raster.raster-join',
    recordedAt: record.createdAt,
    adapterLabel: 'fixture'
  });
  const loaded = fromRecordedLensFile(JSON.parse(JSON.stringify(envelope))).record;
  const certificates = loaded.captures!.snapshots;
  const inspect = () => {
    const context = createViewContext(loaded, {bindings: {entity: 0}});
    const result = buildResultViewModel(context);
    return {
      result,
      receipt: buildDatumReceiptModel(context),
      tree: renderResultView(result, context)
    };
  };
  for (const incomplete of [0, 1, 0]) {
    certificates[2].values = [incomplete];
    const {result, receipt, tree} = inspect();
    expect(Boolean(result.output.values)).toBe(incomplete === 0);
    expect(getSymbolValueAt(loaded, {id: 'joinPointZones', role: 'output'}, 0)?.value).toBe(
      incomplete ? undefined : 0
    );
    expect(receipt.rows.find(row => row.name === 'joinPointZones')!.value).toBe(
      incomplete ? 'unknown' : '0'
    );
    const assignedMarks = findViewNodes(
      tree,
      node =>
        node.tag === 'circle' &&
        node.attributes['data-role'] === 'output' &&
        String(node.attributes['class']).split(' ').includes('cv-v')
    );
    expect(assignedMarks).toHaveLength(incomplete ? 0 : 2);
    expect(
      Array.from(result.panels.find(panel => panel.capture.name === 'temperature')!.values!.values)
    ).toEqual([8, 9]);
  }
  certificates[2].iteration = 0;
  expect(inspect().result.output.values).toBeUndefined();
  certificates[2].iteration = 1;
  certificates[2].graph = 'other-graph';
  expect(inspect().result.output.values).toBeUndefined();
  loaded.captures!.snapshots = certificates.filter(capture => capture.id !== 'certificate');
  expect(inspect().receipt.rows.find(row => row.name === 'joinPointZones')!.value).toBe('unknown');
});

test('PTS-130 buffer aggregates exclude only producer-scoped unknown lanes and preserve bytes', () => {
  const record = makeRecord();
  record.graphs[0].nodes[1].op = 'GPUCatchmentAccessibility';
  const capture = makeCapture({
    name: 'reachableFacilities',
    elementType: 'u32',
    values: new Uint32Array([4, 0xffffffff, 12])
  });
  record.captures!.snapshots = [capture];
  const bytes = Array.from(new Uint8Array((capture.values as Uint32Array).buffer));
  const context = createViewContext(record, {bindings: {buffer: capture.id, entity: 1}});
  const model = buildBufferStripModel(context);
  expect(model.stats.min).toBe(4);
  expect(model.stats.max).toBe(12);
  expect(model.stats.finiteCount).toBe(2);
  expect(model.stats.nanCount).toBe(0);
  expect(model.stats.unknownCount).toBe(1);
  expect(model.histogram.counts.reduce((sum, count) => sum + count, 0)).toBe(2);
  expect(model.table[1].value).toBe('unknown');
  expect(Array.from(new Uint8Array((capture.values as Uint32Array).buffer))).toEqual(bytes);
  record.graphs[0].nodes[1].op = 'CensusCount';
  const genuine = buildBufferStripModel(context);
  expect(genuine.stats.max).toBe(4294967295);
  expect(genuine.stats.finiteCount).toBe(3);
  expect(genuine.table[1].value).toBe(4294967295);
});
