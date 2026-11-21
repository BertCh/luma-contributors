// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {readFileSync} from 'node:fs';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {Buffer} from '@luma.gl/core';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {LensCommandGraph} from '../../src/computation-lenses/lens-view-snapshot';
import {uploadS2CountyShapes} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {createViewContext} from '../../src/computation-views/registry/view-registry';
import {buildDatumReceiptModel} from '../../src/computation-views/views/datum-receipt';
import {
  buildResultViewModel,
  renderResultView
} from '../../src/computation-views/views/result-view';
import {findViewNodes} from '../../src/computation-views/view-node';
import {makeV8Capture, makeV8ResultRecord} from '../computation-views/fixtures/v8-result-record';

test('R1 regenerated Cook County receipt retains class 3 and excludes coordinate aliases', () => {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../docs/lens-records/gpu-spatial-analysis.class-assignment.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const receipt = buildDatumReceiptModel(
    createViewContext(record, {
      bindings: {entity: 15},
      options: {keySpace: 'county'}
    })
  );
  expect(receipt.rows.find(row => row.name === 'classes')?.value).toBe('3');
  expect(receipt.rows.map(row => row.name)).not.toContain('counties');
  expect(receipt.rows.map(row => row.name)).not.toContain('classMap');
  const twin = JSON.parse(
    readFileSync(
      new URL(
        '../../../../docs/lens-records/twins/gpu-spatial-analysis.class-assignment~datum~d15.json',
        import.meta.url
      ),
      'utf8'
    )
  ) as {receipt: {rows: string[][]}};
  expect(twin.receipt.rows.find(row => row[1] === 'classes')?.[2]).toBe('3');
  expect(twin.receipt.rows.map(row => row[1])).not.toContain('counties');
  expect(twin.receipt.rows.map(row => row[1])).not.toContain('classMap');
});

test('county display rings keep vertex receipts separate from picked county scores', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new LensCommandGraph(device);
    const valuesByResource = new Map<string, number[]>();
    const outline = uploadS2CountyShapes(
      {
        own: resource => resource,
        view(name, values, format, length) {
          valuesByResource.set(name, Array.from(values));
          const buffer = graph.createTransientBuffer({
            id: name,
            byteLength: values.byteLength,
            usage: Buffer.STORAGE | Buffer.COPY_SRC
          });
          return graph.createDataView(buffer, {
            format,
            length: length ?? values.length / (format.endsWith('x2') ? 2 : 1)
          });
        }
      },
      // Two open triangle rings have six input vertices, eight closed display vertices,
      // and two feature rows. County 1 starts at vertex 4, not vertex 1.
      {vertices: [0, 0, 1, 0, 0, 1, 10, 10, 11, 10, 10, 11], ringOffsets: [0, 3, 6]}
    );
    const painted = outline.paint('scores');
    const record = makeV8ResultRecord();
    record.axes.keySpaces = [{id: 'county', label: 'county', size: 2}, ...outline.keySpaces];
    record.captures!.snapshots = [
      ...Object.entries({...outline.captures, painted}).map(([name, {view, ...metadata}]) =>
        makeV8Capture(name, valuesByResource.get(view.buffer.id)!, {
          ...metadata,
          components: view.format.endsWith('x2') ? 2 : 1
        })
      ),
      makeV8Capture('scores', [7, 13], {role: 'output', keySpace: 'county'})
    ];
    const context = createViewContext(record, {
      bindings: {entity: 1},
      options: {keySpace: 'county', primaryOutputCaptureId: 'painted-id'}
    });
    const model = buildResultViewModel(context);
    expect(model.output.unknown).toBeUndefined();
    expect(model.output.count).toBe(2);
    const paths = findViewNodes(
      renderResultView(model, context),
      node =>
        node.tag === 'path' &&
        node.attributes['data-capture-id'] === 'painted-id' &&
        node.attributes['data-action'] === 'pick'
    );
    expect(
      paths.map(path => [path.attributes['data-key-space'], path.attributes['data-key']])
    ).toEqual([
      ['county', 0],
      ['county', 1]
    ]);
    const receipt = buildDatumReceiptModel(context);
    expect(receipt.rows.map(row => [row.name, row.value])).toEqual([['scores', '13']]);
    expect(outline.captures['counties'].keySpace).toBe('county-vertex');
    expect(painted.keySpace).toBe('county-vertex');
    expect(painted.geometry).toHaveProperty('featureKeySpace', 'county');
  } finally {
    device.destroy();
  }
});
