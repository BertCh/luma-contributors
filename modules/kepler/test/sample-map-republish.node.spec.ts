// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import type {KeplerTable} from '@kepler.gl/table';
import type {Layer} from '@kepler.gl/layers';
import {NativeAutoPrepare} from '../../../examples/deck/kepler-compute/src/native-auto-prepare';
import {DatasetOperationQueue} from '../../../examples/deck/kepler-compute/src/binding2-dataset-operations';

vi.mock('@kepler.gl/table', () => ({findPointFieldPairs: () => []}));

function createHarness() {
  let table = arrow.tableFromArrays({longitude: [-118], latitude: [34], value: [1]});
  const native = {
    id: 'sample',
    label: 'Sample',
    dataRevision: 1,
    fields: table.schema.fields.map(field => ({name: field.name, type: 'real'})),
    dataContainer: {numRows: () => table.numRows, getTable: () => table}
  } as unknown as KeplerTable;
  const queue = new DatasetOperationQueue();
  const queued = vi.spyOn(queue, 'run');
  let gpuValue = 0;
  let release: () => void = () => {};
  let gate = Promise.resolve();
  let ready = true;
  const publish = vi.fn(
    async (request: {datasetId: string; candidate: {table: arrow.Table; isCurrent(): boolean}}) => {
      await queue.run(request.datasetId, 'replace', async () => {
        await gate;
        if (!request.candidate.isCurrent()) throw new Error('Source changed before upload');
        gpuValue = Number(request.candidate.table.getChild('value')!.get(0));
      });
    }
  );
  const controller = new NativeAutoPrepare(
    {
      getDatasets: () => ({sample: native}),
      getLayers: () =>
        [
          {
            id: 'points',
            type: 'point',
            config: {
              dataId: 'sample',
              isVisible: true,
              columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}}
            }
          }
        ] as unknown as Layer[],
      isResidentDataset: () => true,
      isReady: () => ready,
      captureNativeConfig: () => undefined,
      publish
    },
    true
  );
  return {
    controller,
    publish,
    queued,
    get gpuValue() {
      return gpuValue;
    },
    setReady(value: boolean) {
      ready = value;
    },
    change(value: number) {
      table = arrow.tableFromArrays({longitude: [-118], latitude: [34], value: [value]});
      native.dataRevision++;
    },
    hold() {
      gate = new Promise<void>(resolve => {
        release = resolve;
      });
    },
    release() {
      release();
      gate = Promise.resolve();
    }
  };
}

test('resident revision reload explicitly prepares the current native rows and waits for queued GPU replacement', async () => {
  const harness = createHarness();
  harness.controller.evaluate();
  expect(harness.publish).not.toHaveBeenCalled();
  harness.change(42);
  harness.hold();
  const loading = harness.controller.republish('sample', new AbortController().signal);
  await vi.waitFor(() =>
    expect(harness.queued).toHaveBeenCalledWith('sample', 'replace', expect.any(Function))
  );
  expect(harness.controller.reports.get('sample')!.state).toBe('preparing');
  expect(harness.gpuValue).toBe(0);
  harness.release();
  await loading;
  expect(harness.gpuValue).toBe(42);
  expect(harness.controller.reports.get('sample')!.state).toBe('ready');
});

test('cancelled preparation cannot publish stale rows or overwrite a newer republish report', async () => {
  const harness = createHarness();
  harness.hold();
  const controller = new AbortController();
  const stale = harness.controller.republish('sample', controller.signal);
  const rejected = expect(stale).rejects.toThrow('cancelled');
  await vi.waitFor(() => expect(harness.publish).toHaveBeenCalledOnce());
  controller.abort(new Error('cancelled'));
  await rejected;
  harness.change(99);
  const fresh = harness.controller.republish('sample', new AbortController().signal);
  harness.release();
  await fresh;
  expect(harness.gpuValue).toBe(99);
  expect(harness.controller.reports.get('sample')!.state).toBe('ready');
});

test('a completed sample request does not leave an aborted signal on later native edits', async () => {
  const harness = createHarness();
  const controller = new AbortController();
  await harness.controller.republish('sample', controller.signal);
  controller.abort(new Error('next sample request'));
  harness.change(23);
  harness.controller.requestRepublish('sample');
  await vi.waitFor(() => expect(harness.gpuValue).toBe(23));
  expect(harness.controller.reports.get('sample')!.state).toBe('ready');
});

test('an explicit reload forces an already queued resident ID using the newest native revision', async () => {
  const harness = createHarness();
  harness.setReady(false);
  harness.controller.requestRepublish('sample');
  harness.change(71);
  const loading = harness.controller.republish('sample', new AbortController().signal);
  harness.setReady(true);
  harness.controller.evaluate();
  await loading;
  expect(harness.gpuValue).toBe(71);
  expect(harness.publish).toHaveBeenCalledOnce();
});
