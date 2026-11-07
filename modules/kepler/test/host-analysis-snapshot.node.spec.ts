// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';

test('analysis snapshot reads retain real result storage through reentrant teardown and reject stale reads', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'snapshot',
    table: arrow.tableFromArrays({value: new Float32Array([1, 2, 3])})
  });
  const sourceBytes = session.registry.memoryUsage.totalBytes;
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  let finishRead!: () => void;
  const readCompletion = new Promise<void>(resolve => {
    finishRead = resolve;
  });
  let snapshotCurrent = true;
  let readCleanedUp = false;
  let destructionResolved = false;
  let destruction: Promise<void> | undefined;
  try {
    analysis.setMode('z-score');
    const snapshot = analysis.getTableColumns();
    if (!snapshot) throw new Error('Expected analysis result columns');
    const resultBuffer = snapshot.columns['analysis_z_score'].data[0].buffer;
    // Only encoding freshness is mocked: output storage, reservations and teardown are real.
    vi.spyOn(analysis, 'getTableColumns').mockReturnValue({
      ...snapshot,
      isCurrent: () => snapshotCurrent
    });
    const staleRead = vi.fn(async () => 'unexpected');
    snapshotCurrent = false;
    await expect(analysis.withResultSnapshot(staleRead)).rejects.toThrow('stale');
    snapshotCurrent = true;
    const superseded = analysis.withResultSnapshot(staleRead);
    snapshotCurrent = false;
    await expect(superseded).rejects.toThrow('stale');
    expect(staleRead).not.toHaveBeenCalled();

    snapshotCurrent = true;
    const retainedBytes = session.registry.memoryUsage.totalBytes;
    const pending = analysis.withResultSnapshot(async retainedSnapshot => {
      expect(retainedSnapshot.columns['analysis_z_score'].data[0].buffer).toBe(resultBuffer);
      // Teardown starts inside the callback: the read must already be registered.
      destruction = analysis.destroy(Promise.resolve()).then(() => {
        destructionResolved = true;
      });
      await readCompletion;
      expect(resultBuffer.destroyed).toBe(false);
      readCleanedUp = true;
      return 'download-ready';
    });
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    expect(destruction).toBeDefined();
    expect(destructionResolved).toBe(false);
    expect(readCleanedUp).toBe(false);
    expect(resultBuffer.destroyed).toBe(false);
    expect(session.registry.memoryUsage.totalBytes).toBe(retainedBytes);

    finishRead();
    await expect(pending).resolves.toBe('download-ready');
    await destruction;
    expect(readCleanedUp).toBe(true);
    expect(destructionResolved).toBe(true);
    expect(resultBuffer.destroyed).toBe(true);
    expect(session.registry.memoryUsage.totalBytes).toBe(sourceBytes);
    expect(dataset.value.columns['value'].data[0].buffer.destroyed).toBe(false);
  } finally {
    finishRead();
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
