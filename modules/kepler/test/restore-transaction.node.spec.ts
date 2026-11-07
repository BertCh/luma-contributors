// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  RestoreTransactionError,
  describePreviousWorkspaceRestore,
  preflightRestoreReservation,
  runRestoreTransaction
} from '../../../examples/deck/kepler-compute/src/restore-transaction';

describe('restore transaction', () => {
  it('commits every source in order', async () => {
    const log: string[] = [];
    const result = await runRestoreTransaction({
      sources: ['a', 'b', 'c'],
      getSourceId: source => source,
      createSource: source => {
        log.push(`create ${source}`);
        return {undo: () => void log.push(`undo ${source}`)};
      }
    });
    expect(result.createdSourceIds).toEqual(['a', 'b', 'c']);
    expect(log).toEqual(['create a', 'create b', 'create c']);
  });

  it('undoes created sources in reverse order and restores the previous workspace', async () => {
    const log: string[] = [];
    const failure = runRestoreTransaction({
      sources: ['a', 'b', 'c'],
      getSourceId: source => source,
      createSource: source => {
        if (source === 'c') throw new Error('reservation failed');
        log.push(`create ${source}`);
        return {undo: () => void log.push(`undo ${source}`)};
      },
      restorePreviousWorkspace: () => void log.push('restore previous')
    });
    await expect(failure).rejects.toMatchObject({
      name: 'RestoreTransactionError',
      failedSourceId: 'c',
      rolledBackSourceIds: ['b', 'a'],
      previousWorkspaceRestored: true
    });
    expect(log).toEqual(['create a', 'create b', 'undo b', 'undo a', 'restore previous']);
  });

  it('keeps rolling back when one undo fails and reports it', async () => {
    const error = await runRestoreTransaction({
      sources: ['a', 'b', 'c'],
      getSourceId: source => source,
      createSource: source => {
        if (source === 'c') throw new Error('device lost');
        return {
          undo: () => {
            if (source === 'b') throw new Error('stuck');
          }
        };
      },
      restorePreviousWorkspace: () => {
        throw new Error('no checkpoint');
      }
    }).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(RestoreTransactionError);
    const typed = error as RestoreTransactionError;
    expect(typed.rolledBackSourceIds).toEqual(['a']);
    expect(typed.rollbackFailures.map(item => item.sourceId)).toEqual(['b']);
    expect(typed.previousWorkspaceRestored).toBe(false);
    expect((typed.cause as Error).message).toBe('device lost');
  });

  it('rejects before creating anything when the total reservation does not fit', async () => {
    let created = 0;
    const failure = runRestoreTransaction({
      sources: [40, 40, 40],
      getSourceId: source => String(source),
      estimateBytes: source => source,
      availableBytes: 100,
      createSource: () => {
        created++;
        return {undo() {}};
      }
    });
    await expect(failure).rejects.toThrow(/only 100 are available/);
    expect(created).toBe(0);
    expect(preflightRestoreReservation([1, 2], value => value, 3).fits).toBe(true);
  });

  it('rolls back when aborted between sources', async () => {
    const controller = new AbortController();
    const undone: string[] = [];
    const failure = runRestoreTransaction({
      sources: ['a', 'b'],
      getSourceId: source => source,
      signal: controller.signal,
      createSource: source => {
        if (source === 'a') controller.abort();
        return {undo: () => void undone.push(source)};
      }
    });
    await expect(failure).rejects.toBeInstanceOf(RestoreTransactionError);
    expect(undone).toEqual(['a']);
  });

  it('describes whether the previous workspace was rebuilt', () => {
    const cause = new RestoreTransactionError(
      'Restoring b failed (boom); 1 created source was removed',
      'b',
      ['a'],
      [],
      undefined,
      {cause: new Error('boom')}
    );
    const restored = describePreviousWorkspaceRestore(cause, true);
    expect(restored.message).toContain('the previous workspace was rebuilt from its snapshot');
    expect(restored.previousWorkspaceRestored).toBe(true);
    expect(restored.rolledBackSourceIds).toEqual(['a']);
    const failed = describePreviousWorkspaceRestore(
      new Error('x'),
      false,
      new Error('rebuild broke')
    );
    expect(failed.message).toContain('rebuild broke');
    expect(failed.previousWorkspaceRestored).toBe(false);
  });
});
