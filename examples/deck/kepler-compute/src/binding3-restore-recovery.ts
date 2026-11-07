// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isKeplerAllocationError} from '@luma-contributors/kepler';
import {RestoreTransactionError} from './restore-transaction';

/** Raised when a multi-source restore stopped on an allocation failure; lists every source that cannot fit. */
export class RestoreAllocationError extends RestoreTransactionError {
  constructor(
    message: string,
    readonly failedSourceIds: readonly string[],
    base: RestoreTransactionError
  ) {
    super(
      message,
      base.failedSourceId,
      base.rolledBackSourceIds,
      base.rollbackFailures,
      base.previousWorkspaceRestored,
      {cause: base.cause}
    );
    this.name = 'RestoreAllocationError';
  }
}

/** Typed memory-budget, reservation and device out-of-memory failures (no message matching). */
export function isAllocationFailure(error: unknown): boolean {
  return isKeplerAllocationError(error);
}

/**
 * First-fit admission against the bytes a restore may use: sources that do not fit the remaining
 * budget are reported, later smaller ones may still fit. Used after rollback to tell the user which
 * sources need more memory instead of naming only the first one that failed.
 */
export function findSourcesExceedingBudget(
  sources: readonly {sourceId: string; bytes: number}[],
  availableBytes: number
): string[] {
  let remaining = availableBytes;
  const exceeding: string[] = [];
  for (const source of sources) {
    if (source.bytes <= remaining) remaining -= source.bytes;
    else exceeding.push(source.sourceId);
  }
  return exceeding;
}

/**
 * Wraps a rolled-back restore failure that was an allocation failure with the full list of sources
 * that cannot be allocated (always including the one that threw). Other failures pass through.
 */
export function describeRestoreAllocationFailure(
  error: unknown,
  sources: readonly {sourceId: string; bytes: number}[],
  availableBytes: number
): unknown {
  if (!(error instanceof RestoreTransactionError) || !isAllocationFailure(error.cause ?? error))
    return error;
  const failed = new Set(findSourcesExceedingBudget(sources, availableBytes));
  if (error.failedSourceId) failed.add(error.failedSourceId);
  const failedSourceIds = [...failed];
  return new RestoreAllocationError(
    `${error.message}; sources that could not be allocated: ${failedSourceIds.join(', ')}`,
    failedSourceIds,
    error
  );
}
