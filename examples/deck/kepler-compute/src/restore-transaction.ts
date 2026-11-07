// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Raised when staged restoration fails after rollback has run; `cause` is the original failure. */
export class RestoreTransactionError extends Error {
  constructor(
    message: string,
    readonly failedSourceId: string | undefined,
    readonly rolledBackSourceIds: readonly string[],
    readonly rollbackFailures: readonly {sourceId: string; error: unknown}[],
    readonly previousWorkspaceRestored: boolean | undefined,
    options: {cause: unknown}
  ) {
    super(message, options);
    this.name = 'RestoreTransactionError';
  }
}

export type RestoreStaged = {
  /** Reverse of the creation: remove the publication, native dataset and any reservation. */
  undo(): void | Promise<void>;
};

export type RestoreTransactionProps<Source> = {
  sources: readonly Source[];
  getSourceId(source: Source): string;
  /** Upper-bound bytes the source needs while it is created (CPU retention plus GPU upload). */
  estimateBytes?(source: Source): number;
  /** Bytes still available for the whole restoration; checked before anything is replaced. */
  availableBytes?: number;
  /** Creates one source and returns how to undo it. Throwing triggers rollback. */
  createSource(source: Source, index: number): RestoreStaged | Promise<RestoreStaged>;
  /** Optional best-effort return to the previous workspace after every staged source is undone. */
  restorePreviousWorkspace?(): void | Promise<void>;
  signal?: AbortSignal;
};

export type RestorePreflight = {totalBytes: number; availableBytes: number; fits: boolean};

/** Sums source reservations so the whole bundle is admitted or rejected before the old workspace changes. */
export function preflightRestoreReservation<Source>(
  sources: readonly Source[],
  estimateBytes: (source: Source) => number,
  availableBytes: number
): RestorePreflight {
  const totalBytes = sources.reduce((total, source) => total + estimateBytes(source), 0);
  return {totalBytes, availableBytes, fits: totalBytes <= availableBytes};
}

/**
 * Stages sources one at a time. When source k fails (memory reservation, device loss, validation or
 * abort), sources k-1..0 are undone in reverse order and the previous workspace is restored when a
 * callback is supplied, so a failed restore never leaves a half-restored workspace.
 */
export async function runRestoreTransaction<Source>(
  props: RestoreTransactionProps<Source>
): Promise<{createdSourceIds: string[]}> {
  if (props.estimateBytes && props.availableBytes !== undefined) {
    const preflight = preflightRestoreReservation(
      props.sources,
      props.estimateBytes,
      props.availableBytes
    );
    // Rejected before any creation so the current workspace is untouched.
    if (!preflight.fits)
      throw new RestoreTransactionError(
        `Bundle needs about ${preflight.totalBytes.toLocaleString()} bytes of memory but only ${preflight.availableBytes.toLocaleString()} are available; the current workspace was not changed`,
        undefined,
        [],
        [],
        undefined,
        {cause: preflight}
      );
  }
  const staged: {sourceId: string; entry: RestoreStaged}[] = [];
  let failedSourceId: string | undefined;
  try {
    for (const [index, source] of props.sources.entries()) {
      props.signal?.throwIfAborted();
      failedSourceId = props.getSourceId(source);
      const entry = await props.createSource(source, index);
      staged.push({sourceId: failedSourceId, entry});
      props.signal?.throwIfAborted();
    }
    return {createdSourceIds: staged.map(item => item.sourceId)};
  } catch (error) {
    const rolledBackSourceIds: string[] = [];
    const rollbackFailures: {sourceId: string; error: unknown}[] = [];
    for (const item of staged.reverse()) {
      try {
        await item.entry.undo();
        rolledBackSourceIds.push(item.sourceId);
      } catch (undoError) {
        rollbackFailures.push({sourceId: item.sourceId, error: undoError});
      }
    }
    let previousWorkspaceRestored: boolean | undefined;
    if (props.restorePreviousWorkspace) {
      try {
        await props.restorePreviousWorkspace();
        previousWorkspaceRestored = true;
      } catch {
        previousWorkspaceRestored = false;
      }
    }
    const reason = error instanceof Error ? error.message : String(error);
    throw new RestoreTransactionError(
      `Restoring ${failedSourceId ?? 'the workspace'} failed (${reason}); ${rolledBackSourceIds.length} created ${rolledBackSourceIds.length === 1 ? 'source was' : 'sources were'} removed${
        rollbackFailures.length ? `, ${rollbackFailures.length} could not be removed` : ''
      }${
        previousWorkspaceRestored === undefined
          ? ''
          : previousWorkspaceRestored
            ? '; the previous workspace was restored'
            : '; the previous workspace could not be restored'
      }`,
      failedSourceId,
      rolledBackSourceIds,
      rollbackFailures,
      previousWorkspaceRestored,
      {cause: error}
    );
  }
}

/**
 * Re-labels a failed restore after the host tried to rebuild the previous workspace from a snapshot,
 * so the user message says whether their previous workspace is back or could not be rebuilt.
 */
export function describePreviousWorkspaceRestore(
  error: unknown,
  restored: boolean,
  rebuildFailure?: unknown
): RestoreTransactionError {
  const original = error instanceof RestoreTransactionError ? error : undefined;
  const reason = error instanceof Error ? error.message : String(error);
  const suffix = restored
    ? '; the previous workspace was rebuilt from its snapshot'
    : `; the previous workspace could not be rebuilt (${
        rebuildFailure instanceof Error
          ? rebuildFailure.message
          : String(rebuildFailure ?? 'no snapshot')
      }) and only the observations fixture remains`;
  return new RestoreTransactionError(
    `${reason.replace(/; the (current|previous) workspace[^;]*$/, '')}${suffix}`,
    original?.failedSourceId,
    original?.rolledBackSourceIds ?? [],
    original?.rollbackFailures ?? [],
    restored,
    {cause: error}
  );
}
