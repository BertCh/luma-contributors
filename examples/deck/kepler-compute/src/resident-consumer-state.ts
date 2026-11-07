// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Consumers that must present a distinct, honest state instead of silently retaining old values. */
export type ResidentConsumerName =
  | 'tooltip'
  | 'table'
  | 'chart'
  | 'legend'
  | 'analysis'
  | 'extraction';

export type ResidentConsumerStatus =
  | 'unsupported'
  | 'pending'
  | 'empty'
  | 'stale'
  | 'error'
  | 'ready';

export type ResidentConsumerState<Value = unknown> =
  | {status: 'ready'; consumer: ResidentConsumerName; revision: string; value: Value}
  | {
      status: Exclude<ResidentConsumerStatus, 'ready'>;
      consumer: ResidentConsumerName;
      /** Revision the state was produced for, when known. */
      revision?: string;
      reason: string;
    };

export function makeReadyState<Value>(
  consumer: ResidentConsumerName,
  revision: string,
  value: Value
): ResidentConsumerState<Value> {
  return {status: 'ready', consumer, revision, value};
}

export function makeConsumerState(
  status: Exclude<ResidentConsumerStatus, 'ready'>,
  consumer: ResidentConsumerName,
  reason: string,
  revision?: string
): ResidentConsumerState<never> {
  return {status, consumer, reason, ...(revision === undefined ? {} : {revision})};
}

/**
 * A ready value is only current for the revision it was read at. A mismatch yields a stale state
 * that deliberately drops the value so callers cannot render it as current.
 */
export function reconcileConsumerState<Value>(
  state: ResidentConsumerState<Value>,
  currentRevision: string
): ResidentConsumerState<Value> {
  if (state.status !== 'ready' || state.revision === currentRevision) return state;
  return makeConsumerState(
    'stale',
    state.consumer,
    `Values were read at revision ${state.revision}; the display is now at ${currentRevision}.`,
    state.revision
  );
}

export function getConsumerStateMessage(state: ResidentConsumerState): string {
  switch (state.status) {
    case 'ready':
      return '';
    case 'pending':
      return state.reason || 'Reading current values...';
    case 'empty':
      return state.reason || 'No rows are available for the current selection.';
    case 'stale':
      return state.reason || 'These values are out of date and are being refreshed.';
    case 'unsupported':
      return state.reason || 'This feature is not supported for the resident data.';
    case 'error':
      return state.reason || 'The values could not be read.';
  }
}

/** Run an async read and classify the outcome, discarding results whose revision moved on. */
export async function readConsumerState<Value>(
  consumer: ResidentConsumerName,
  getRevision: () => string,
  read: () => Promise<Value | null | undefined>,
  isEmpty: (value: Value) => boolean = () => false
): Promise<ResidentConsumerState<Value>> {
  const revision = getRevision();
  try {
    const value = await read();
    if (getRevision() !== revision)
      return makeConsumerState('stale', consumer, 'The data changed while reading.', revision);
    if (value === null || value === undefined)
      return makeConsumerState('empty', consumer, 'The row is no longer available.', revision);
    if (isEmpty(value)) return makeConsumerState('empty', consumer, '', revision);
    return makeReadyState(consumer, revision, value);
  } catch (failure) {
    return makeConsumerState(
      'error',
      consumer,
      failure instanceof Error ? failure.message : String(failure),
      revision
    );
  }
}
