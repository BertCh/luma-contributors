// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Middleware, UnknownAction} from 'redux';
import {ActionTypes} from '@kepler.gl/actions';
import {
  cancelSampleMapLoads,
  isSampleMapMutationPending,
  whenSampleMapMutationsSettled
} from './sample-map-lifecycle';
import {unwrapForInstance} from './resident-action-guards';
import {isSampleRecord} from './sample-map-catalog';

const FINISH_SAMPLE_MAP = 'LUMA_SAMPLE_MAP_FINISHED';
let nextRequestId = 0;
type Pending = {isCurrent(): boolean; finish(results: unknown[]): void; fail(error: unknown): void};
const pending = new Map<number, Pending>();

/** Observe Kepler's specific CREATE/UPDATE_TABLE_TASK result, not an already-present table ID. */
export function commitNativeSampleMap(
  dispatch: (onFinish: {type: string; requestId: number}) => void,
  signal: AbortSignal,
  isCurrent: () => boolean
): Promise<void> {
  signal.throwIfAborted();
  const requestId = ++nextRequestId;
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      signal.removeEventListener('abort', abort);
      pending.delete(requestId);
    };
    const fail = (error: unknown) => {
      cleanup();
      reject(error);
    };
    const abort = () => fail(signal.reason);
    pending.set(requestId, {
      isCurrent,
      fail,
      finish(results) {
        const failed = results.find(
          result => !isSampleRecord(result) || result['status'] !== 'fulfilled'
        );
        if (failed) {
          fail(
            new Error(
              `Native sample registration failed: ${isSampleRecord(failed) ? String(failed['reason']) : 'invalid task result'}`
            )
          );
        } else {
          cleanup();
          resolve();
        }
      }
    });
    signal.addEventListener('abort', abort, {once: true});
    try {
      dispatch({type: FINISH_SAMPLE_MAP, requestId});
    } catch (error) {
      fail(error);
    }
  });
}

/** Late cancelled task results must not overwrite a rolled-back or replacement workspace. */
export const sampleMapNativeMiddleware: Middleware = () => next => action => {
  const native = unwrapForInstance(action as UnknownAction, 'compute');
  if (native?.type === ActionTypes.LOAD_FILES || native?.type === ActionTypes.RESET_MAP_CONFIG) {
    cancelSampleMapLoads('Sample loading was superseded by an explicit native load or reset.');
    if (isSampleMapMutationPending())
      return whenSampleMapMutationsSettled().then(() => next(action));
  }
  const payload = native?.['payload'];
  const onFinish = isSampleRecord(payload) ? payload['onFinish'] : undefined;
  if (
    !isSampleRecord(payload) ||
    !Array.isArray(payload['results']) ||
    !isSampleRecord(onFinish) ||
    onFinish['type'] !== FINISH_SAMPLE_MAP ||
    typeof onFinish['requestId'] !== 'number'
  )
    return next(action);
  const request = pending.get(onFinish['requestId']);
  if (!request) return action;
  if (!request.isCurrent()) {
    request.fail(new Error('The GPU workspace changed during native sample registration.'));
    return action;
  }
  const result = next(action);
  request.finish(payload['results']);
  return result;
};
