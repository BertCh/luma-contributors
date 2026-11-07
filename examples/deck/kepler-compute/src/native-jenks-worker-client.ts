// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeNativeJenksBreaks,
  type NativeJenksWorkerRequest,
  type NativeJenksWorkerResponse,
  type NativeNaturalBreaks
} from './native-jenks-breaks';

/**
 * Computes exact native breaks in a module Web Worker, transferring the values. Without Worker
 * support (Node, tests) the same pure function runs in place.
 */
export async function computeJenksBreaksInWorker(
  request: NativeJenksWorkerRequest,
  loadNaturalBreaks: () => Promise<NativeNaturalBreaks>
): Promise<number[]> {
  let response: NativeJenksWorkerResponse;
  if (typeof Worker === 'undefined') {
    response = computeNativeJenksBreaks(request, await loadNaturalBreaks());
  } else {
    const worker = new Worker(new URL('./native-jenks.worker.ts', import.meta.url), {
      type: 'module'
    });
    try {
      response = await new Promise<NativeJenksWorkerResponse>((resolve, reject) => {
        worker.onmessage = event => resolve(event.data as NativeJenksWorkerResponse);
        worker.onerror = event => reject(new Error(event.message || 'Jenks worker failed'));
        worker.postMessage(request, [request.values.buffer]);
      });
    } finally {
      worker.terminate();
    }
  }
  if ('error' in response) throw new Error(response.error);
  return response.breaks;
}
