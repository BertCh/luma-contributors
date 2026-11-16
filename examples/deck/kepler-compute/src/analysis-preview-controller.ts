// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AnalysisPreviewSession, type AnalysisPreviewRequest} from '@luma-contributors/kepler';
import {createAnalysisRunIdentity} from './analysis-run-identity';

/** Host adapter: bind every queued preview to the runtime, selected population and input revision. */
export function createAnalysisPreviewController<Runtime>(options: {
  session: AnalysisPreviewSession;
  getRuntime: () => Runtime;
  getRevisionKey: () => string;
}) {
  const runtime = options.getRuntime();
  let controller: AbortController | undefined;
  return {
    request(
      request: Omit<AnalysisPreviewRequest, 'revisionKey' | 'isCurrent'> & {
        isCurrent?: () => boolean;
      }
    ): Promise<boolean> {
      controller?.abort();
      controller = new AbortController();
      const currentController = controller;
      const abort = () => currentController.abort();
      request.signal?.addEventListener('abort', abort, {once: true});
      if (request.signal?.aborted) abort();
      const revisionKey = options.getRevisionKey();
      const identity = createAnalysisRunIdentity({
        signal: currentController.signal,
        getRuntime: options.getRuntime,
        getRevisionKey: options.getRevisionKey,
        isCurrent: () => options.getRuntime() === runtime && request.isCurrent?.() !== false
      });
      return options.session
        .request({
          ...request,
          signal: currentController.signal,
          revisionKey,
          isCurrent: identity.isCurrent
        })
        .finally(() => request.signal?.removeEventListener('abort', abort));
    },
    cancel(): void {
      controller?.abort();
      options.session.cancel();
    },
    destroy(): Promise<void> {
      controller?.abort();
      return options.session.destroy();
    }
  };
}
