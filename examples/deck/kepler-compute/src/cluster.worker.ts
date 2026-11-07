// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getKeplerClusterResponseTransferables,
  handleKeplerClusterMessage,
  type KeplerClusterRequest,
  type KeplerClusterWorkerState
} from '@luma-contributors/kepler';

/** Builds the exact native supercluster index off the main thread and answers viewport queries. */
const state: KeplerClusterWorkerState = {};
const scope = self as unknown as {
  onmessage: ((event: MessageEvent<KeplerClusterRequest>) => void) | null;
  postMessage(message: unknown, transfer: ArrayBuffer[]): void;
};
scope.onmessage = event => {
  const response = handleKeplerClusterMessage(state, event.data);
  scope.postMessage(response, getKeplerClusterResponseTransferables(response));
};
