// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerClusterRanges, KeplerClusterView} from './kepler-cluster';
import {
  buildKeplerCompactClusterIndex,
  queryKeplerCompactClusterRanges,
  type KeplerCompactClusterIndex
} from './kepler-compact-cluster';

/**
 * Builds and queries the exact native supercluster index. The same pure logic runs inside the
 * example's Web Worker and inline in node tests, so both produce identical output.
 */
export type KeplerClusterEngine = {
  /**
   * (Re)builds the index. The first call supplies positions; later calls reuse them and only
   * change the filter mask or radius. Positions are owned by the engine afterwards.
   */
  build(request: {
    positions?: Float32Array;
    selectionMask?: Uint32Array;
    clusterRadius: number;
    /** Bound on retained hierarchy bytes; a larger hierarchy rejects with the budget message. */
    maximumHierarchyBytes?: number;
    /** Coarsest zoom built eagerly; coarser levels are built by the first query that needs them. */
    lowestZoom?: number;
  }): Promise<{hierarchyBytes: number}>;
  /** `hierarchyBytes` is the retained hierarchy after any level the query had to build. */
  query(
    view: KeplerClusterView,
    minClusterSize: number
  ): Promise<KeplerClusterRanges & {hierarchyBytes: number}>;
  destroy(): void;
};

/** Messages understood by {@link handleKeplerClusterMessage}. */
export type KeplerClusterRequest =
  | {
      type: 'build';
      requestId: number;
      positions?: Float32Array;
      selectionMask?: Uint32Array;
      clusterRadius: number;
      maximumHierarchyBytes?: number;
      lowestZoom?: number;
    }
  | {type: 'query'; requestId: number; view: KeplerClusterView; minClusterSize: number};

export type KeplerClusterResponse =
  | {type: 'built'; requestId: number; hierarchyBytes: number}
  | ({type: 'queried'; requestId: number; hierarchyBytes: number} & KeplerClusterRanges)
  | {type: 'failed'; requestId: number; message: string};

export type KeplerClusterWorkerState = {
  positions?: Float32Array;
  index?: KeplerCompactClusterIndex;
};

/** Pure message handler shared by the worker entry point and the inline engine. */
export function handleKeplerClusterMessage(
  state: KeplerClusterWorkerState,
  message: KeplerClusterRequest
): KeplerClusterResponse {
  try {
    if (message.type === 'build') {
      if (message.positions) state.positions = message.positions;
      // A build before positions arrive is a protocol error.
      if (!state.positions) throw new Error('Cluster build needs positions');
      state.index = undefined;
      state.index = buildKeplerCompactClusterIndex(
        {positions: state.positions, selectionMask: message.selectionMask},
        message.clusterRadius,
        message.maximumHierarchyBytes,
        message.lowestZoom
      );
      return {type: 'built', requestId: message.requestId, hierarchyBytes: state.index.byteLength};
    }
    // A query before the first build is a protocol error.
    if (!state.index) throw new Error('Cluster query needs a built index');
    const ranges = queryKeplerCompactClusterRanges(
      state.index,
      message.view,
      message.minClusterSize
    );
    return {
      type: 'queried',
      requestId: message.requestId,
      hierarchyBytes: state.index.byteLength,
      ...ranges
    };
  } catch (error) {
    return {
      type: 'failed',
      requestId: message.requestId,
      message: error instanceof Error ? error.message : String(error)
    };
  }
}

/** Typed arrays of a response that can move to the main thread without copying. */
export function getKeplerClusterResponseTransferables(
  response: KeplerClusterResponse
): ArrayBuffer[] {
  return response.type === 'queried'
    ? [response.centers.buffer, response.offsets.buffer, response.rows.buffer].filter(
        (buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer
      )
    : [];
}

/**
 * Message-based client engine: `post` sends a request (optionally transferring buffers) and
 * `subscribe` registers the response listener. Used with a Worker, or looped back inline.
 */
export function createKeplerClusterMessageEngine(transport: {
  post(message: KeplerClusterRequest, transfer: ArrayBuffer[]): void;
  subscribe(listener: (response: KeplerClusterResponse) => void): void;
  /** Reports a transport failure such as a worker that cannot load; fails every pending request. */
  subscribeError?(listener: (message: string) => void): void;
  close(): void;
}): KeplerClusterEngine {
  let nextRequestId = 1;
  const pending = new Map<
    number,
    {resolve(response: KeplerClusterResponse): void; reject(error: Error): void}
  >();
  transport.subscribe(response => {
    const entry = pending.get(response.requestId);
    pending.delete(response.requestId);
    if (!entry) return;
    if (response.type === 'failed') entry.reject(new Error(response.message));
    else entry.resolve(response);
  });
  transport.subscribeError?.(message => {
    for (const entry of pending.values()) entry.reject(new Error(message));
    pending.clear();
  });
  const send = (
    message: DistributiveOmit<KeplerClusterRequest, 'requestId'>,
    transfer: ArrayBuffer[]
  ) =>
    new Promise<KeplerClusterResponse>((resolve, reject) => {
      const requestId = nextRequestId++;
      pending.set(requestId, {resolve, reject});
      transport.post({...message, requestId} as KeplerClusterRequest, transfer);
    });
  return {
    async build(request) {
      const transfer: ArrayBuffer[] = [];
      if (request.positions?.buffer instanceof ArrayBuffer) transfer.push(request.positions.buffer);
      // The mask is copied so the caller keeps its own array.
      const response = await send({type: 'build', ...request}, transfer);
      return {hierarchyBytes: response.type === 'built' ? response.hierarchyBytes : 0};
    },
    async query(view, minClusterSize) {
      const response = await send({type: 'query', view, minClusterSize}, []);
      if (response.type !== 'queried') throw new Error('Unexpected cluster response');
      return {
        clusterCount: response.clusterCount,
        centers: response.centers,
        offsets: response.offsets,
        rows: response.rows,
        hierarchyBytes: response.hierarchyBytes
      };
    },
    destroy() {
      transport.close();
      for (const entry of pending.values()) entry.reject(new Error('The cluster engine closed.'));
      pending.clear();
    }
  };
}

type DistributiveOmit<Type, Key extends PropertyKey> = Type extends unknown
  ? Omit<Type, Key>
  : never;

/** Inline engine for node tests and hosts without Worker support; same handler, same output. */
export function createKeplerInlineClusterEngine(): KeplerClusterEngine {
  const state: KeplerClusterWorkerState = {};
  let listener: ((response: KeplerClusterResponse) => void) | undefined;
  return createKeplerClusterMessageEngine({
    post(message) {
      const response = handleKeplerClusterMessage(state, message);
      queueMicrotask(() => listener?.(response));
    },
    subscribe(next) {
      listener = next;
    },
    close() {
      state.index = undefined;
      state.positions = undefined;
    }
  });
}
