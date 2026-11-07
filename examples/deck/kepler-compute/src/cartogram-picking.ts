// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PickingInfo} from '@deck.gl/core';
import {isCartogramMercatorViewport} from './analysis-cartogram-projection';
import type {CartogramRow} from './cartogram-runtime';

/** Overlay identity is independent of native Kepler layers and geographic map coordinates. */
export type CartogramPick = {
  datasetId: string;
  sourceRowIndex: number;
  sourceRevision: number;
  analysisRevision: number;
  selectionRevision?: number;
  screenX: number;
  screenY: number;
  row: CartogramRow;
  isCurrent(): boolean;
};

export type CartogramPickingCallbacks = {
  onHover?(pick: CartogramPick | null): void;
  onClick?(pick: CartogramPick | null): void;
  onError?(error: unknown): void;
};

type Candidate = {sourceRowIndex: number; screenX: number; screenY: number; version: number};
type Channel = {
  version: number;
  running: boolean;
  published: boolean;
  pending?: Candidate;
  cached?: CartogramRow;
  notify?: (pick: CartogramPick | null) => void;
};

/**
 * Own one instance per published result snapshot and dispose it when that snapshot changes.
 * At most one read and one latest queued identity per event kind are retained. The runtime's
 * readRow owns GPU read fences and its 20-byte result read; this helper owns no source buffers.
 */
export function createCartogramPickingHandlers(
  options: CartogramPickingCallbacks & {
    datasetId: string;
    rowCount: number;
    revision: number;
    sourceRevision: number;
    selectionRevision?: number;
    isCurrent(): boolean;
    /** Defaults to the Web Mercator gate; declared-plane overlays draw in every viewport. */
    isViewportSupported?(viewport: NonNullable<PickingInfo['viewport']>): boolean;
    readRow(sourceRowIndex: number): Promise<CartogramRow | null>;
  }
): {
  onHover(information: PickingInfo): true;
  onClick(information: PickingInfo): true;
  dispose(): void;
} {
  let disposed = false;
  const hover: Channel = {version: 0, running: false, published: false, notify: options.onHover};
  const click: Channel = {version: 0, running: false, published: false, notify: options.onClick};
  const isCurrent = () => !disposed && options.isCurrent();
  const matches = (channel: Channel, candidate: Candidate) =>
    isCurrent() && channel.version === candidate.version;

  function publish(channel: Channel, candidate: Candidate, row: CartogramRow): void {
    if (!matches(channel, candidate)) return;
    channel.published = true;
    channel.notify?.({
      datasetId: options.datasetId,
      sourceRowIndex: candidate.sourceRowIndex,
      sourceRevision: options.sourceRevision,
      analysisRevision: options.revision,
      selectionRevision: options.selectionRevision,
      screenX: candidate.screenX,
      screenY: candidate.screenY,
      row,
      isCurrent: () => matches(channel, candidate)
    });
  }

  function clearPublished(channel: Channel): void {
    if (!channel.published) return;
    channel.published = false;
    channel.notify?.(null);
  }

  function latestMatchingCandidate(channel: Channel, candidate: Candidate): Candidate {
    if (channel.pending?.sourceRowIndex !== candidate.sourceRowIndex) return candidate;
    const latest = channel.pending;
    channel.pending = undefined;
    return latest;
  }

  async function drain(channel: Channel): Promise<void> {
    if (channel.running) return;
    channel.running = true;
    try {
      while (channel.pending && isCurrent()) {
        let candidate = channel.pending;
        channel.pending = undefined;
        try {
          const row = await options.readRow(candidate.sourceRowIndex);
          // Pointer motion over the same row reuses this read at the latest screen position.
          candidate = latestMatchingCandidate(channel, candidate);
          if (!matches(channel, candidate)) continue;
          if (
            !row ||
            row.sourceRowIndex !== candidate.sourceRowIndex ||
            row.sourceId !== candidate.sourceRowIndex ||
            row.sourceRevision !== options.sourceRevision ||
            row.analysisRevision !== options.revision ||
            row.centerX === null ||
            row.centerY === null ||
            row.radius === null ||
            !Number.isFinite(row.centerX) ||
            !Number.isFinite(row.centerY) ||
            !Number.isFinite(row.radius) ||
            row.radius <= 0
          ) {
            clearPublished(channel);
            continue;
          }
          channel.cached = row;
          publish(channel, candidate, row);
        } catch (error) {
          if (matches(channel, candidate)) {
            clearPublished(channel);
            options.onError?.(error);
          }
        }
      }
    } finally {
      channel.running = false;
      if (!isCurrent()) {
        channel.pending = undefined;
        channel.cached = undefined;
      }
    }
  }

  function handle(channel: Channel, information: PickingInfo): true {
    channel.version++;
    channel.pending = undefined;
    if (!channel.notify || disposed) return true;
    const sourceRowIndex = information.index;
    if (
      !isCurrent() ||
      !information.picked ||
      !information.viewport ||
      !(options.isViewportSupported ?? isCartogramMercatorViewport)(information.viewport) ||
      !Number.isSafeInteger(sourceRowIndex) ||
      sourceRowIndex < 0 ||
      sourceRowIndex >= options.rowCount ||
      !Number.isFinite(information.x) ||
      !Number.isFinite(information.y)
    ) {
      channel.cached = undefined;
      clearPublished(channel);
      return true;
    }
    const candidate: Candidate = {
      sourceRowIndex,
      screenX: information.x,
      screenY: information.y,
      version: channel.version
    };
    if (channel.cached?.sourceRowIndex === sourceRowIndex) {
      publish(channel, candidate, channel.cached);
    } else {
      channel.cached = undefined;
      clearPublished(channel);
      channel.pending = candidate;
      void drain(channel);
    }
    // This overlay has no native Kepler layer identity. Stop native hover/click propagation.
    return true;
  }

  return {
    onHover: information => handle(hover, information),
    onClick: information => handle(click, information),
    dispose: () => {
      if (disposed) return;
      disposed = true;
      for (const channel of [hover, click]) {
        channel.version++;
        channel.pending = undefined;
        channel.cached = undefined;
        clearPublished(channel);
      }
    }
  };
}
