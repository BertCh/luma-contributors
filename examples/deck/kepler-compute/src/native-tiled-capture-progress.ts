// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {planImageTiles} from './native-tiled-capture';

import {clamp} from '@math.gl/core';

/** Pure helpers for the "Large image (tiled)" section of the Export Image modal. */

export type TiledImageSummary = {
  width: number;
  height: number;
  megapixels: number;
  tileCount: number;
  columns: number;
  rows: number;
  /** RGBA bytes held by the stitch canvas. */
  stitchBytes: number;
};

/** Describe a tiled export, or return the validation message when it is out of bounds. */
export function summarizeTiledImage(
  width: number,
  height: number,
  tileWidth: number,
  tileHeight: number
): {summary: TiledImageSummary} | {error: string} {
  try {
    const tiles = planImageTiles(width, height, tileWidth, tileHeight);
    return {
      summary: {
        width,
        height,
        megapixels: Math.round((width * height) / 1e4) / 100,
        tileCount: tiles.length,
        columns: Math.ceil(width / tileWidth),
        rows: Math.ceil(height / tileHeight),
        stitchBytes: width * height * 4
      }
    };
  } catch (error) {
    return {error: error instanceof Error ? error.message : String(error)};
  }
}

export function formatByteCount(bytes: number): string {
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(1)} GiB`;
  return `${Math.round(bytes / 2 ** 20)} MiB`;
}

export function formatTiledProgress(
  completed: number,
  count: number,
  elapsedMilliseconds: number
): string {
  const base = `Rendered tile ${completed} of ${count}`;
  if (completed < 1 || completed >= count || elapsedMilliseconds <= 0) return base;
  const remainingSeconds = Math.ceil(
    ((elapsedMilliseconds / completed) * (count - completed)) / 1000
  );
  return `${base} (about ${remainingSeconds} s left)`;
}

export function getTiledProgressFraction(completed: number, count: number): number {
  return count > 0 ? clamp(completed / count, 0, 1) : 0;
}

export function isTiledCaptureCancelled(error: unknown): boolean {
  return error instanceof Error && error.name === 'TiledCaptureCancelledError';
}
