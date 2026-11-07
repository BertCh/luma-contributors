// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WebMercatorViewport} from '@math.gl/web-mercator';
import {
  packAcceptedLabelRows,
  placeLabelsGreedy,
  type LabelCollisionCandidate
} from './label-collision';

/** CPU anchors of one label source: longitude/latitude pairs per row and optional validity words. */
export type LabelAnchorSource = {
  /** Interleaved longitude, latitude per source row. */
  positions: Float32Array | Float64Array;
  /** Optional one uint32 per row; zero marks a row without a usable anchor (hidden). */
  validity?: Uint32Array;
};

export type LabelCollisionViewport = {
  longitude: number;
  latitude: number;
  zoom: number;
  width: number;
  height: number;
  pitch?: number;
  bearing?: number;
  /** Project with Deck's globe viewport (GPU collision only; the CPU projector is Web Mercator). */
  globe?: boolean;
};

export type LabelVisibilityOptions = {
  rowCount: number;
  anchors: LabelAnchorSource;
  /** Four floats per row: widest line advance and paragraph height in atlas pixels. */
  rowBounds: Float32Array;
  viewport: LabelCollisionViewport;
  /** CSS size and atlas font size; their ratio scales the atlas-pixel bounds. */
  size: number;
  fontSize: number;
  anchor: 'start' | 'middle' | 'end';
  alignment: 'top' | 'center' | 'bottom';
  /** Constant per-row radius and its pixel scale, as in the label shader. */
  radius: number;
  radiusPixelScale: number;
  radiusPadding: number;
  verticalPadding: number;
  /** Extra clear pixels kept between labels. */
  padding?: number;
  /** Larger priorities win; default is row order. */
  priorities?: Float32Array;
  maximumCandidates?: number;
};

/**
 * Mirrors the label shader's pixel placement on the CPU, then runs bounded greedy placement.
 * Returns one visibility bit per row (row index = bit position), also clearing rows whose anchors
 * are invalid. With collision disabled use `getAnchorValidityWords` instead.
 */
export function computeLabelVisibilityWords(options: LabelVisibilityOptions): Uint32Array {
  const {rowCount, anchors, rowBounds, viewport} = options;
  const projector = new WebMercatorViewport({
    longitude: viewport.longitude,
    latitude: viewport.latitude,
    zoom: viewport.zoom,
    width: viewport.width,
    height: viewport.height,
    pitch: viewport.pitch ?? 0,
    bearing: viewport.bearing ?? 0
  });
  const scale = options.size / options.fontSize;
  const anchorShift = options.anchor === 'start' ? 0 : options.anchor === 'end' ? -1 : -0.5;
  const baselineShift =
    options.alignment === 'top' ? 0 : options.alignment === 'bottom' ? -1 : -0.5;
  const directionX = options.anchor === 'start' ? 1 : options.anchor === 'end' ? -1 : 0;
  const directionY = options.alignment === 'top' ? -1 : options.alignment === 'bottom' ? 1 : 0;
  const radiusOffset = options.radius * options.radiusPixelScale + options.radiusPadding;
  const offsetX = directionX * radiusOffset;
  const offsetY = directionY * (radiusOffset + options.verticalPadding);
  const margin = 256;
  const candidates: LabelCollisionCandidate[] = [];
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    if (anchors.validity && !anchors.validity[rowIndex]) continue;
    const width = rowBounds[rowIndex * 4] * scale;
    const height = rowBounds[rowIndex * 4 + 1] * scale;
    if (!(width > 0 && height > 0)) continue;
    const [screenX, screenY] = projector.project([
      anchors.positions[rowIndex * 2],
      anchors.positions[rowIndex * 2 + 1]
    ]);
    if (
      !Number.isFinite(screenX) ||
      !Number.isFinite(screenY) ||
      screenX < -margin ||
      screenY < -margin ||
      screenX > viewport.width + margin ||
      screenY > viewport.height + margin
    )
      continue;
    candidates.push({
      rowIndex,
      x: screenX + offsetX + width * anchorShift,
      y: screenY + offsetY + height * baselineShift,
      width,
      height,
      priority: options.priorities ? options.priorities[rowIndex] : 0
    });
  }
  return packAcceptedLabelRows(
    rowCount,
    placeLabelsGreedy(candidates, {
      padding: options.padding,
      maximumCandidates: options.maximumCandidates
    })
  );
}

/** Visibility words that only hide rows without a usable anchor. */
export function getAnchorValidityWords(rowCount: number, validity: Uint32Array): Uint32Array {
  const accepted: number[] = [];
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++)
    if (validity[rowIndex]) accepted.push(rowIndex);
  return packAcceptedLabelRows(rowCount, accepted);
}
