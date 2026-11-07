// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Kepler HIGHLIGH_COLOR_3D: the translucent white that native extruded H3 hover blends in. */
export const H3_EXTRUDED_HIGHLIGHT_COLOR: number[] = [255, 255, 255, 60];

export type H3HighlightProps = {
  autoHighlight: boolean;
  highlightColor: number[];
};

/**
 * Native H3 sets `autoHighlight: visConfig.enable3d` with HIGHLIGH_COLOR_3D, so extruded cells blend
 * translucent white over the hovered column or prism and flat cells are not auto-highlighted: they
 * get the separate hovered-outline GeoJSON layer (see getH3HoveredOutline). Brushing owns the pointer.
 */
export function getH3HighlightProps(extruded: boolean, brushEnabled: boolean): H3HighlightProps {
  return {
    autoHighlight: extruded && !brushEnabled,
    highlightColor: [...H3_EXTRUDED_HIGHLIGHT_COLOR]
  };
}

/** Native `lineWidthScale` of the H3 hovered-outline GeoJsonLayer: DEFAULT_LINE_SCALE_VALUE (8) * zoomFactor. */
export function getH3OutlineWidthScale(zoomFactor: number): number {
  return 8 * zoomFactor;
}

type H3BoundarySource = {
  batches: readonly {sourceRowOffset: number; rowCount: number}[];
};

/**
 * Closed boundary of the hovered source row for the native hovered-outline layer
 * (`idToPolygonGeo(hoveredObject)`); undefined for invalid rows or when native draws none
 * (native skips the outline when a size field is set).
 */
export function getH3HoveredOutline(
  source: H3BoundarySource,
  sourceRow: number,
  hasSizeField: boolean,
  getBoundary: (batchIndex: number, rowIndex: number) => [number, number][]
): [number, number][] | undefined {
  if (hasSizeField || !Number.isInteger(sourceRow) || sourceRow < 0) return undefined;
  const batchIndex = source.batches.findIndex(
    batch =>
      sourceRow >= batch.sourceRowOffset && sourceRow < batch.sourceRowOffset + batch.rowCount
  );
  if (batchIndex < 0) return undefined;
  const boundary = getBoundary(batchIndex, sourceRow - source.batches[batchIndex].sourceRowOffset);
  return boundary.length ? boundary : undefined;
}
