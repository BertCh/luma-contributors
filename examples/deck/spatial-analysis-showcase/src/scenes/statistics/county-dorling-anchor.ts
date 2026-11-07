// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Returns the arithmetic mean of every authored polygon vertex. Closing vertices and hole rings
 * participate, so this is a reproducible display anchor, not a centroid or an interior guarantee.
 */
export function getCountyDorlingAuthoredVertexMeanAnchor(feature: {
  geometry?: {coordinates?: unknown} | null;
}): [number, number] {
  const coordinates: [number, number][] = [];
  const visit = (value: unknown): void => {
    if (!Array.isArray(value)) return;
    if (value.length >= 2 && typeof value[0] === 'number' && typeof value[1] === 'number') {
      coordinates.push([value[0], value[1]]);
      return;
    }
    for (const child of value) visit(child);
  };
  visit(feature.geometry?.coordinates);
  if (coordinates.length === 0) return [-74.55, 40.05];
  return [
    coordinates.reduce((total, coordinate) => total + coordinate[0], 0) / coordinates.length,
    coordinates.reduce((total, coordinate) => total + coordinate[1], 0) / coordinates.length
  ];
}
