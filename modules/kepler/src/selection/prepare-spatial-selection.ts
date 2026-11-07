// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One point in the same Float32 coordinate frame as the resident point dataset. */
export type KeplerSelectionPosition = readonly [number, number];
/** A shell followed by zero or more holes. Rings close implicitly. */
export type KeplerSelectionPolygon = readonly (readonly KeplerSelectionPosition[])[];

/** Small application-owned drawn geometry; source dataset positions stay on the GPU. */
export type KeplerSpatialSelection =
  | {kind: 'rectangle'; bounds: readonly [number, number, number, number]}
  | {kind: 'polygon'; polygons: readonly KeplerSelectionPolygon[]};

/**
 * The interpretation of linear edges; longitude wrapping or geodesic edges are not inferred.
 * 'native-linear-degrees' reproduces native Kepler, whose isInPolygon is turf booleanWithin on
 * raw [longitude, latitude] with no antimeridian split, wrapping or range validation: coordinates
 * such as 190 or edges spanning more than 180 degrees are ordinary planar vertices.
 */
export type KeplerSelectionCoordinateSpace =
  | 'planar'
  | 'longitude-latitude'
  | 'native-linear-degrees';

/** One union feature in the graph's polygon/ring/vertex offset layout. */
export type PreparedKeplerSpatialSelection = {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Ring and polygon structure, used to distinguish parameter changes from topology changes. */
  topologyKey: string;
};

/**
 * Prepares only drawn geometry, never dataset rows. Coordinates are explicitly rounded to Float32
 * to match resident positions. Simple shells and holes are caller-owned validity requirements.
 * Longitude/latitude means linear degree-plane containment. Split crossing geometry before use;
 * this adapter rejects antimeridian and pole geometry rather than guessing a longitude frame.
 */
export function prepareKeplerSpatialSelection(
  selection: KeplerSpatialSelection,
  coordinateSpace: KeplerSelectionCoordinateSpace = 'planar'
): PreparedKeplerSpatialSelection {
  if (
    coordinateSpace !== 'planar' &&
    coordinateSpace !== 'longitude-latitude' &&
    coordinateSpace !== 'native-linear-degrees'
  ) {
    throw new Error('Unsupported selection coordinate space');
  }
  let polygons: readonly KeplerSelectionPolygon[];
  if (selection.kind === 'rectangle') {
    const [minimumX, minimumY, maximumX, maximumY] = selection.bounds;
    if (
      !(
        Math.fround(minimumX) < Math.fround(maximumX) &&
        Math.fround(minimumY) < Math.fround(maximumY)
      )
    ) {
      throw new Error('Selection bounds must have positive width and height');
    }
    polygons = [
      [
        [
          [minimumX, minimumY],
          [maximumX, minimumY],
          [maximumX, maximumY],
          [minimumX, maximumY]
        ]
      ]
    ];
  } else if (selection.kind === 'polygon') {
    polygons = selection.polygons;
  } else {
    throw new Error('Unsupported selection geometry');
  }
  if (polygons.length === 0) throw new Error('Selection requires at least one polygon');

  const positions: number[] = [];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const polygon of polygons) {
    if (polygon.length === 0) throw new Error('Selection polygon requires a shell');
    for (const ring of polygon) {
      let vertexCount = ring.length;
      if (
        vertexCount > 1 &&
        ring[0][0] === ring[vertexCount - 1][0] &&
        ring[0][1] === ring[vertexCount - 1][1]
      ) {
        vertexCount--;
      }
      if (vertexCount < 3) throw new Error('Selection rings require at least three vertices');
      const rounded: KeplerSelectionPosition[] = [];
      for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
        const [sourceX, sourceY] = ring[vertexIndex];
        const coordinateX = Math.fround(sourceX);
        const coordinateY = Math.fround(sourceY);
        if (!Number.isFinite(coordinateX) || !Number.isFinite(coordinateY)) {
          throw new Error('Selection coordinates must be finite Float32 values');
        }
        if (
          coordinateSpace === 'longitude-latitude' &&
          (coordinateX < -180 || coordinateX > 180 || coordinateY <= -90 || coordinateY >= 90)
        ) {
          throw new Error('Selection longitude or latitude requires geographic preparation');
        }
        rounded.push([coordinateX, coordinateY]);
        positions.push(coordinateX, coordinateY);
      }
      if (coordinateSpace === 'longitude-latitude') {
        for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
          const nextIndex = (vertexIndex + 1) % vertexCount;
          if (Math.abs(rounded[vertexIndex][0] - rounded[nextIndex][0]) >= 180) {
            throw new Error('Selection antimeridian edges require geographic preparation');
          }
        }
      }
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
  }

  return {
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from([0, polygons.length]),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    topologyKey: `${polygonOffsets.join(',')}:${ringOffsets.join(',')}`
  };
}
