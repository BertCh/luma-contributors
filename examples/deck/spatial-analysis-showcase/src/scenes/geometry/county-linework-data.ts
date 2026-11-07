// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PolygonLayout} from './b3-common';
import {projectAlbers} from '../regression/b6-polygons';

export const COUNTY_LINEWORK_FIPS = {left: 41015, right: 41033} as const;
export const COUNTY_LINEWORK_NAMES = {
  left: 'Curry County, Oregon',
  right: 'Josephine County, Oregon'
} as const;

export type CountyLineworkEncoding = 'archived' | 'reverse-right-ring' | 'omit-outer-edge';

export type CountyLineworkInput = {
  leftPositions: Float32Array;
  leftOffsets: Uint32Array;
  rightPositions: Float32Array;
  rightOffsets: Uint32Array;
  /** Original projected coordinates after global float32 quantization. */
  globalPositions: Float32Array;
  /** Rounded Albers-metre midpoint subtracted from `globalPositions` to make local topology coordinates. */
  coordinateOrigin: [number, number];
  positions: Float32Array;
  lineOffsets: Uint32Array;
  lineSourceIds: Uint32Array;
  sourcePaths: {path: [number, number][]; countyFips: number}[];
  inputVertexCount: number;
  leftRingCount: number;
  rightRingCount: number;
  omittedEdge: {countyFips: number; sourceEdgeId: number} | null;
  omittedSegment: [[number, number], [number, number]] | null;
};

/** Keep the two complete source rings and change only the selected representation scenario. */
export function buildCountyLineworkInput(
  layout: PolygonLayout,
  countyFips: Uint32Array,
  encoding: CountyLineworkEncoding
): CountyLineworkInput {
  const featureRows = new Map<number, number>();
  for (let feature = 0; feature < countyFips.length; feature++) {
    if (
      countyFips[feature] === COUNTY_LINEWORK_FIPS.left ||
      countyFips[feature] === COUNTY_LINEWORK_FIPS.right
    ) {
      featureRows.set(countyFips[feature], feature);
    }
  }

  const leftFeature = featureRows.get(COUNTY_LINEWORK_FIPS.left);
  const rightFeature = featureRows.get(COUNTY_LINEWORK_FIPS.right);
  if (leftFeature === undefined || rightFeature === undefined) {
    throw new Error('The archived county dataset must contain Curry County and Josephine County.');
  }

  const leftRings = getFeatureRings(layout, leftFeature);
  const rightRings = getFeatureRings(layout, rightFeature);
  const leftPaths = leftRings.map(ring => makeRingPath(layout, ring, 'archived'));
  const rightPaths = rightRings.map(ring =>
    makeRingPath(layout, ring, encoding === 'reverse-right-ring' ? encoding : 'archived')
  );
  let omittedEdge: CountyLineworkInput['omittedEdge'] = null;
  let omittedSegment: CountyLineworkInput['omittedSegment'] = null;
  if (encoding === 'omit-outer-edge') {
    const firstLeft = leftPaths[0];
    const closedVertexCount = firstLeft.length;
    const uniquePath = firstLeft.slice(0, -1);
    omittedSegment = [uniquePath[0], uniquePath[1]];
    leftPaths[0] = [...uniquePath.slice(1), uniquePath[0]];
    omittedEdge = {countyFips: COUNTY_LINEWORK_FIPS.left, sourceEdgeId: 0};
    if (leftPaths[0].length !== closedVertexCount - 1) {
      throw new Error('The omitted-edge scenario must remove exactly one closing vertex.');
    }
  }

  const positions: number[] = [];
  const lineOffsets = [0];
  const lineSourceIds: number[] = [];
  const sourcePaths: CountyLineworkInput['sourcePaths'] = [];
  const appendRings = (paths: [number, number][][], sourceId: number) => {
    for (const path of paths) {
      sourcePaths.push({path, countyFips: sourceId});
      lineSourceIds.push(sourceId);
      for (const [longitude, latitude] of path)
        positions.push(...projectAlbers(longitude, latitude));
      lineOffsets.push(positions.length / 2);
    }
  };
  appendRings(leftPaths, COUNTY_LINEWORK_FIPS.left);
  const leftEnd = positions.length / 2;
  const leftLineEnd = lineOffsets.length;
  appendRings(rightPaths, COUNTY_LINEWORK_FIPS.right);
  const rightStart = leftEnd;
  const rightLineStart = leftLineEnd - 1;
  const globalPositions = Float32Array.from(positions);
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let vertex = 0; vertex < globalPositions.length / 2; vertex++) {
    const positionX = globalPositions[vertex * 2];
    const positionY = globalPositions[vertex * 2 + 1];
    minimumX = Math.min(minimumX, positionX);
    minimumY = Math.min(minimumY, positionY);
    maximumX = Math.max(maximumX, positionX);
    maximumY = Math.max(maximumY, positionY);
  }
  const coordinateOrigin: [number, number] = [
    Math.fround((minimumX + maximumX) / 2),
    Math.fround((minimumY + maximumY) / 2)
  ];
  const projectedPositions = new Float32Array(globalPositions.length);
  for (let coordinate = 0; coordinate < globalPositions.length; coordinate++) {
    projectedPositions[coordinate] = Math.fround(
      globalPositions[coordinate] - coordinateOrigin[coordinate % 2]
    );
  }
  const leftOffsets = lineOffsets.slice(0, leftLineEnd);
  const rightOffsets = lineOffsets.slice(rightLineStart).map(offset => offset - rightStart);

  return {
    leftPositions: projectedPositions.slice(0, rightStart * 2),
    leftOffsets: Uint32Array.from(leftOffsets),
    rightPositions: projectedPositions.slice(rightStart * 2),
    rightOffsets: Uint32Array.from(rightOffsets),
    globalPositions,
    coordinateOrigin,
    positions: projectedPositions,
    lineOffsets: Uint32Array.from(lineOffsets),
    lineSourceIds: Uint32Array.from(lineSourceIds),
    sourcePaths,
    inputVertexCount: projectedPositions.length / 2,
    leftRingCount: leftPaths.length,
    rightRingCount: rightPaths.length,
    omittedEdge,
    omittedSegment
  };
}

function getFeatureRings(layout: PolygonLayout, feature: number): number[] {
  const rings: number[] = [];
  for (
    let part = layout.featureOffsets[feature];
    part < layout.featureOffsets[feature + 1];
    part++
  ) {
    for (let ring = layout.polygonOffsets[part]; ring < layout.polygonOffsets[part + 1]; ring++) {
      rings.push(ring);
    }
  }
  return rings;
}

function makeRingPath(
  layout: PolygonLayout,
  ring: number,
  encoding: CountyLineworkEncoding
): [number, number][] {
  const start = layout.ringOffsets[ring];
  const end = layout.ringOffsets[ring + 1];
  const coordinates: [number, number][] = [];
  for (let vertex = start; vertex < end; vertex++) {
    coordinates.push([layout.lngLat[vertex * 2], layout.lngLat[vertex * 2 + 1]]);
  }
  if (coordinates.length > 1) {
    const first = coordinates[0];
    const last = coordinates.at(-1)!;
    if (first[0] === last[0] && first[1] === last[1]) coordinates.pop();
  }
  if (encoding === 'reverse-right-ring') coordinates.reverse();
  coordinates.push(coordinates[0]);
  return coordinates;
}
