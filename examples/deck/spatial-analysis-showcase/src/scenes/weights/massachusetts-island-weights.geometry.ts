// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {ALBERS_CONUS_FRAME, createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {
  getFeaturePolygonOffsets,
  readPolygonGeometry,
  subsetPolygonGeometry
} from '../regression/b6-polygons';

const MASSACHUSETTS_STATE_FIPS = 25;
const MAXIMUM_FUZZY_VERTICES = 65_536;

export type MassachusettsIslandGeometry = {
  featureCount: number;
  sourceRows: Uint32Array;
  fips: Uint32Array;
  names: string[];
  /** Census area-weighted centroids, projected into the showcase spherical CONUS Albers frame. */
  representatives: Float32Array;
  /** The same source representatives in the dataset's longitude/latitude frame for map links. */
  representativeLngLat: Float32Array;
  /** Unclosed metric polygon positions for the GPU geometry port. */
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  featureValidity: Uint32Array;
  vertexCount: number;
  geoJson: {
    type: 'FeatureCollection';
    features: NonNullable<LoadedDataset['geojson']>['features'];
  };
};

function cross(first: readonly [number, number], second: readonly [number, number]): number {
  return first[0] * second[1] - first[1] * second[0];
}

function orientation(
  first: readonly [number, number],
  second: readonly [number, number],
  third: readonly [number, number]
): number {
  return cross(
    [second[0] - first[0], second[1] - first[1]],
    [third[0] - first[0], third[1] - first[1]]
  );
}

function segmentsIntersect(
  first: readonly [number, number],
  second: readonly [number, number],
  third: readonly [number, number],
  fourth: readonly [number, number]
): boolean {
  const firstTurn = orientation(first, second, third);
  const secondTurn = orientation(first, second, fourth);
  const thirdTurn = orientation(third, fourth, first);
  const fourthTurn = orientation(third, fourth, second);
  const onSegment = (
    start: readonly [number, number],
    end: readonly [number, number],
    point: readonly [number, number]
  ) =>
    point[0] >= Math.min(start[0], end[0]) &&
    point[0] <= Math.max(start[0], end[0]) &&
    point[1] >= Math.min(start[1], end[1]) &&
    point[1] <= Math.max(start[1], end[1]);
  if (firstTurn === 0 && onSegment(first, second, third)) return true;
  if (secondTurn === 0 && onSegment(first, second, fourth)) return true;
  if (thirdTurn === 0 && onSegment(third, fourth, first)) return true;
  if (fourthTurn === 0 && onSegment(third, fourth, second)) return true;
  return firstTurn < 0 !== secondTurn < 0 && thirdTurn < 0 !== fourthTurn < 0;
}

function hasSimpleFiniteRings(
  positions: Float32Array,
  ringOffsets: Uint32Array,
  firstRing: number,
  lastRing: number
): boolean {
  for (let ring = firstRing; ring < lastRing; ring++) {
    const start = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    const count = end - start;
    if (count < 3) return false;
    let twiceArea = 0;
    for (let vertex = 0; vertex < count; vertex++) {
      const firstIndex = (start + vertex) * 2;
      const secondIndex = (start + ((vertex + 1) % count)) * 2;
      const first: [number, number] = [positions[firstIndex], positions[firstIndex + 1]];
      const second: [number, number] = [positions[secondIndex], positions[secondIndex + 1]];
      if (!Number.isFinite(first[0]) || !Number.isFinite(first[1])) return false;
      if (first[0] === second[0] && first[1] === second[1]) return false;
      twiceArea += first[0] * second[1] - second[0] * first[1];
      for (let other = vertex + 1; other < count; other++) {
        if (other === vertex + 1 || (vertex === 0 && other === count - 1)) continue;
        const thirdIndex = (start + other) * 2;
        const fourthIndex = (start + ((other + 1) % count)) * 2;
        const third: [number, number] = [positions[thirdIndex], positions[thirdIndex + 1]];
        const fourth: [number, number] = [positions[fourthIndex], positions[fourthIndex + 1]];
        if (segmentsIntersect(first, second, third, fourth)) return false;
      }
    }
    if (!Number.isFinite(twiceArea) || Math.abs(twiceArea) < 1e-4) return false;
  }
  return true;
}

/** Selects all original Massachusetts county rows, preserving the source feature identity and rings. */
export function createMassachusettsIslandGeometry(
  dataset: LoadedDataset
): MassachusettsIslandGeometry {
  const stateFips = dataset.column<Uint8Array>('stateFips');
  const fipsColumn = dataset.column<Uint32Array>('fips');
  const sourceRows = Uint32Array.from(
    Array.from({length: dataset.count}, (_, row) => row).filter(
      row => stateFips[row] === MASSACHUSETTS_STATE_FIPS
    )
  );
  const fips = Uint32Array.from(sourceRows, row => fipsColumn[row]);
  const sourceGeometry = readPolygonGeometry(dataset);
  const selectedGeometry = subsetPolygonGeometry(sourceGeometry, sourceRows);
  const projection = createAnalysisFrameProjection(ALBERS_CONUS_FRAME);
  const projectedClosedPositions = new Float32Array(selectedGeometry.vertices.length);
  for (let vertex = 0; vertex < selectedGeometry.vertices.length; vertex += 2) {
    const [x, y] = projection.project(
      selectedGeometry.vertices[vertex],
      selectedGeometry.vertices[vertex + 1]
    );
    projectedClosedPositions[vertex] = x;
    projectedClosedPositions[vertex + 1] = y;
  }

  const openPositions: number[] = [];
  const ringOffsets = new Uint32Array(selectedGeometry.ringOffsets.length);
  const featureValidity = new Uint32Array(selectedGeometry.featureCount);
  for (let ring = 0; ring < selectedGeometry.ringOffsets.length - 1; ring++) {
    const start = selectedGeometry.ringOffsets[ring];
    let end = selectedGeometry.ringOffsets[ring + 1];
    const firstIndex = start * 2;
    const lastIndex = (end - 1) * 2;
    if (
      end > start + 1 &&
      projectedClosedPositions[firstIndex] === projectedClosedPositions[lastIndex] &&
      projectedClosedPositions[firstIndex + 1] === projectedClosedPositions[lastIndex + 1]
    ) {
      end--;
    }
    for (let vertex = start; vertex < end; vertex++) {
      openPositions.push(
        projectedClosedPositions[vertex * 2],
        projectedClosedPositions[vertex * 2 + 1]
      );
    }
    ringOffsets[ring + 1] = openPositions.length / 2;
  }
  const positions = Float32Array.from(openPositions);
  if (positions.length / 2 > MAXIMUM_FUZZY_VERTICES) {
    // Fuzzy contiguity has a documented 65,536-vertex bound; never simplify source boundaries silently.
    throw new Error(
      `Massachusetts county geometry exceeds ${MAXIMUM_FUZZY_VERTICES} fuzzy-contiguity vertices`
    );
  }
  const featureOffsets = getFeaturePolygonOffsets(selectedGeometry);
  for (let feature = 0; feature < selectedGeometry.featureCount; feature++) {
    const firstRing = selectedGeometry.partRingOffsets[featureOffsets[feature]];
    const lastRing = selectedGeometry.partRingOffsets[featureOffsets[feature + 1]];
    featureValidity[feature] = hasSimpleFiniteRings(positions, ringOffsets, firstRing, lastRing)
      ? 0
      : 1;
  }
  const names = Array.from(sourceRows, sourceRow => {
    const sourceFeature = dataset.geojson?.features[sourceRow];
    const sourceName = sourceFeature?.properties?.name;
    return typeof sourceName === 'string' ? sourceName : `FIPS ${fipsColumn[sourceRow]}`;
  });
  const sourceGeoJson = dataset.geojson;
  if (!sourceGeoJson) throw new Error('US county geometry is unavailable');
  const geoJsonFeatures = Array.from(sourceRows, sourceRow => sourceGeoJson.features[sourceRow]);
  const sourceRepresentatives = dataset.column<Float32Array>('centroid');
  const representatives = new Float32Array(sourceRows.length * 2);
  const representativeLngLat = new Float32Array(sourceRows.length * 2);
  for (let feature = 0; feature < sourceRows.length; feature++) {
    const sourceRow = sourceRows[feature];
    const [x, y] = projection.project(
      sourceRepresentatives[sourceRow * 2],
      sourceRepresentatives[sourceRow * 2 + 1]
    );
    representatives[feature * 2] = x;
    representatives[feature * 2 + 1] = y;
    representativeLngLat[feature * 2] = sourceRepresentatives[sourceRow * 2];
    representativeLngLat[feature * 2 + 1] = sourceRepresentatives[sourceRow * 2 + 1];
  }
  return {
    featureCount: selectedGeometry.featureCount,
    sourceRows,
    fips,
    names,
    representatives,
    representativeLngLat,
    positions,
    featureOffsets,
    polygonOffsets: selectedGeometry.partRingOffsets,
    ringOffsets,
    featureValidity,
    vertexCount: positions.length / 2,
    geoJson: {type: 'FeatureCollection', features: geoJsonFeatures}
  };
}
