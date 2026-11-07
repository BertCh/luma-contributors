// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PolygonLayout} from './b3-common';
import {projectRows, projectAlbers} from '../regression/b6-polygons';

export const COVERAGE_AUDIT_STATE_FIPS = [17, 18, 55] as const;
export const COVERAGE_AUDIT_STATE_NAMES: Record<number, string> = {
  17: 'Illinois',
  18: 'Indiana',
  55: 'Wisconsin'
};
export const COVERAGE_AUDIT_PERTURBATION_METERS = 50;

export type CoverageAuditData = {
  layout: PolygonLayout;
  projectedPositions: Float32Array;
  perturbedPositions: Float32Array;
  featureStateFips: Uint32Array;
  sourceFeatureRows: Uint32Array;
  countyFips: Uint32Array;
  countsByState: Map<number, number>;
  sourceFeatureCount: number;
  perturbation: {
    targetFeatureRow: number;
    targetCountyFips: number;
    vertexIndex: number;
    partnerCountyFips: number;
    offsetX: number;
    offsetY: number;
  };
};

/** Copy every county polygon, ring and stored coordinate for the three adjacent source states. */
export function buildCoverageAuditData(
  sourceLayout: PolygonLayout,
  stateFips: Uint8Array,
  countyFips: Uint32Array,
  sourceFeatureCount: number
): CoverageAuditData {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  const featureStateFips: number[] = [];
  const sourceFeatureRows: number[] = [];
  const selectedCountyFips: number[] = [];
  const countsByState = new Map<number, number>();

  for (let feature = 0; feature < sourceLayout.featureCount; feature++) {
    const state = stateFips[feature];
    if (state !== 17 && state !== 18 && state !== 55) continue;
    featureStateFips.push(state);
    sourceFeatureRows.push(feature);
    selectedCountyFips.push(countyFips[feature]);
    countsByState.set(state, (countsByState.get(state) ?? 0) + 1);
    for (
      let part = sourceLayout.featureOffsets[feature];
      part < sourceLayout.featureOffsets[feature + 1];
      part++
    ) {
      for (
        let ring = sourceLayout.polygonOffsets[part];
        ring < sourceLayout.polygonOffsets[part + 1];
        ring++
      ) {
        const start = sourceLayout.ringOffsets[ring];
        const end = sourceLayout.ringOffsets[ring + 1];
        for (let vertex = start; vertex < end; vertex++) {
          positions.push(sourceLayout.lngLat[vertex * 2], sourceLayout.lngLat[vertex * 2 + 1]);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }

  const lngLat = Float32Array.from(positions);
  const subsetRingOffsets = Uint32Array.from(ringOffsets);
  const subsetPolygonOffsets = Uint32Array.from(polygonOffsets);
  const subsetFeatureOffsets = Uint32Array.from(featureOffsets);
  const subsetFeatureRingOffsets = new Uint32Array(featureStateFips.length + 1);
  const ringFeature = new Uint32Array(subsetRingOffsets.length - 1);
  const featureBounds = new Float32Array(featureStateFips.length * 4);
  for (let feature = 0; feature <= featureStateFips.length; feature++) {
    subsetFeatureRingOffsets[feature] = subsetPolygonOffsets[subsetFeatureOffsets[feature]];
  }
  for (let feature = 0; feature < featureStateFips.length; feature++) {
    let west = Number.POSITIVE_INFINITY;
    let south = Number.POSITIVE_INFINITY;
    let east = Number.NEGATIVE_INFINITY;
    let north = Number.NEGATIVE_INFINITY;
    for (
      let ring = subsetFeatureRingOffsets[feature];
      ring < subsetFeatureRingOffsets[feature + 1];
      ring++
    ) {
      ringFeature[ring] = feature;
      for (let vertex = subsetRingOffsets[ring]; vertex < subsetRingOffsets[ring + 1]; vertex++) {
        const longitude = lngLat[vertex * 2];
        const latitude = lngLat[vertex * 2 + 1];
        west = Math.min(west, longitude);
        east = Math.max(east, longitude);
        south = Math.min(south, latitude);
        north = Math.max(north, latitude);
      }
    }
    featureBounds.set([west, south, east, north], feature * 4);
  }
  const layout: PolygonLayout = {
    featureCount: featureStateFips.length,
    partCount: subsetPolygonOffsets.length - 1,
    ringCount: subsetRingOffsets.length - 1,
    vertexCount: lngLat.length / 2,
    lngLat,
    ringOffsets: subsetRingOffsets,
    polygonOffsets: subsetPolygonOffsets,
    featureOffsets: subsetFeatureOffsets,
    featureRingOffsets: subsetFeatureRingOffsets,
    ringFeature,
    featureBounds
  };
  const projectedPositions = projectRows(lngLat, projectAlbers);
  const perturbation = findPerturbation(
    layout,
    projectedPositions,
    sourceFeatureRows,
    selectedCountyFips
  );
  const perturbedPositions = Float32Array.from(projectedPositions);
  perturbedPositions[perturbation.vertexIndex * 2] += perturbation.offsetX;
  perturbedPositions[perturbation.vertexIndex * 2 + 1] += perturbation.offsetY;

  return {
    layout,
    projectedPositions,
    perturbedPositions,
    featureStateFips: Uint32Array.from(featureStateFips),
    sourceFeatureRows: Uint32Array.from(sourceFeatureRows),
    countyFips: Uint32Array.from(selectedCountyFips),
    countsByState,
    sourceFeatureCount,
    perturbation
  };
}

function findPerturbation(
  layout: PolygonLayout,
  projectedPositions: Float32Array,
  sourceFeatureRows: readonly number[],
  countyFips: readonly number[]
): CoverageAuditData['perturbation'] {
  const featureBySourceRow = new Map(sourceFeatureRows.map((row, index) => [row, index]));
  const coordinateOwners = new Map<string, Map<number, number>>();
  const vertexRings = new Uint32Array(layout.vertexCount);
  for (let ring = 0; ring < layout.ringCount; ring++) {
    for (let vertex = layout.ringOffsets[ring]; vertex < layout.ringOffsets[ring + 1]; vertex++) {
      vertexRings[vertex] = ring;
      const sourceFeature = sourceFeatureRows[layout.ringFeature[ring]];
      const key = `${layout.lngLat[vertex * 2]},${layout.lngLat[vertex * 2 + 1]}`;
      const owners = coordinateOwners.get(key) ?? new Map<number, number>();
      if (!owners.has(sourceFeature)) owners.set(sourceFeature, vertex);
      coordinateOwners.set(key, owners);
    }
  }

  for (let vertex = 0; vertex < layout.vertexCount; vertex++) {
    const [ringStart, ringEnd] = [
      layout.ringOffsets[vertexRings[vertex]],
      layout.ringOffsets[vertexRings[vertex] + 1]
    ];
    const hasExplicitClosure =
      ringEnd - ringStart > 1 &&
      layout.lngLat[ringStart * 2] === layout.lngLat[(ringEnd - 1) * 2] &&
      layout.lngLat[ringStart * 2 + 1] === layout.lngLat[(ringEnd - 1) * 2 + 1];
    if (hasExplicitClosure && (vertex === ringStart || vertex === ringEnd - 1)) continue;
    const key = `${layout.lngLat[vertex * 2]},${layout.lngLat[vertex * 2 + 1]}`;
    const owners = coordinateOwners.get(key);
    if (!owners || owners.size < 2) continue;
    const featureIndex = layout.ringFeature[vertexRings[vertex]];
    const sourceFeatureRow = sourceFeatureRows[featureIndex];
    const partner = Array.from(owners).find(([owner]) => owner !== sourceFeatureRow);
    if (!partner) continue;
    const nextVertex = vertex + 1 < ringEnd ? vertex + 1 : ringStart;
    const previousVertex = vertex - 1 >= ringStart ? vertex - 1 : ringEnd - 1;
    const adjacentVertex = nextVertex !== vertex ? nextVertex : previousVertex;
    let tangentX = projectedPositions[adjacentVertex * 2] - projectedPositions[vertex * 2];
    let tangentY = projectedPositions[adjacentVertex * 2 + 1] - projectedPositions[vertex * 2 + 1];
    let tangentLength = Math.hypot(tangentX, tangentY);
    if (!(tangentLength > 0)) {
      tangentX = projectedPositions[vertex * 2] - projectedPositions[previousVertex * 2];
      tangentY = projectedPositions[vertex * 2 + 1] - projectedPositions[previousVertex * 2 + 1];
      tangentLength = Math.hypot(tangentX, tangentY);
    }
    if (!(tangentLength > 0)) continue;
    const targetFeature = featureBySourceRow.get(sourceFeatureRow);
    const partnerFeature = featureBySourceRow.get(partner[0]);
    if (targetFeature === undefined || partnerFeature === undefined) continue;
    return {
      targetFeatureRow: sourceFeatureRow,
      targetCountyFips: countyFips[targetFeature],
      vertexIndex: vertex,
      partnerCountyFips: countyFips[partnerFeature],
      offsetX: (-tangentY / tangentLength) * COVERAGE_AUDIT_PERTURBATION_METERS,
      offsetY: (tangentX / tangentLength) * COVERAGE_AUDIT_PERTURBATION_METERS
    };
  }
  throw new Error(
    'The selected three-state county subset must contain an exactly shared county vertex.'
  );
}
