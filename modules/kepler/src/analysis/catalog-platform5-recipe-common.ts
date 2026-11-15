// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterSpec} from './catalog-binding-types';
import {
  getPointPositions,
  getPolygonParts,
  type CatalogCoordinateSystem,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {isCatalogPointInPolygon, packCatalogPolygons} from './catalog-gpu-pack';
import {collectAllPositions, readNumericProperty} from './catalog-platform3-b-common';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

/** Points in a local frame; values are NaN when the property is missing or non numeric. */
export type RecipePoints = {
  frame: LocalFrame;
  /** Float32 frame coordinates, interleaved. */
  positions: Float32Array;
  /** Float32 values, NaN for missing. */
  values: Float32Array;
  sourceIndex: Int32Array;
  count: number;
  droppedCount: number;
};

export type RecipePolygons = {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Source feature index of every packed polygon feature. */
  sourceIndices: number[];
  /** Per packed feature, per polygon part, rings of float32-rounded frame coordinates. */
  parts: CatalogPosition[][][][];
  featureCount: number;
};

/** Frame covering every vertex of the given collections. */
export function makeRecipeFrame(
  coordinateSystem: CatalogCoordinateSystem,
  collections: readonly CatalogFeatureCollection[]
): LocalFrame {
  return makeLocalFrame(
    coordinateSystem,
    collections.flatMap(collection => collectAllPositions(collection)),
    {shift: 'minimum'}
  );
}

/** Reads point positions. A set value property keeps points without a number as NaN. */
export function readRecipePoints(
  collection: CatalogFeatureCollection,
  valueProperty: string,
  frame: LocalFrame
): RecipePoints {
  const positions: number[] = [];
  const values: number[] = [];
  const sourceIndex: number[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, index) => {
    const featurePositions = getPointPositions(feature.geometry);
    if (!featurePositions.length) {
      droppedCount++;
      return;
    }
    const numeric = valueProperty ? readNumericProperty(feature, valueProperty) : 1;
    for (const position of featurePositions) {
      const [x, y] = frame.forward(position);
      positions.push(x, y);
      values.push(numeric ?? Number.NaN);
      sourceIndex.push(index);
    }
  });
  return {
    frame,
    positions: Float32Array.from(positions),
    values: Float32Array.from(values),
    sourceIndex: Int32Array.from(sourceIndex),
    count: values.length,
    droppedCount
  };
}

/** Packs polygon features in the frame; features without polygon parts are skipped. */
export function readRecipePolygons(
  collection: CatalogFeatureCollection,
  frame: LocalFrame
): RecipePolygons {
  const packed = packCatalogPolygons(collection.features, {
    project: position => frame.forward(position),
    closing: 'drop'
  });
  const parts = packed.sourceIndices.map(sourceIndex =>
    getPolygonParts(collection.features[sourceIndex].geometry).map(polygon =>
      polygon.map(ring =>
        ring.map(position => {
          const [x, y] = frame.forward(position);
          return [Math.fround(x), Math.fround(y)] as CatalogPosition;
        })
      )
    )
  );
  return {
    positions: packed.positions,
    featureOffsets: packed.featureOffsets,
    polygonOffsets: packed.polygonOffsets,
    ringOffsets: packed.ringOffsets,
    sourceIndices: packed.sourceIndices,
    parts,
    featureCount: packed.sourceIndices.length
  };
}

/** Smallest packed feature row whose polygon parts contain each point, or -1. */
export function assignRecipePoints(points: RecipePoints, polygons: RecipePolygons): Int32Array {
  const assignment = new Int32Array(points.count).fill(-1);
  for (let point = 0; point < points.count; point++) {
    const position: CatalogPosition = [
      points.positions[2 * point],
      points.positions[2 * point + 1]
    ];
    assignment[point] = polygons.parts.findIndex(feature =>
      feature.some(rings => isCatalogPointInPolygon(position, rings))
    );
  }
  return assignment;
}

/** Visits each containing feature once, including every match in overlapping features. */
export function visitRecipePointMatches(
  points: RecipePoints,
  polygons: RecipePolygons,
  visit?: (point: number, feature: number) => void
): {unmatchedCount: number; multipleMatchCount: number} {
  let unmatchedCount = 0;
  let multipleMatchCount = 0;
  for (let point = 0; point < points.count; point++) {
    const position: CatalogPosition = [
      points.positions[2 * point],
      points.positions[2 * point + 1]
    ];
    let matchCount = 0;
    polygons.parts.forEach((feature, featureIndex) => {
      if (!feature.some(rings => isCatalogPointInPolygon(position, rings))) return;
      matchCount++;
      visit?.(point, featureIndex);
    });
    if (!matchCount) unmatchedCount++;
    if (matchCount > 1) multipleMatchCount++;
  }
  return {unmatchedCount, multipleMatchCount};
}

/** Candidate pair capacity of the join: every bounding box pair, capped. */
export function getRecipeCandidateCapacity(pointCount: number, featureCount: number): number {
  return Math.max(1, Math.min(pointCount * featureCount, 4000000));
}

export const MAXIMUM_RECIPE_POINTS = 100000;

export function makeRecipeEnumParameter(
  name: string,
  label: string,
  options: readonly string[],
  defaultValue: string
): CatalogParameterSpec {
  return {name, label, kind: 'enum', options, defaultValue};
}
