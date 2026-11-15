// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeGPUAnalysisSpacingStatistics,
  type GPUAnalysisColumnProfile,
  type GPUAnalysisCoordinateModel,
  type GPUAnalysisRowSemantics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  countCollectionVertices,
  type CatalogFeature,
  type CatalogFeatureCollection
} from '../../src/analysis/catalog-geojson';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

// Profiles for the engine specifications, built from the curated samples (real coordinates, so the
// spacing statistics are realistic) without the profile module, which another stream is writing.

/** Mean of every position of a geometry: the anchor the spacing statistics use. */
function getAnchor(feature: CatalogFeature): [number, number] | undefined {
  const positions: [number, number][] = [];
  const visit = (value: unknown): void => {
    if (Array.isArray(value) && typeof value[0] === 'number' && typeof value[1] === 'number') {
      positions.push([value[0], value[1]]);
    } else if (Array.isArray(value)) {
      value.forEach(visit);
    }
  };
  visit(feature.geometry?.coordinates);
  if (positions.length === 0) {
    return undefined;
  }
  return [
    positions.reduce((total, position) => total + position[0], 0) / positions.length,
    positions.reduce((total, position) => total + position[1], 0) / positions.length
  ];
}

/** Column statistics from feature properties. A string that parses as a date is a time column. */
export function inferColumns(
  features: readonly CatalogFeature[],
  keep?: readonly string[]
): GPUAnalysisColumnProfile[] {
  const names = [...new Set(features.flatMap(feature => Object.keys(feature.properties ?? {})))];
  return names
    .filter(name => keep === undefined || keep.includes(name))
    .map((name): GPUAnalysisColumnProfile => {
      const values = features.map(feature => feature.properties?.[name]);
      const present = values.filter(value => value !== null && value !== undefined);
      const base = {name, rowCount: features.length, nullCount: values.length - present.length};
      if (present.every(value => typeof value === 'number')) {
        const numbers = present as number[];
        const mean = numbers.reduce((total, value) => total + value, 0) / numbers.length;
        const integral = numbers.every(value => Number.isInteger(value));
        return {
          ...base,
          type: integral ? 'integer' : 'number',
          minimum: Math.min(...numbers),
          maximum: Math.max(...numbers),
          mean,
          standardDeviation: Math.sqrt(
            numbers.reduce((total, value) => total + (value - mean) ** 2, 0) / numbers.length
          ),
          distinctCount: new Set(numbers).size,
          integral
        };
      }
      const strings = present.map(String);
      if (strings.every(text => /^\d{4}-\d{2}-\d{2}/.test(text))) {
        const times = strings.map(text => Date.parse(text));
        return {
          ...base,
          type: 'time',
          minimum: Math.min(...times),
          maximum: Math.max(...times),
          distinctCount: new Set(strings).size
        };
      }
      return {...base, type: 'category', distinctCount: new Set(strings).size};
    });
}

export type TestProfileOptions = {
  datasetId: string;
  label: string;
  collection: CatalogFeatureCollection;
  kind: 'points' | 'lines' | 'polygons' | 'network';
  catalogGeometry: 'point' | 'line' | 'polygon';
  coordinates?: {model: GPUAnalysisCoordinateModel; source: 'guess' | 'declared'};
  rowSemantics?: GPUAnalysisRowSemantics;
  /** Property names to keep; all by default. */
  keepColumns?: readonly string[];
  columns?: readonly GPUAnalysisColumnProfile[];
};

/** A profile of a feature collection: spacing over the anchors, extent, vertex count, columns. */
export function buildTestProfile(options: TestProfileOptions): KeplerDatasetProfile {
  const features = options.collection.features;
  const anchors = features
    .map(getAnchor)
    .filter((anchor): anchor is [number, number] => anchor !== undefined);
  const coordinates = options.coordinates ?? {
    model: 'geographic' as const,
    source: 'guess' as const
  };
  const spacing = computeGPUAnalysisSpacingStatistics(anchors.flat(), {
    coordinateModel: coordinates.model
  });
  const isArea = options.kind === 'polygons';
  return {
    datasetId: options.datasetId,
    label: options.label,
    kind: options.kind,
    rowCount: features.length,
    vertexCount: countCollectionVertices(options.collection),
    coordinates,
    ...(spacing ? {extent: spacing.extent, spacing} : {}),
    columns: options.columns ?? inferColumns(features, options.keepColumns),
    ...(options.rowSemantics
      ? {rowSemantics: {value: options.rowSemantics, source: 'declared' as const}}
      : {}),
    ...(isArea ? {contiguity: {minimum: 3, median: 8, maximum: 8, islandCount: 0}} : {}),
    catalogGeometry: options.catalogGeometry,
    geometryColumns: ['geometry']
  };
}

export type CuratedSampleName =
  | 'incidents'
  | 'districts'
  | 'gauges'
  | 'streets'
  | 'facilities'
  | 'demand'
  | 'before'
  | 'after';

/** One of the curated samples as a profile. */
export function buildSampleProfile(
  name: CuratedSampleName,
  overrides: Partial<TestProfileOptions> = {}
): KeplerDatasetProfile {
  const common = {datasetId: name, label: name} as const;
  switch (name) {
    case 'incidents':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('city-incidents')['points'],
        kind: 'points',
        catalogGeometry: 'point',
        rowSemantics: 'events',
        ...overrides
      });
    case 'districts':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('city-district-rates')['areas'],
        kind: 'polygons',
        catalogGeometry: 'polygon',
        rowSemantics: 'objects',
        ...overrides
      });
    case 'gauges':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('rain-gauges')['points'],
        kind: 'points',
        catalogGeometry: 'point',
        rowSemantics: 'surface-samples',
        ...overrides
      });
    case 'streets':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('street-grid-facilities')['network'],
        kind: 'lines',
        catalogGeometry: 'line',
        ...overrides
      });
    case 'facilities':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('street-grid-facilities')['facilities'],
        kind: 'points',
        catalogGeometry: 'point',
        ...overrides
      });
    case 'demand':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('street-grid-facilities')['demand'],
        kind: 'points',
        catalogGeometry: 'point',
        ...overrides
      });
    case 'before':
    case 'after':
      return buildTestProfile({
        ...common,
        collection: createKeplerCuratedAnalysisSample('before-after-incidents')[name],
        kind: 'points',
        catalogGeometry: 'point',
        rowSemantics: 'events',
        ...overrides
      });
  }
}

/** A table without geometry. */
export function buildTableProfile(
  datasetId: string,
  columns: readonly GPUAnalysisColumnProfile[],
  rowCount: number
): KeplerDatasetProfile {
  return {datasetId, label: datasetId, kind: 'attributes', rowCount, columns, geometryColumns: []};
}

/** The same profile with a different number of rows, as if the layer were bigger or smaller. */
export function withRowCount(
  profile: KeplerDatasetProfile,
  rowCount: number
): KeplerDatasetProfile {
  return {
    ...profile,
    rowCount,
    vertexCount: profile.kind === 'points' ? rowCount : profile.vertexCount,
    columns: profile.columns.map(column => ({...column, rowCount}))
  };
}

/** `count` incident points spread evenly through the sample, as a small dataset. */
export function buildSmallPointProfile(count: number): KeplerDatasetProfile {
  const collection = createKeplerCuratedAnalysisSample('city-incidents')['points'];
  return buildTestProfile({
    datasetId: 'few-points',
    label: 'few-points',
    collection: {
      type: 'FeatureCollection',
      features: collection.features
        .filter((_, index) => index % Math.floor(collection.features.length / count) === 0)
        .slice(0, count)
    },
    kind: 'points',
    catalogGeometry: 'point',
    rowSemantics: 'events'
  });
}

/** Small point collections for engine regressions, preserving the exact probe coordinates. */
export function createTestPointCollection(
  positions: readonly (readonly [number, number])[],
  properties: (index: number) => Record<string, unknown> = () => ({})
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: positions.map((coordinates, index) => ({
      type: 'Feature',
      geometry: {type: 'Point', coordinates: [...coordinates]},
      properties: properties(index)
    }))
  };
}
