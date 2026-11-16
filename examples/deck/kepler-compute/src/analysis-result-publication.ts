// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  encodeKeplerResultPresentation,
  type CatalogOutputTable,
  type KeplerBindingResultPresentation,
  type KeplerResultEncoding,
  type KeplerResultReport
} from '@luma-contributors/kepler';
import {makeArrowTableFromCatalogTable} from './analysis-catalog-table';
import {getPointCloudCatalogCoordinates} from './analysis-catalog-pointcloud';
import type {CoordinateRoles} from './source-publication';
import {planRasterReprojection} from './raster-reprojection';

export type AnalysisResultPublication = {
  table: CatalogOutputTable;
  destination: 'map' | 'report';
  /** Explains a CRS transform or why geographic map display was refused. */
  mapMessage?: string;
  encoding?: KeplerResultEncoding;
  presentation?: KeplerBindingResultPresentation;
  report?: KeplerResultReport;
  coordinates?: CoordinateRoles;
};

/** Transform only the map copy. Computational geometry and raster samples remain in source units. */
function prepareAnalysisDisplayTable(table: CatalogOutputTable): {
  displayTable: CatalogOutputTable;
  mapMessage?: string;
  refused?: boolean;
} {
  const sourceCrs =
    table.coordinateSystem === 'EPSG:4326'
      ? 'EPSG:4326'
      : (table.raster?.crs ?? table.coordinateSystem);
  if (!sourceCrs || sourceCrs === 'EPSG:4326') return {displayTable: table};
  const refuse = (reason: string) => ({
    displayTable: table,
    refused: true,
    mapMessage: `Map display withheld: ${reason} The exact result remains available in this report/table. Declare a supported projection or transform the geometry to EPSG:4326 before map publication.`
  });
  if (sourceCrs === 'planar')
    return refuse('planar coordinates have no declared projection to longitude/latitude.');
  if (!table.geometryColumn) return refuse(`${sourceCrs} has no geographic display geometry.`);
  try {
    const transform = planRasterReprojection(sourceCrs, 'EPSG:4326').transformSourceToTarget;
    const transformCoordinates = (coordinates: unknown): unknown => {
      if (!Array.isArray(coordinates)) throw new Error('invalid geometry coordinates');
      if (typeof coordinates[0] === 'number') {
        const [longitude, latitude] = transform(coordinates[0], coordinates[1]);
        if (
          !Number.isFinite(longitude) ||
          !Number.isFinite(latitude) ||
          Math.abs(longitude) > 180 ||
          Math.abs(latitude) > 90
        )
          throw new Error('the declared projection produced invalid longitude/latitude');
        return [longitude, latitude, ...coordinates.slice(2)];
      }
      return coordinates.map(transformCoordinates);
    };
    const transformGeometry = (value: unknown): unknown => {
      if (value === null) return null;
      if (!value || typeof value !== 'object' || Array.isArray(value))
        throw new Error('invalid geometry');
      const {bbox: _bounds, crs: _projection, ...geometry} = value as Record<string, unknown>;
      if (geometry['type'] === 'Feature')
        return {...geometry, geometry: transformGeometry(geometry['geometry'])};
      if (geometry['type'] === 'GeometryCollection' && Array.isArray(geometry['geometries']))
        return {...geometry, geometries: geometry['geometries'].map(transformGeometry)};
      return {...geometry, coordinates: transformCoordinates(geometry['coordinates'])};
    };
    const geometry = table.columns[table.geometryColumn].map(value =>
      value === null ? null : JSON.stringify(transformGeometry(JSON.parse(String(value))))
    );
    return {
      displayTable: {
        ...table,
        coordinateSystem: 'EPSG:4326',
        columns: {...table.columns, [table.geometryColumn]: geometry}
      },
      mapMessage: `Map display transformed from ${sourceCrs} to EPSG:4326. The exact analysis table retains its source coordinates.`
    };
  } catch (error) {
    return refuse(
      `${sourceCrs} could not be transformed: ${error instanceof Error ? error.message : String(error)}.`
    );
  }
}

/** Adapt the shared encoder to source staging without altering the exact analysis table. */
export function prepareAnalysisResultPublication(
  table: CatalogOutputTable,
  presentation?: KeplerBindingResultPresentation,
  report?: KeplerResultReport,
  colorColumn?: string
) {
  const encoding = presentation
    ? encodeKeplerResultPresentation(table, presentation, colorColumn)
    : undefined;
  const display = prepareAnalysisDisplayTable(encoding?.table ?? table);
  const displayTable = makeArrowTableFromCatalogTable(display.displayTable);
  const coordinates =
    display.refused || display.displayTable.geometryColumn
      ? undefined
      : getPointCloudCatalogCoordinates(displayTable, table.label);
  const analysisResult: AnalysisResultPublication = {
    table,
    presentation,
    encoding,
    report,
    coordinates,
    mapMessage: display.mapMessage,
    destination: !display.refused && (table.geometryColumn || coordinates) ? 'map' : 'report'
  };
  return {table: displayTable, displayTable: display.displayTable, analysisResult};
}

/** Seed all classes, including empty ones, so native resident legends retain their colours. */
export function getAnalysisResultDictionaries(result?: AnalysisResultPublication) {
  const encoding = result?.encoding;
  return encoding ? {[encoding.classColumn]: {values: encoding.categories, ordered: false}} : {};
}

export const ANALYSIS_RESULT_METADATA_KEY = 'lumaAnalysisResult';

/** Serializable interpretation accompanies the published source and its saved native dataset. */
export function getAnalysisResultMetadata(result?: AnalysisResultPublication) {
  return result
    ? {
        destination: result.destination,
        mapMessage: result.mapMessage,
        presentation: result.encoding?.presentation ?? result.presentation?.presentation,
        legend: result.encoding?.legend ?? result.presentation?.legend ?? [],
        report: result.report
      }
    : undefined;
}

export type AnalysisResultReportView = {
  id: string;
  label: string;
  analysisResult: AnalysisResultPublication;
};
let activeReport: AnalysisResultReportView | null = null;
const reportListeners = new Set<() => void>();
export function getAnalysisResultReportView() {
  return activeReport;
}
export function subscribeAnalysisResultReports(listener: () => void) {
  reportListeners.add(listener);
  return () => {
    reportListeners.delete(listener);
  };
}
export function showAnalysisResultReport(report: AnalysisResultReportView) {
  activeReport = report;
  for (const listener of reportListeners) listener();
}
export function dismissAnalysisResultReport(report: AnalysisResultReportView) {
  if (activeReport !== report) return;
  activeReport = null;
  for (const listener of reportListeners) listener();
}
