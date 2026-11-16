// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {renderToStaticMarkup} from 'react-dom/server';
import {createElement} from 'react';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
import {
  computeClassBreaksOnCPU,
  getGPUClassBreaksParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  getKeplerBindingResultPresentations,
  encodeKeplerResultPresentation,
  getKeplerResultEncodingStyle,
  getKeplerOperationDescriptor,
  getKeplerResultRampColors,
  type CatalogOutputTable,
  type KeplerOperationDescriptor
} from '../src/index';
import {getAnalysisLegendModel} from '../../../examples/deck/kepler-compute/src/analysis-legend';
import {
  getAnalysisRampPalette,
  getAnalysisHotspotColors,
  getAnalysisMoranColors
} from '../../../examples/deck/kepler-compute/src/analysis-result-colors';
import {AnalysisResultReport} from '../../../examples/deck/kepler-compute/src/analysis-result-report-view';
import {
  prepareAnalysisResultPublication,
  getAnalysisResultMetadata,
  getAnalysisResultDictionaries,
  showAnalysisResultReport,
  dismissAnalysisResultReport,
  getAnalysisResultReportView
} from '../../../examples/deck/kepler-compute/src/analysis-result-publication';

function present(table: CatalogOutputTable, descriptor: KeplerOperationDescriptor) {
  return getKeplerBindingResultPresentations(
    {
      operationId: descriptor.bindingId,
      engine: 'cpu-reference',
      parameters: {},
      tables: [table],
      diagnostics: [],
      summary: {},
      dispatchedContributors: []
    },
    descriptor
  )[0];
}
function makeNumericTable(values: (number | null)[]): CatalogOutputTable {
  return {
    id: 'density-cells',
    label: 'Density',
    rowCount: values.length,
    columns: {density: values, geometry: values.map(() => '{"type":"Point","coordinates":[0,0]}')},
    columnTypes: {density: 'float64', geometry: 'string'},
    geometryColumn: 'geometry'
  };
}
const densityDescriptor = getKeplerOperationDescriptor('kernel-density-grid')!;

describe('AX shared publication encoding', () => {
  test('catalog and workbench use the same exact breaks before Float32 rounding', () => {
    const table = makeNumericTable([0, 1, 1, 1.000000001, 2, 3, null, 9]);
    const original = structuredClone(table);
    const presentation = present(table, densityDescriptor);
    const encoding = encodeKeplerResultPresentation(table, presentation)!;
    const staged = prepareAnalysisResultPublication(table, presentation);
    expect(staged.analysisResult.encoding).toEqual(encoding);
    expect(
      JSON.parse(JSON.stringify(getAnalysisResultMetadata(staged.analysisResult))).presentation
        .breaks
    ).toEqual(encoding.presentation.breaks);
    expect(Array.from(staged.table.getChild(encoding.classColumn)!)).toEqual(encoding.values);
    expect(encoding.values[1]).toBe(encoding.values[2]);
    expect(encoding.values[2]).not.toBe(encoding.values[3]);
    for (const [index, entry] of encoding.legend.entries())
      expect(encoding.values.filter(value => value === encoding.categories[index])).toHaveLength(
        entry.count!
      );
    const style = getKeplerResultEncodingStyle(encoding);
    expect(style.visualChannels.colorField.name).toContain('density');
    expect(style.visConfig.colorRange.name).toBe('density');
    expect(style.visConfig.outline).toBe(true);
    expect(table).toEqual(original);
  });

  test('hotspot complete legends agree, including a row without neighbours', () => {
    const table: CatalogOutputTable = {
      id: 'hotspots',
      label: 'Hotspots',
      rowCount: 8,
      columns: {
        classification: [
          'hot-99',
          'hot-95',
          'hot-90',
          'not-significant',
          'cold-90',
          'cold-95',
          'cold-99',
          'not-tested'
        ],
        neighbor_count: [1, 1, 1, 1, 1, 1, 1, 0]
      },
      columnTypes: {classification: 'string', neighbor_count: 'int32'}
    };
    const presentation = present(table, getKeplerOperationDescriptor('hotspot-gi-star')!);
    const encoding = encodeKeplerResultPresentation(table, presentation)!;
    const legend = getAnalysisLegendModel('hot-spots-lattice', {hotspot: {invalidCount: 1}})!;
    expect(legend.entries).toEqual(encoding.legend.map(({label, color}) => ({label, color})));
    expect(encoding.values[7]).toBe('Excluded: no neighbours');
    expect(encoding.legend.at(-1)).toMatchObject({color: [70, 70, 70], count: 1});
    expect(getAnalysisHotspotColors()).toEqual(
      [...legend.entries.slice(0, 7)].reverse().map(entry => entry.color)
    );
  });

  test('local Moran complete legends agree and untested rows have the excluded fill', () => {
    const table: CatalogOutputTable = {
      id: 'local-moran',
      label: 'Local Moran',
      rowCount: 6,
      columns: {
        cluster: ['HH', 'HL', 'LH', 'LL', 'not-significant', 'not-tested'],
        p_value: [0.01, 0.02, 0.03, 0.04, 0.5, null]
      },
      columnTypes: {cluster: 'string', p_value: 'float64'}
    };
    const descriptor = getKeplerOperationDescriptor('local-moran-statistics')!;
    table.id = descriptor.outputs[0].tableId;
    const presentation = present(table, descriptor);
    const encoding = encodeKeplerResultPresentation(table, presentation)!;
    const legend = getAnalysisLegendModel('local-moran-points', {localMoran: {invalidCount: 1}})!;
    expect(legend.entries).toEqual(encoding.legend.map(({label, color}) => ({label, color})));
    expect(getAnalysisMoranColors().at(-1)).toEqual([70, 70, 70]);
    const selected = encodeKeplerResultPresentation(table, presentation, 'p_value')!;
    expect(selected.presentation.column).toBe('p_value');
    expect(selected.presentation.scale).toBe('sequential');
    expect(selected.legend.some(entry => entry.label.includes('cluster'))).toBe(false);
    expect(encodeKeplerResultPresentation(table, presentation, 'cluster')).toEqual(encoding);
  });

  test('choropleth legend and produced palette use only active classes, including constant data', () => {
    for (const values of [
      [7, 7, 7, 7],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]) {
      const classified = computeClassBreaksOnCPU(
        new Float32Array(values),
        getGPUClassBreaksParameterValues({method: 'quantile', classCount: 5}, 5),
        {maximumClassCount: 5, methods: ['quantile']}
      );
      const classCount = classified.classCount[0];
      const legend = getAnalysisLegendModel('quantile', {
        minimum: values[0],
        maximum: values.at(-1),
        classCount,
        breaks: Array.from(classified.breaks)
      })!;
      expect(legend.entries).toHaveLength(classCount);
      expect(legend.entries.some(entry => entry.label.includes('NaN'))).toBe(false);
      expect(legend.entries.map(entry => entry.color)).toEqual(
        getKeplerResultRampColors('sequential', classCount)
      );
      const palette = getAnalysisRampPalette();
      for (const [index, entry] of legend.entries.entries()) {
        const packed = palette[(classCount - 1) * 8 + index];
        expect([packed & 255, (packed >>> 8) & 255, (packed >>> 16) & 255]).toEqual(entry.color);
      }
      if (classCount === 1)
        expect(legend.entries).toEqual([{label: '7 to 7', color: [66, 162, 142]}]);
    }
    const table = makeNumericTable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const presentation = present(table, densityDescriptor);
    const plan = presentation.presentation!;
    const handwritten = getAnalysisLegendModel('quantile', {
      minimum: 0,
      maximum: 9,
      breaks: [0, ...plan.breaks, 9],
      classCount: plan.ranges.length
    })!;
    expect(handwritten.entries).toEqual(
      presentation.legend.map(({label, color}) => ({label, color}))
    );
  });

  test('coordinate-based point-cloud results remain map results with their AX style', () => {
    const table: CatalogOutputTable = {
      id: 'pointcloud-height',
      label: 'Point cloud heights',
      rowCount: 2,
      columns: {longitude: [0, 1], latitude: [0, 1], altitude: [1, 2], height: [1, 2]},
      columnTypes: {
        longitude: 'float64',
        latitude: 'float64',
        altitude: 'float64',
        height: 'float64'
      }
    };
    const descriptor = {
      ...densityDescriptor,
      outputs: [
        {
          tableId: table.id,
          kind: 'points' as const,
          presentation: {column: 'height', scale: 'sequential' as const}
        }
      ]
    };
    const staged = prepareAnalysisResultPublication(table, present(table, descriptor));
    expect(staged.analysisResult.destination).toBe('map');
    expect(staged.analysisResult.coordinates).toEqual({
      longitude: 'longitude',
      latitude: 'latitude',
      altitude: 'altitude'
    });
    expect(
      getKeplerResultEncodingStyle(staged.analysisResult.encoding!).visualChannels.colorScale
    ).toBe('customOrdinal');
  });

  test('seeds empty classes in the native dictionary and avoids colliding with source columns', () => {
    const table = makeNumericTable([1, 1, 1]);
    table.columns['analysis_result_class (density)'] = ['source', 'source', 'source'];
    table.columnTypes['analysis_result_class (density)'] = 'string';
    const staged = prepareAnalysisResultPublication(table, present(table, densityDescriptor));
    const encoding = staged.analysisResult.encoding!;
    expect(encoding.classColumn).toBe('analysis_result_class (density)_');
    const prepared = prepareArrowTableForGPUAnalytics(staged.table, {
      columns: [encoding.classColumn],
      dictionaries: getAnalysisResultDictionaries(staged.analysisResult)
    });
    expect(prepared.dictionaries[encoding.classColumn].values).toEqual(encoding.categories);
    expect(table.columns['analysis_result_class (density)']).toEqual([
      'source',
      'source',
      'source'
    ]);
  });

  test('empty hotspot confidence classes remain seeded with exact native colours', () => {
    const table: CatalogOutputTable = {
      id: 'hotspots',
      label: 'Hotspots',
      rowCount: 1,
      columns: {classification: ['hot-99'], neighbor_count: [1]},
      columnTypes: {classification: 'string', neighbor_count: 'int32'}
    };
    const staged = prepareAnalysisResultPublication(
      table,
      present(table, getKeplerOperationDescriptor('hotspot-gi-star')!)
    );
    const encoding = staged.analysisResult.encoding!;
    expect(encoding.legend.filter(entry => entry.count === 0)).toHaveLength(6);
    const prepared = prepareArrowTableForGPUAnalytics(staged.table, {
      columns: [encoding.classColumn],
      dictionaries: getAnalysisResultDictionaries(staged.analysisResult)
    });
    expect(prepared.dictionaries[encoding.classColumn].values).toEqual(encoding.categories);
    expect(prepared.dictionaries[encoding.classColumn].values).toHaveLength(7);
  });

  test('report-only results retain exact table values and use identity-safe view dismissal', () => {
    const table: CatalogOutputTable = {
      id: 'global',
      label: 'Global Moran',
      rowCount: 1,
      columns: {moran_i: [0.12345678912345]},
      columnTypes: {moran_i: 'float64'}
    };
    const result = prepareAnalysisResultPublication(table).analysisResult;
    expect(result.destination).toBe('report');
    const first = {id: 'first', label: table.label, analysisResult: result};
    const second = {...first, id: 'second'};
    showAnalysisResultReport(first);
    showAnalysisResultReport(second);
    dismissAnalysisResultReport(first);
    expect(getAnalysisResultReportView()).toBe(second);
    expect(renderToStaticMarkup(createElement(AnalysisResultReport, {result}))).toContain(
      '0.12345678912345'
    );
    dismissAnalysisResultReport(second);
    expect(getAnalysisResultReportView()).toBeNull();
  });
});

describe('CRS-aware analysis map publication', () => {
  test.each([
    0, 500000
  ])('keeps unreferenced planar geometry at origin %s in the report even when values resemble degrees', origin => {
    const table = {...makeNumericTable([1.234567891234]), coordinateSystem: 'planar' as const};
    table.columns['geometry'] = [JSON.stringify({type: 'Point', coordinates: [origin, 20]})];
    const original = structuredClone(table);
    const staged = prepareAnalysisResultPublication(table, present(table, densityDescriptor));
    expect(staged.analysisResult.destination).toBe('report');
    expect(staged.analysisResult.coordinates).toBeUndefined();
    expect(staged.analysisResult.table).toBe(table);
    expect(staged.analysisResult.mapMessage).toMatch(/planar.*no declared projection/);
    const markup = renderToStaticMarkup(
      createElement(AnalysisResultReport, {result: staged.analysisResult})
    );
    expect(markup).toContain('Map display withheld');
    expect(markup).toContain('1.234567891234');
    expect(getAnalysisResultMetadata(staged.analysisResult)?.mapMessage).toBe(
      staged.analysisResult.mapMessage
    );
    expect(table).toEqual(original);
  });

  test('transforms a declared Web Mercator display copy, including nested geometry and height, while retaining exact source data and classes', () => {
    const table = makeNumericTable([1, 2, null]);
    table.coordinateSystem = 'planar';
    table.raster = {
      type: 'raster',
      id: 'mercator',
      crs: 'EPSG:3857',
      cellSizeMode: 'web-mercator',
      grid: makeGPURasterMetadata({width: 3, height: 1, cellSize: 1}),
      band: {
        index: 1,
        values: new Float32Array([1, 2, 3]),
        validity: new Uint8Array([1, 1, 1]),
        noDataValue: null
      }
    };
    const position = [1113194.9079327357, 2273030.926987689, 7];
    table.label = 'Point cloud projected geometry';
    // Retained scalar columns must not override transformed GeoJSON as the display position.
    table.columns['longitude'] = [position[0], position[0], null];
    table.columns['latitude'] = [position[1], position[1], null];
    table.columns['altitude'] = [7, 7, null];
    table.columnTypes['longitude'] = 'float64';
    table.columnTypes['latitude'] = 'float64';
    table.columnTypes['altitude'] = 'float64';

    table.columns['geometry'] = [
      JSON.stringify({type: 'Point', coordinates: position, bbox: [1, 2, 3, 4]}),
      JSON.stringify({
        type: 'GeometryCollection',
        geometries: [
          {type: 'LineString', coordinates: [[0, 0], position]},
          {type: 'Polygon', coordinates: [[[0, 0], position, [0, 0]]]}
        ]
      }),
      null
    ];
    const original = structuredClone(table);
    const staged = prepareAnalysisResultPublication(table, present(table, densityDescriptor));
    expect(staged.analysisResult.destination).toBe('map');
    expect(staged.displayTable.coordinateSystem).toBe('EPSG:4326');
    expect(staged.analysisResult.coordinates).toBeUndefined();
    const point = JSON.parse(String(staged.table.getChild('geometry')!.get(0)));
    expect(point.coordinates[0]).toBeCloseTo(10, 10);
    expect(point.coordinates[1]).toBeCloseTo(20, 10);
    expect(point.coordinates[2]).toBe(7);
    expect(point.bbox).toBeUndefined();
    const collection = JSON.parse(String(staged.table.getChild('geometry')!.get(1)));
    expect(collection.geometries[0].coordinates[1]).toEqual(point.coordinates);
    expect(collection.geometries[1].coordinates[0][1]).toEqual(point.coordinates);
    expect(staged.table.getChild('geometry')!.get(2)).toBeNull();
    expect(Array.from(staged.table.getChild(staged.analysisResult.encoding!.classColumn)!)).toEqual(
      staged.analysisResult.encoding!.values
    );
    expect(staged.analysisResult.table).toBe(table);
    expect(staged.analysisResult.mapMessage).toMatch(/transformed from EPSG:3857 to EPSG:4326/);
    expect(table).toEqual(original);
  });

  test('refuses malformed projected geometry instead of falling back to drawing source coordinates', () => {
    const table = makeNumericTable([1]);
    table.raster = {
      type: 'raster',
      id: 'mercator',
      crs: 'EPSG:3857',
      cellSizeMode: 'web-mercator',
      grid: makeGPURasterMetadata({width: 1, height: 1, cellSize: 1}),
      band: {
        index: 1,
        values: new Float32Array([1]),
        validity: new Uint8Array([1]),
        noDataValue: null
      }
    };
    table.columns['geometry'] = ['not GeoJSON'];
    const staged = prepareAnalysisResultPublication(table);
    expect(staged.analysisResult.destination).toBe('report');
    expect(staged.analysisResult.mapMessage).toMatch(/could not be transformed/);
    expect(staged.analysisResult.table.columns['geometry']).toEqual(['not GeoJSON']);
  });
});
