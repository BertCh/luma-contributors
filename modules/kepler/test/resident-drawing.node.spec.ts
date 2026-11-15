// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {Feature, Polygon} from 'geojson';
import {describe, expect, it} from 'vitest';
import {
  findDrawnFeatureAtPoint,
  measureDrawnGeometry,
  validateDrawnPolygon
} from '../../../examples/deck/kepler-compute/src/resident-drawing-geometry';
import {
  describeDrawingMaskProblems,
  getChangedDrawingLayerMasks,
  planDrawingLayerMasks
} from '../../../examples/deck/kepler-compute/src/resident-drawing-layer-masks';
import {
  buildExtractedTable,
  estimateArrowSchemaRowBytes,
  extractSelectedRows,
  makeDrawingLayerSource,
  makeDrawingPolygonFilter
} from '../../../examples/deck/kepler-compute/src/resident-drawing-actions';

const square = (x: number, y: number, size = 1): Feature<Polygon> => ({
  type: 'Feature',
  properties: {shape: 'Rectangle'},
  geometry: {
    type: 'Polygon',
    coordinates: [
      [
        [x, y],
        [x + size, y],
        [x + size, y + size],
        [x, y + size],
        [x, y]
      ]
    ]
  }
});

describe('validateDrawnPolygon', () => {
  it('accepts a simple polygon and a polygon with an inner hole', () => {
    expect(validateDrawnPolygon(square(0, 0)).status).toBe('valid');
    const withHole = square(0, 0, 10);
    withHole.geometry.coordinates.push([
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2],
      [2, 2]
    ]);
    expect(validateDrawnPolygon(withHole).status).toBe('valid');
  });

  it('reports reasons for invalid shapes', () => {
    const codes = (feature: Parameters<typeof validateDrawnPolygon>[0]) => {
      const result = validateDrawnPolygon(feature);
      return result.status === 'invalid' ? result.reasons.map(reason => reason.code) : [];
    };
    const bowtie: Feature<Polygon> = {
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2, 2],
            [2, 0],
            [0, 2],
            [0, 0]
          ]
        ]
      }
    };
    expect(codes(bowtie)).toEqual(['self-intersection']);
    const line = {
      type: 'LineString',
      coordinates: [
        [0, 0],
        [1, 1]
      ]
    } as const;
    expect(codes(line as unknown as Parameters<typeof codes>[0])).toEqual(['unsupported-geometry']);
    expect(codes(null)).toEqual(['empty-geometry']);
    const collinear = square(0, 0);
    collinear.geometry.coordinates = [
      [
        [0, 0],
        [1, 1],
        [2, 2],
        [0, 0]
      ]
    ];
    expect(codes(collinear)).toEqual(['zero-area']);
    const tiny = square(0, 0);
    tiny.geometry.coordinates = [
      [
        [0, 0],
        [1, 1],
        [0, 0]
      ]
    ];
    expect(codes(tiny)).toEqual(['too-few-vertices']);
    const outside = square(0, 0, 10);
    outside.geometry.coordinates.push([
      [20, 20],
      [21, 20],
      [21, 21],
      [20, 20]
    ]);
    expect(codes(outside)).toEqual(['hole-outside-shell']);
    const crossing = square(0, 0, 10);
    crossing.geometry.coordinates.push([
      [5, 5],
      [15, 5],
      [15, 6],
      [5, 5]
    ]);
    expect(codes(crossing)).toEqual(['hole-crosses-shell']);
    const nan = square(0, 0);
    nan.geometry.coordinates[0][1] = [Number.NaN, 0];
    expect(codes(nan)).toEqual(['non-finite-coordinate']);
  });

  it('warns, without splitting, for antimeridian and out-of-range geometry', () => {
    const result = validateDrawnPolygon(square(170, 10, 20));
    expect(result.status).toBe('valid');
    expect(result.warnings.map(warning => warning.code)).toContain('longitude-out-of-range');
    const spanning = square(-179, 0, 1);
    spanning.geometry.coordinates[0][1] = [179, 0];
    spanning.geometry.coordinates[0][2] = [179, 1];
    const spanResult = validateDrawnPolygon(spanning);
    expect(spanResult.warnings.map(warning => warning.code)).toContain('antimeridian-edge');
  });
});

describe('measureDrawnGeometry and hit testing', () => {
  it('measures a one-degree equatorial square', () => {
    const measurement = measureDrawnGeometry(square(0, 0).geometry);
    expect(measurement.lengthMeters / 1000).toBeCloseTo(444.8, 0);
    expect(measurement.areaSquareMeters / 1e6).toBeCloseTo(12364, -2);
  });

  it('selects the top-most feature, honoring holes', () => {
    const outer = square(0, 0, 10);
    outer.geometry.coordinates.push([
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2],
      [2, 2]
    ]);
    const small = square(1, 1, 1);
    expect(findDrawnFeatureAtPoint([outer, small], [1.5, 1.5], 0.01)).toBe(1);
    expect(findDrawnFeatureAtPoint([outer, small], [3, 3], 0.01)).toBe(-1);
    expect(findDrawnFeatureAtPoint([outer, small], [8, 8], 0.01)).toBe(0);
  });
});

describe('planDrawingLayerMasks', () => {
  const layers = [
    {id: 'a', type: 'point', dataId: 'd'},
    {id: 'b', type: 'point', dataId: 'd'},
    {id: 'c', type: 'point', dataId: 'd'},
    {id: 'g', type: 'geojson', dataId: 'd'}
  ];
  const filterFor = (id: string, layerId: string[], feature: Feature<Polygon>) => {
    const result = makeDrawingPolygonFilter(feature, {id, dataIds: ['d'], layerIds: layerId});
    if (result.status !== 'ready') throw new Error('expected a filter');
    return result.filter;
  };
  it('keeps distinct masks per targeted layer', () => {
    const plan = planDrawingLayerMasks(layers, [
      filterFor('f1', ['a'], square(0, 0)),
      filterFor('f2', ['b'], square(5, 5))
    ]);
    const [a, b, c] = plan;
    expect(a.status).toBe('masked');
    expect(b.status).toBe('masked');
    expect(c.status).toBe('unfiltered');
    expect(a.status === 'masked' && b.status === 'masked' && a.signature !== b.signature).toBe(
      true
    );
    const moved = planDrawingLayerMasks(layers, [
      filterFor('f1', ['a'], square(0, 0)),
      filterFor('f2', ['b'], square(6, 6))
    ]);
    expect(getChangedDrawingLayerMasks(plan, moved)).toEqual(['b']);
  });

  it('marks invalid and unmaskable targets with reasons', () => {
    const bowtie = square(0, 0);
    bowtie.geometry.coordinates = [
      [
        [0, 0],
        [2, 2],
        [2, 0],
        [0, 2],
        [0, 0]
      ]
    ];
    const plan = planDrawingLayerMasks(layers, [
      {id: 'bad', dataId: 'd', layerId: ['a'], type: 'polygon', value: bowtie},
      filterFor('f3', ['g'], square(0, 0))
    ]);
    expect(plan[0].status).toBe('invalid');
    expect(plan[3].status).toBe('unsupported');
    expect(describeDrawingMaskProblems(plan)).toHaveLength(2);
  });
});

describe('drawing to layer', () => {
  it('builds one row per feature and rejects invalid polygons', () => {
    const result = makeDrawingLayerSource([square(0, 0), square(2, 2)], {
      id: 'drawn',
      label: 'Drawn'
    });
    expect(result.status).toBe('ready');
    if (result.status === 'ready') {
      expect(result.source.source.numRows).toBe(2);
      expect(result.source.geometryColumn).toBe('geometry');
      expect(result.source.source.getChild('shape')!.get(0)).toBe('Rectangle');
    }
    const bad = square(0, 0);
    bad.geometry.coordinates = [
      [
        [0, 0],
        [2, 2],
        [2, 0],
        [0, 2],
        [0, 0]
      ]
    ];
    expect(makeDrawingLayerSource([bad], {id: 'x', label: 'x'}).status).toBe('invalid');
    expect(makeDrawingLayerSource([], {id: 'x', label: 'x'}).status).toBe('empty');
  });
});

describe('row extraction', () => {
  const makeSource = () =>
    new arrow.Table({
      value: arrow.vectorFromArray(Float32Array.from({length: 10}, (_, index) => index * 10))
    });
  it('slices ascending runs and appends original row identities', () => {
    const {table, sourceRowColumn} = buildExtractedTable(makeSource(), Uint32Array.of(1, 2, 7), {});
    expect(table.numRows).toBe(3);
    expect(Array.from(table.getChild('value')!.toArray())).toEqual([10, 20, 70]);
    expect(Array.from(table.getChild(sourceRowColumn)!.toArray())).toEqual([1, 2, 7]);
    expect(() => buildExtractedTable(makeSource(), Uint32Array.of(2, 1), {})).toThrow();
  });

  const makeFakes = (selectedCount: number, selectedRows: number[]) => {
    const source = makeSource();
    const dataset = {
      id: 'd',
      revision: 1,
      value: {
        source,
        resident: undefined,
        columns: {},
        prepared: {dictionaries: {}},
        analytics: {validity: {}}
      },
      rowCount: 10
    };
    const explorer = {
      dataset,
      isCurrent: true,
      currentRevision: 3,
      maximumInspectionPageSize: 100,
      maximumInspectionReadbackBytes: 1 << 20,
      async readSelectedCount() {
        return {revision: 3, datasetRevision: 1, selectedCount};
      },
      async readPage(offset: number, length: number) {
        return {
          revision: 3,
          datasetRevision: 1,
          sourceRowIndices: Uint32Array.from(selectedRows.slice(offset, offset + length)),
          columns: {},
          validity: {}
        };
      }
    };
    const session = {
      registry: {
        getDataset: () => dataset,
        reserveMemory: () => ({release() {}})
      }
    };
    return {session, explorer};
  };

  it('refuses over-limit and empty selections with distinct states', async () => {
    const {session, explorer} = makeFakes(3, [1, 2, 7]);
    const limits = {maximumRows: 2, maximumOutputBytes: 1 << 20};
    const tooMany = await extractSelectedRows(session as never, explorer as never, limits);
    expect(tooMany.status).toBe('unsupported');
    const none = makeFakes(0, []);
    expect(
      (await extractSelectedRows(none.session as never, none.explorer as never, limits)).status
    ).toBe('empty');
  });

  it('extracts exactly the selected rows', async () => {
    const {session, explorer} = makeFakes(3, [1, 2, 7]);
    const result = await extractSelectedRows(session as never, explorer as never, {
      maximumRows: 10,
      maximumOutputBytes: 1 << 20
    });
    if (result.status !== 'ready') throw new Error(JSON.stringify(result));
    if (result.status === 'ready') {
      expect(Array.from(result.value.sourceRowIndices)).toEqual([1, 2, 7]);
      expect(result.value.table.numRows).toBe(3);
    }
  });
});

it('schema row-byte floor covers unmaterialized resident sources', () => {
  const schema = new arrow.Schema([
    new arrow.Field('x', new arrow.Float32()),
    new arrow.Field('id', new arrow.Int32()),
    new arrow.Field('name', new arrow.Utf8())
  ]);
  expect(estimateArrowSchemaRowBytes(schema)).toBe(4 + 4 + 24);
});
