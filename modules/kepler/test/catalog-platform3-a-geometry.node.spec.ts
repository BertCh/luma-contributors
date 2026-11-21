// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  KEPLER_CATALOG_BINDINGS,
  getCatalogBinding,
  runCatalogOperation,
  type CatalogRunResult
} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM3_A_BINDINGS} from '../src/analysis/catalog-platform3-a-bindings';

type Properties = Record<string, unknown>;

function collection(
  geometries: {type: string; coordinates: unknown}[],
  properties: Properties[] = []
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: geometries.map((geometry, index) => ({
      id: `f${index}`,
      geometry,
      properties: properties[index] ?? {}
    }))
  };
}

const points = (positions: number[][], properties: Properties[] = []) =>
  collection(
    positions.map(coordinates => ({type: 'Point', coordinates})),
    properties
  );
const lines = (parts: number[][][], properties: Properties[] = []) =>
  collection(
    parts.map(coordinates => ({type: 'LineString', coordinates})),
    properties
  );
const polygons = (rings: number[][][][], properties: Properties[] = []) =>
  collection(
    rings.map(coordinates => ({type: 'Polygon', coordinates})),
    properties
  );
const square = (x: number, y: number, size: number) => [
  [x, y],
  [x + size, y],
  [x + size, y + size],
  [x, y + size],
  [x, y]
];

function run(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): CatalogRunResult {
  const binding = getCatalogBinding(id);
  if (!binding) throw new Error(`Missing binding ${id}`);
  const result = runCatalogOperation(binding, parameters, inputs);
  // Every declared identity column must exist in the matching output table.
  for (const table of result.tables) {
    const spec = binding.outputs.find(output => output.tableId === table.id);
    expect(spec, `${id} declares table ${table.id}`).toBeDefined();
    for (const column of spec!.identityColumns)
      expect(Object.keys(table.columns), `${id}.${table.id} has ${column}`).toContain(column);
    if (table.geometryColumn)
      for (const text of table.columns[table.geometryColumn])
        if (text !== null) expect(() => JSON.parse(String(text))).not.toThrow();
  }
  return result;
}

const planar = {coordinateSystem: 'planar'};
const column = (result: CatalogRunResult, name: string, tableIndex = 0) =>
  result.tables[tableIndex].columns[name];
const geometryAt = (result: CatalogRunResult, row: number, tableIndex = 0) =>
  JSON.parse(String(result.tables[tableIndex].columns['geometry'][row]));

test('family A bindings are registered with unique ids and honest declarations', () => {
  const ids = KEPLER_CATALOG_BINDINGS.map(binding => binding.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(PLATFORM3_A_BINDINGS.length).toBeGreaterThanOrEqual(20);
  for (const binding of PLATFORM3_A_BINDINGS) {
    expect(binding.capabilityIds.length).toBeGreaterThan(0);
    expect(binding.contributorNames.length).toBeGreaterThan(0);
    expect(binding.declaredBehavior.length).toBeGreaterThan(0);
    expect(binding.limitation.length).toBeGreaterThan(10);
    expect(binding.outputs.every(output => output.identityColumns.length > 0)).toBe(true);
    if (binding.engine === 'gpu-or-cpu') {
      expect(binding.gpuContributorNames.length).toBeGreaterThan(0);
      expect(binding.runGpu).toBeTypeOf('function');
      expect(binding.run).toBeTypeOf('function');
    } else {
      expect(binding.engine).toBe('cpu-reference');
      expect(binding.gpuContributorNames).toEqual([]);
    }
  }
});

test('shape descriptors score a square and an L shape', () => {
  const result = run('geometry-shape-descriptors', planar, {
    polygons: polygons([
      [square(0, 0, 1)],
      [
        [
          [0, 0],
          [2, 0],
          [2, 1],
          [1, 1],
          [1, 2],
          [0, 2],
          [0, 0]
        ]
      ]
    ])
  });
  expect(column(result, 'area')[0]).toBeCloseTo(1, 9);
  expect(column(result, 'perimeter')[0]).toBeCloseTo(4, 9);
  expect(column(result, 'compactness')[0]).toBeCloseTo(Math.PI / 4, 9);
  expect(column(result, 'convexity')[0]).toBeCloseTo(1, 9);
  expect(column(result, 'elongation')[0]).toBeCloseTo(0, 9);
  expect(column(result, 'area')[1]).toBeCloseTo(3, 9);
  expect(column(result, 'convexity')[1] as number).toBeCloseTo(3 / 3.5, 9);
});

test('minimum bounds find the rotated rectangle and enclosing circle', () => {
  const diamond = polygons([
    [
      [
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
        [0, 1]
      ]
    ]
  ]);
  const rectangle = run(
    'geometry-minimum-bounds',
    {...planar, shape: 'rectangle'},
    {features: diamond}
  );
  expect(column(rectangle, 'area')[0]).toBeCloseTo(2, 9);
  expect(column(rectangle, 'width')[0]).toBeCloseTo(Math.SQRT2, 9);
  const circle = run('geometry-minimum-bounds', {...planar, shape: 'circle'}, {features: diamond});
  expect(column(circle, 'radius')[0]).toBeCloseTo(1, 9);
  expect(column(circle, 'source_id')[0]).toBe('f0');
  const degenerate = run('geometry-minimum-bounds', planar, {features: points([[1, 1]])});
  expect(degenerate.tables[0].rowCount).toBe(0);
  expect(degenerate.diagnostics.map(item => item.code)).toContain('degenerate-feature');
});

test('label point and maximum inscribed circle share the pole of inaccessibility', () => {
  const input = {polygons: polygons([[square(0, 0, 10)]])};
  const label = run('geometry-label-point', planar, input);
  expect(column(label, 'clearance')[0]).toBeCloseTo(5, 3);
  expect(geometryAt(label, 0).coordinates[0]).toBeCloseTo(5, 2);
  const circle = run('geometry-maximum-inscribed-circle', planar, input);
  expect(column(circle, 'clearance')[0]).toBeCloseTo(5, 3);
  expect(geometryAt(circle, 0).type).toBe('Polygon');
});

test('minimum clearance measures the closest vertex to a non-adjacent segment', () => {
  const result = run('geometry-minimum-clearance', planar, {
    features: lines([
      [
        [0, 0],
        [10, 0],
        [10, 1],
        [0, 1]
      ]
    ])
  });
  expect(column(result, 'clearance')[0]).toBeCloseTo(1, 9);
  expect(column(result, 'status')[0]).toBe('ok');
  const tiny = run('geometry-minimum-clearance', planar, {
    features: lines([
      [
        [0, 0],
        [1, 1]
      ]
    ])
  });
  expect(column(tiny, 'status')[0]).toBe('degenerate');
});

test('geographic distribution reports centers and an ellipse per group', () => {
  const result = run(
    'geographic-distribution',
    {...planar, groupProperty: 'kind'},
    {
      features: points(
        [
          [0, 0],
          [2, 0],
          [0, 2],
          [2, 2],
          [10, 10]
        ],
        [{kind: 'a'}, {kind: 'a'}, {kind: 'a'}, {kind: 'a'}, {kind: 'b'}]
      )
    }
  );
  expect(result.tables[0].rowCount).toBe(2);
  expect(column(result, 'group')).toEqual(['a', 'b']);
  expect(column(result, 'standard_distance')[0]).toBeCloseTo(Math.SQRT2, 9);
  expect(column(result, 'major_radius')[0]).toBeCloseTo(1, 9);
  expect(column(result, 'major_radius')[1]).toBeNull();
  expect(result.tables[1].rowCount).toBe(4);
  const mean = JSON.parse(String(column(result, 'geometry', 1)[0]));
  expect(mean.coordinates[0]).toBeCloseTo(1, 9);
});

test('hilbert order follows the curve and puts invalid features last', () => {
  const features = collection([
    {type: 'Point', coordinates: [1, 0]},
    {type: 'Point', coordinates: [0, 1]},
    {type: 'Point', coordinates: [0, 0]},
    {type: 'Point', coordinates: [1, 1]}
  ]);
  const result = run('spatial-hilbert-order', {order: 1}, {features});
  expect(column(result, 'source_index')).toEqual([2, 1, 3, 0]);
  expect(column(result, 'hilbert_key')).toEqual([0, 1, 2, 3]);
  const withInvalid = {
    type: 'FeatureCollection' as const,
    features: [...features.features, {geometry: null, properties: {}}]
  };
  const invalid = run('spatial-hilbert-order', {order: 4}, {features: withInvalid});
  expect(column(invalid, 'source_index')[4]).toBe(4);
  expect(invalid.diagnostics.map(item => item.code)).toContain('feature-without-geometry');
});

test('join candidates and predicates keep both identities', () => {
  const left = collection([
    {type: 'Point', coordinates: [5, 5]},
    {type: 'Point', coordinates: [50, 50]},
    {type: 'Polygon', coordinates: [square(2, 2, 3)]},
    {type: 'Polygon', coordinates: [square(8, 8, 5)]}
  ]);
  const right = polygons([[square(0, 0, 10)], [square(40, 40, 20)]]);
  const candidates = run('spatial-join-bounds-candidates', planar, {left, right});
  expect(column(candidates, 'left_index')).toEqual([0, 1, 2, 3]);
  expect(column(candidates, 'right_index')).toEqual([0, 1, 0, 0]);
  const grown = run('spatial-join-bounds-candidates', {...planar, distance: 3}, {left, right});
  expect(grown.tables[0].rowCount).toBe(4);
  const within = run('spatial-join-predicate', {predicate: 'within'}, {left, right});
  expect(column(within, 'join_id')).toEqual(['0:0', '1:1', '2:0']);
  const intersects = run('spatial-join-predicate', {predicate: 'intersects'}, {left, right});
  expect(column(intersects, 'join_id')).toEqual(['0:0', '1:1', '2:0', '3:0']);
  const contains = run(
    'spatial-join-predicate',
    {predicate: 'contains'},
    {left: polygons([[square(0, 0, 100)]]), right}
  );
  expect(column(contains, 'right_index')).toEqual([0, 1]);
  const holed = run(
    'spatial-join-predicate',
    {predicate: 'within'},
    {
      left: points([[5, 5]]),
      right: polygons([[square(0, 0, 10), square(4, 4, 2).reverse()]])
    }
  );
  expect(holed.tables[0].rowCount).toBe(0);
});

test('coverage validity reports overlaps and accepts shared edges', () => {
  const result = run(
    'geometry-coverage-validity',
    {},
    {
      polygons: polygons([
        [square(0, 0, 10)],
        [square(5, 5, 10)],
        [square(10, 0, 10)],
        [square(0, 0, 10)]
      ])
    }
  );
  const ids = column(result, 'overlap_id');
  expect(ids).toContain('0:1');
  expect(ids).toContain('0:3');
  expect(ids).not.toContain('0:2');
  expect(column(result, 'status', 1)[2]).toBe('overlaps');
  const clean = run(
    'geometry-coverage-validity',
    {},
    {
      polygons: polygons([[square(0, 0, 10)], [square(10, 0, 10)]])
    }
  );
  expect(clean.tables[0].rowCount).toBe(0);
  expect(column(clean, 'status', 1)).toEqual(['valid', 'valid']);
});

test('voronoi cells tile the padded extent', () => {
  const result = run(
    'voronoi-diagram',
    {...planar, padding: 0.1},
    {
      points: points([
        [0, 0],
        [10, 0],
        [0, 10],
        [10, 10],
        [0, 0]
      ])
    }
  );
  expect(result.tables[0].rowCount).toBe(4);
  const total = (column(result, 'area') as number[]).reduce((sum, area) => sum + area, 0);
  expect(total).toBeCloseTo(144, 6);
  expect(result.diagnostics.map(item => item.code)).toContain('duplicate-points');
});

test('delaunay triangulation of a square with a center point', () => {
  const result = run('delaunay-triangulation', planar, {
    points: points([
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4],
      [2, 2]
    ])
  });
  expect(result.tables[0].rowCount).toBe(4);
  expect(result.tables[1].rowCount).toBe(8);
  expect((column(result, 'area') as number[]).reduce((sum, area) => sum + area, 0)).toBeCloseTo(
    16,
    9
  );
  const collinear = run('delaunay-triangulation', planar, {
    points: points([
      [0, 0],
      [1, 0],
      [2, 0]
    ])
  });
  expect(collinear.tables[0].rowCount).toBe(0);
});

test('concave hull trims a U shape and equals the convex hull at ratio 1', () => {
  const positions: number[][] = [];
  for (let step = 0; step <= 4; step++) positions.push([step, 0]);
  for (let step = 1; step <= 4; step++) positions.push([0, step], [4, step]);
  const input = {features: points(positions)};
  const convex = run('concave-hull', {...planar, ratio: 1}, input);
  expect(column(convex, 'area')[0]).toBeCloseTo(16, 9);
  const concave = run('concave-hull', {...planar, ratio: 0.2}, input);
  expect(column(concave, 'area')[0] as number).toBeLessThan(16);
  expect(column(concave, 'area')[0] as number).toBeGreaterThan(0);
  const grouped = run(
    'concave-hull',
    {...planar, ratio: 1, groupProperty: 'g'},
    {
      features: points(
        [
          [0, 0],
          [1, 0],
          [0, 1],
          [10, 10],
          [11, 10],
          [10, 11]
        ],
        [{g: 'a'}, {g: 'a'}, {g: 'a'}, {g: 'b'}, {g: 'b'}, {g: 'b'}]
      )
    }
  );
  expect(column(grouped, 'group')).toEqual(['a', 'b']);
  expect(column(grouped, 'area')[0]).toBeCloseTo(0.5, 9);
});

test('polygon triangulation covers a polygon with a hole', () => {
  const result = run('polygon-triangulation', planar, {
    polygons: polygons([[square(0, 0, 10), square(4, 4, 2).reverse()]])
  });
  expect(result.tables[0].rowCount).toBe(8);
  expect((column(result, 'area') as number[]).reduce((sum, area) => sum + area, 0)).toBeCloseTo(
    96,
    9
  );
  expect(column(result, 'source_id')[0]).toBe('f0');
});

test('polygon outline and polygonize round trip', () => {
  const outline = run('polygon-outline', planar, {
    polygons: polygons([[square(0, 0, 10), square(4, 4, 2)]])
  });
  expect(column(outline, 'is_hole')).toEqual([false, true]);
  const faces = run('geometry-polygonize', planar, {
    lines: lines([
      [
        [0, 0],
        [1, 0]
      ],
      [
        [1, 0],
        [1, 1]
      ],
      [
        [1, 1],
        [0, 1]
      ],
      [
        [0, 1],
        [0, 0]
      ],
      [
        [1, 1],
        [2, 2]
      ]
    ])
  });
  expect(faces.tables[0].rowCount).toBe(1);
  expect(column(faces, 'area')[0]).toBeCloseTo(1, 9);
  expect(faces.diagnostics.map(item => item.code)).toContain('dangles-removed');
  const crossing = run(
    'geometry-polygonize',
    {...planar, nodeCrossings: true},
    {
      lines: lines([
        [
          [0, 0],
          [4, 0]
        ],
        [
          [4, 0],
          [4, 4]
        ],
        [
          [4, 4],
          [0, 4]
        ],
        [
          [0, 4],
          [0, 0]
        ],
        [
          [2, -1],
          [2, 5]
        ]
      ])
    }
  );
  expect(crossing.tables[0].rowCount).toBe(2);
  expect((column(crossing, 'area') as number[]).reduce((sum, area) => sum + area, 0)).toBeCloseTo(
    16,
    9
  );
});

test('shape generator and dot density place shapes and counted dots', () => {
  const shapes = run(
    'geometry-shape-generator',
    {...planar, shape: 'square', radius: 1},
    {
      points: points([[5, 5]])
    }
  );
  expect(column(shapes, 'area')[0]).toBeCloseTo(2, 9);
  const hexagon = run(
    'geometry-shape-generator',
    {...planar, shape: 'hexagon', radius: 1},
    {
      points: points([[5, 5]], [{}])
    }
  );
  expect(column(hexagon, 'area')[0]).toBeCloseTo((3 * Math.sqrt(3)) / 2, 9);
  const geographic = run(
    'geometry-shape-generator',
    {shape: 'circle', radius: 1000},
    {
      points: points([[10, 50]])
    }
  );
  expect(column(geographic, 'area')[0] as number).toBeGreaterThan(3.0e6);
  const dots = run(
    'geometry-dot-density',
    {...planar, valueProperty: 'population', dotValue: 5},
    {
      polygons: polygons([[square(0, 0, 10)]], [{population: 22}])
    }
  );
  expect(dots.tables[0].rowCount).toBe(4);
  for (let row = 0; row < 4; row++) {
    const [x, y] = geometryAt(dots, row).coordinates;
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThanOrEqual(10);
    expect(y).toBeGreaterThanOrEqual(0);
    expect(y).toBeLessThanOrEqual(10);
  }
  const again = run(
    'geometry-dot-density',
    {...planar, valueProperty: 'population', dotValue: 5},
    {
      polygons: polygons([[square(0, 0, 10)]], [{population: 22}])
    }
  );
  expect(column(again, 'geometry')).toEqual(column(dots, 'geometry'));
  expect(() =>
    run(
      'geometry-dot-density',
      {...planar, valueProperty: 'population', dotValue: 1, maximumDots: 10},
      {
        polygons: polygons([[square(0, 0, 10)]], [{population: 22}])
      }
    )
  ).toThrow();
});

test('local-frame projection inverts exactly and keeps the origin', () => {
  const input = {
    features: points([
      [10, 50],
      [10.1, 50.1]
    ])
  };
  const forward = run(
    'geometry-local-frame-projection',
    {originLongitude: 10, originLatitude: 50, useDataCenter: false},
    input
  );
  expect(geometryAt(forward, 0).coordinates[0]).toBeCloseTo(0, 6);
  const north = geometryAt(forward, 1).coordinates[1];
  expect(north).toBeGreaterThan(11000);
  const back = run(
    'geometry-local-frame-projection',
    {direction: 'inverse', originLongitude: 10, originLatitude: 50},
    {features: collection([geometryAt(forward, 1)])}
  );
  expect(geometryAt(back, 0).coordinates[0]).toBeCloseTo(10.1, 7);
  expect(geometryAt(back, 0).coordinates[1]).toBeCloseTo(50.1, 7);
});

test('line split, clip, intersections and shared paths use planar topology', () => {
  const split = run(
    'line-split-at-points',
    {...planar, tolerance: 1},
    {
      lines: lines([
        [
          [0, 0],
          [10, 0]
        ]
      ]),
      splitters: points([
        [3, 0.1],
        [7, -0.1],
        [50, 50]
      ])
    }
  );
  expect(split.tables[0].rowCount).toBe(3);
  expect(
    (column(split, 'length') as number[]).reduce((sum, length) => sum + length, 0)
  ).toBeCloseTo(10, 9);
  expect(split.diagnostics.map(item => item.code)).toContain('splitter-unmatched');
  const clip = run(
    'line-clip-by-polygon',
    {...planar, keep: 'both'},
    {
      lines: lines([
        [
          [-5, 5],
          [15, 5]
        ]
      ]),
      polygons: polygons([[square(0, 0, 10)]])
    }
  );
  expect(column(clip, 'side')).toEqual(['outside', 'inside', 'outside']);
  expect(column(clip, 'length')[1]).toBeCloseTo(10, 9);
  const inside = run(
    'line-clip-by-polygon',
    {...planar, keep: 'inside'},
    {
      lines: lines([
        [
          [-5, 5],
          [15, 5]
        ]
      ]),
      polygons: polygons([[square(0, 0, 10)]])
    }
  );
  expect(inside.tables[0].rowCount).toBe(1);
  const crossings = run(
    'line-segment-intersections',
    {},
    {
      lines: lines([
        [
          [0, 0],
          [10, 10]
        ],
        [
          [0, 10],
          [10, 0]
        ],
        [
          [10, 10],
          [20, 10]
        ]
      ])
    }
  );
  expect(crossings.tables[0].rowCount).toBe(2);
  expect(column(crossings, 'kind').filter(kind => kind === 'crossing')).toHaveLength(1);
  const shared = run(
    'line-shared-paths',
    {...planar, tolerance: 0.5},
    {
      left: lines([
        [
          [0, 0],
          [10, 0]
        ]
      ]),
      right: lines([
        [
          [5, 0.1],
          [15, 0.1]
        ],
        [
          [10, 0],
          [4, 0]
        ]
      ])
    }
  );
  expect(column(shared, 'length')[0]).toBeCloseTo(5, 6);
  expect(column(shared, 'direction')).toEqual(['same', 'opposite']);
});

test('great-circle arcs, offset curves and loop removal', () => {
  const arcs = run(
    'line-great-circle-arcs',
    {segments: 8},
    {
      origins: points([
        [0, 0],
        [170, 10]
      ]),
      destinations: points([
        [90, 0],
        [-170, 10]
      ])
    }
  );
  expect(column(arcs, 'distance')[0]).toBeCloseTo((6371008.8 * Math.PI) / 2, 0);
  expect(geometryAt(arcs, 0).coordinates).toHaveLength(9);
  expect(geometryAt(arcs, 0).coordinates[4][0]).toBeCloseTo(45, 6);
  const wrapped = geometryAt(arcs, 1).coordinates;
  expect(wrapped[wrapped.length - 1][0]).toBeGreaterThan(180);
  const offset = run(
    'geometry-offset-curve',
    {...planar, distance: 2},
    {
      lines: lines([
        [
          [0, 0],
          [10, 0],
          [10, 10]
        ]
      ])
    }
  );
  const coordinates = geometryAt(offset, 0).coordinates;
  expect(coordinates[0]).toEqual([0, 2]);
  expect(coordinates[coordinates.length - 1][0]).toBeCloseTo(8, 9);
  const loops = run('line-loop-removal', planar, {
    lines: lines([
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [5, -5]
      ]
    ])
  });
  expect(column(loops, 'removed_loops')[0]).toBe(1);
  expect(geometryAt(loops, 0).coordinates).toHaveLength(3);
});

test('cleanup and precision reduction keep a row per feature', () => {
  const cleaned = run(
    'geometry-cleanup',
    {...planar, tolerance: 0.001},
    {
      features: collection([
        {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [0, 0],
              [10, 0],
              [10, 10],
              [0, 10],
              [0, 0]
            ]
          ]
        },
        {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [5, 0],
            [0, 0.0001]
          ]
        },
        {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [10, 0]
          ]
        }
      ])
    }
  );
  expect(column(cleaned, 'status')).toEqual(['cleaned', 'dropped', 'unchanged']);
  expect(column(cleaned, 'removed_vertices')[0]).toBe(1);
  const reduced = run(
    'polygon-precision-reduction',
    {gridSize: 1},
    {
      polygons: polygons([[square(0.2, 0.2, 5.1)], [square(0, 0, 0.3)]])
    }
  );
  expect(column(reduced, 'status')).toEqual(['cleaned', 'dropped']);
  expect(column(reduced, 'geometry')[1]).toBeNull();
  expect(geometryAt(reduced, 0).coordinates[0][0]).toEqual([0, 0]);
});

test('invalid input is rejected by the declared capacities and kinds', () => {
  const binding = getCatalogBinding('voronoi-diagram')!;
  expect(() =>
    runCatalogOperation(binding, planar, {
      points: lines([
        [
          [0, 0],
          [1, 1]
        ]
      ])
    })
  ).toThrow();
  expect(() =>
    runCatalogOperation(binding, {...planar, padding: -1}, {points: points([[0, 0]])})
  ).toThrow();
});

test('TSC-01 readonly predicate rings preserve hole interiors and boundary contact', () => {
  const shell = Object.freeze(square(0, 0, 4).map(position => Object.freeze(position)));
  const hole = Object.freeze(square(1, 1, 2).map(position => Object.freeze(position)));
  const right = collection([{type: 'Polygon', coordinates: Object.freeze([shell, hole])}]);
  const left = points([
    [0, 2],
    [1, 2],
    [2, 2],
    [0.5, 0.5]
  ]);
  // Both boundaries intersect; the open hole interior does not. No predicate mutates a ring.
  const intersecting = run('spatial-join-predicate', {predicate: 'intersects'}, {left, right});
  expect(column(intersecting, 'left_index')).toEqual([0, 1, 3]);
  const coveringHole = run(
    'spatial-join-predicate',
    {predicate: 'within'},
    {
      left: collection([{type: 'Polygon', coordinates: [shell]}]),
      right
    }
  );
  expect(coveringHole.tables[0].rowCount).toBe(0);
});
