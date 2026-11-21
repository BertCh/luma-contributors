// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it, vi} from 'vitest';
import {
  getCatalogBinding,
  runCatalogOperation,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import {
  createCatalogFrame,
  packCatalogLines,
  readCatalogPolygons
} from '../src/analysis/catalog-gpu-pack';
import {runGpuPolygonTriangulation} from '../src/analysis/catalog-platform4-geometry-tessellation';
import * as sessions from '../src/analysis/catalog-gpu-session';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
const collection = (
  geometry: CatalogFeatureCollection['features'][number]['geometry']
): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features: [{geometry, properties: {}}]
});
it('B1 classifies a translated unit right triangle by stable area', async () => {
  const open = vi.spyOn(sessions, 'withCatalogGpuSession').mockResolvedValue({
    indices: new Uint32Array([0, 1, 2]),
    valid: new Uint32Array([1]),
    triangleCount: new Uint32Array([1]),
    triangleOffsets: new Uint32Array([0, 1])
  } as never);
  try {
    const result = await runGpuPolygonTriangulation(
      {device: {} as never},
      collection({
        type: 'Polygon',
        coordinates: [
          [
            [1e8, 1e8],
            [1e8 + 1, 1e8],
            [1e8, 1e8 + 1],
            [1e8, 1e8]
          ]
        ]
      }),
      'planar'
    );
    const table = result.tables[0];
    expect(table.columns['area']).toEqual([0.5]);
    expect(table.columns['degeneracy']).toEqual(['none']);
    expect(JSON.parse(table.columns['geometry'][0] as string)).toEqual({
      type: 'Polygon',
      coordinates: [
        [
          [1e8, 1e8],
          [1e8 + 1, 1e8],
          [1e8, 1e8 + 1],
          [1e8, 1e8]
        ]
      ]
    });
  } finally {
    open.mockRestore();
  }
});
it('B3 binding rejects raw malformed clip vertices and accepts consecutive duplicates', () => {
  const binding = getCatalogBinding('geometry-convex-clip')!;
  const features = collection({
    type: 'LineString',
    coordinates: [
      [-1, 1],
      [5, 1]
    ]
  });
  const clip = (ring: unknown) => collection({type: 'Polygon', coordinates: [ring]});
  expect(() =>
    runCatalogOperation(
      binding,
      {},
      {features, clip: clip([[0, 0], [3, 0], null, [3, 2], [0, 2], [0, 0]])}
    )
  ).toThrow(/geometry-convex-clip/);
  const result = runCatalogOperation(
    binding,
    {},
    {
      features,
      clip: clip([
        [0, 0],
        [3, 0],
        [3, 0],
        [3, 2],
        [0, 2],
        [0, 0]
      ])
    }
  );
  expect(JSON.parse(result.tables[0].columns['geometry'][0] as string).coordinates).toEqual([
    [0, 1],
    [3, 1]
  ]);
});
it('B4 chooses a common origin preserving both admitted line parts', () => {
  const input = collection({
    type: 'MultiLineString',
    coordinates: [
      [
        [0, 0],
        [0.25, 0]
      ],
      [
        [20000000, 0],
        [20000002, 0]
      ]
    ]
  });
  const frame = createCatalogFrame('planar', [input]);
  const packed = packCatalogLines(input.features, {project: frame.project});
  expect(
    Array.from(packed.positions)
      .filter((_, index) => index % 2 === 0)
      .map(x => frame.unproject(x, 0)[0])
  ).toEqual([0, 0.25, 20000000, 20000002]);
});
it('B7 suppresses collinear shells without dropping a later valid polygon', () => {
  expect(
    readCatalogPolygons(
      [0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 1, 3, 3, 4, 3, 3, 4],
      [0, 3, 6, 9],
      [0, 2, 3],
      [0, 2]
    )
  ).toEqual([
    [
      [
        [3, 3],
        [4, 3],
        [3, 4],
        [3, 3]
      ]
    ]
  ]);
});
it('S1 rejects offsets outside storage and reinterprets byte-backed coordinates', () => {
  expect(() => readCatalogPolygons([0, 0, 1, 1], [0, 3], [0, 1], [0, 1])).toThrow(/offset/);
  const coordinates = new Float32Array([0, 0, 1, 0, 0, 1]);
  expect(
    readCatalogPolygons(
      new Uint8Array(coordinates.buffer),
      new Uint8Array(new Uint32Array([0, 3]).buffer),
      new Uint8Array(new Uint32Array([0, 1]).buffer),
      [0, 1]
    )
  ).toEqual([
    [
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [0, 0]
      ]
    ]
  ]);
});
it('B8 executable bindings prepend large lines and collect a large multipoint', () => {
  const vertices = Array.from({length: 150000}, (_, index) => [-150000 + index, 0]);
  const lines = collection({
    type: 'MultiLineString',
    coordinates: [
      [
        [0, 0],
        [1, 0]
      ],
      [...vertices, [0, 0]]
    ]
  });
  const merged = runCatalogOperation(
    getCatalogBinding('line-merge')!,
    {coordinateSystem: 'planar'},
    {lines}
  );
  const coordinates = JSON.parse(merged.tables[0].columns['geometry'][0] as string).coordinates;
  expect(coordinates[0]).toEqual([-150000, 0]);
  expect(coordinates.at(-1)).toEqual([1, 0]);
  const points = collection({type: 'MultiPoint', coordinates: [...vertices, [0, 1]]});
  const hull = runCatalogOperation(
    getCatalogBinding('group-convex-hull')!,
    {coordinateSystem: 'planar'},
    {points}
  );
  expect(JSON.parse(hull.tables[0].columns['geometry'][0] as string).type).toBe('Polygon');
});
it('earlier-fix note rejects malformed raw vertices before packing', () => {
  expect(() =>
    packCatalogLines(
      collection({
        type: 'LineString',
        coordinates: [
          [0, 0],
          ['broken', 5],
          [2, 4]
        ]
      }).features
    )
  ).toThrow(/coordinate/);
});
import {runGpuRectangleClip} from '../src/analysis/catalog-gpu-geometry';
it('B9 refuses a rectangle whose raw shortcut loses its unit width', async () => {
  const ring = [
    [20000000, 0],
    [20000001, 0],
    [20000001, 1],
    [20000000, 1],
    [20000000, 0]
  ];
  await expect(
    runGpuRectangleClip(
      {device: {} as never},
      collection({
        type: 'LineString',
        coordinates: [
          [19999999, 0.5],
          [20000002, 0.5]
        ]
      }),
      collection({type: 'Polygon', coordinates: [ring]})
    )
  ).rejects.toThrow(/float32/);
});
it('B1 CPU fallback publishes stable translated triangle area and a closed Polygon', () => {
  const result = runCatalogOperation(
    getCatalogBinding('polygon-triangulation')!,
    {coordinateSystem: 'planar'},
    {
      polygons: collection({
        type: 'Polygon',
        coordinates: [
          [
            [1e8, 1e8],
            [1e8 + 1, 1e8],
            [1e8, 1e8 + 1],
            [1e8, 1e8]
          ]
        ]
      })
    }
  );
  expect(result.tables[0].columns['degeneracy']).toEqual(['none']);
  expect(result.tables[0].columns['area']).toEqual([0.5]);
  expect(JSON.parse(result.tables[0].columns['geometry'][0] as string).type).toBe('Polygon');
});
import {triangulateMonotoneOnCPU} from '../../analysis/src/analysis-core/polygon-triangulation/cpu-monotone-triangulation';
it('B2 serializer preserves counted monotone rows, both positive coordinates and degeneracies', async () => {
  const rings: [number, number][][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ],
    [
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2]
    ],
    [
      [4, 4],
      [4, 6],
      [6, 6],
      [6, 4]
    ]
  ];
  const triangles = triangulateMonotoneOnCPU(rings).triangles;
  const open = vi.spyOn(sessions, 'withCatalogGpuSession').mockResolvedValue({
    indices: Uint32Array.from(triangles.flat()),
    valid: new Uint32Array([1]),
    triangleCount: new Uint32Array([14]),
    triangleOffsets: new Uint32Array([0, 14])
  } as never);
  try {
    const output = await runGpuPolygonTriangulation(
      {device: {} as never},
      collection({type: 'Polygon', coordinates: rings.map(ring => [...ring, ring[0]])}),
      'planar'
    );
    const table = output.tables[0];
    expect(table.rowCount).toBe(14);
    expect(table.columns['area'].reduce<number>((sum, value) => sum + (value as number), 0)).toBe(
      92
    );
    for (let row = 0; row < 14; row++) {
      const geometry = JSON.parse(table.columns['geometry'][row] as string);
      const corners = triangles[row].map(vertex => rings.flat()[vertex]);
      expect(table.columns['triangle_index'][row]).toBe(row);
      expect(table.columns['triangle_id'][row]).toBe(`0:0:${row}`);
      if (row === 11 || row === 12) {
        expect(table.columns['area'][row]).toBe(0);
        expect(table.columns['degeneracy'][row]).toBe('line');
        expect(geometry.type).toBe('LineString');
        expect(geometry.coordinates).toEqual(
          corners.filter(
            (point, index) =>
              corners.findIndex(other => point[0] === other[0] && point[1] === other[1]) === index
          )
        );
      } else {
        expect(table.columns['degeneracy'][row]).toBe('none');
        expect(geometry).toEqual({type: 'Polygon', coordinates: [[...corners, corners[0]]]});
      }
    }
  } finally {
    open.mockRestore();
  }
});
import * as lineGeometry from '../src/analysis/catalog-platform4-geometry-common';
import {runGpuLineChunk} from '../src/analysis/catalog-platform4-geometry-lines';
it.each([
  0.25, 0.75
])('B1 line runner keeps measures consistent with geometry for edge %s', async firstLength => {
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _name, callback) => callback({} as never));
  const run = vi
    .spyOn(lineGeometry, 'runCatalogLineGeometry')
    .mockImplementation(async (_session, packed) => ({
      ...packed,
      lineCount: 2,
      sourceLineIndices: new Uint32Array([0, 1])
    }));
  try {
    const result = await runGpuLineChunk(
      {device: {} as never},
      collection({
        type: 'MultiLineString',
        coordinates: [
          [
            [0, 0],
            [firstLength, 0]
          ],
          [
            [20000000, 0],
            [20000002, 0]
          ]
        ]
      }),
      {coordinateSystem: 'planar', maximumChunkLength: 3},
      10
    );
    expect(result.tables[0].columns['length']).toEqual([firstLength, 2]);
    expect(
      result.tables[0].columns['geometry'].map(text => JSON.parse(text as string).coordinates)
    ).toEqual([
      [
        [0, 0],
        [firstLength, 0]
      ],
      [
        [20000000, 0],
        [20000002, 0]
      ]
    ]);
  } finally {
    run.mockRestore();
    open.mockRestore();
  }
});
it('B2 ordinary contributor default route emits twelve positive triangles for touching holes', async () => {
  const rings: [number, number][][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ],
    [
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2]
    ],
    [
      [4, 4],
      [4, 6],
      [6, 6],
      [6, 4]
    ]
  ];
  const {computePolygonTriangulationOnCPU} = await import(
    '../../analysis/src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation'
  );
  const result = computePolygonTriangulationOnCPU(
    {
      positions: Float32Array.from(rings.flat().flatMap(([x, y]) => [x - 5, y - 5])),
      featureOffsets: new Uint32Array([0, 1]),
      polygonOffsets: new Uint32Array([0, 3]),
      ringOffsets: new Uint32Array([0, 4, 8, 12])
    },
    {indexCapacity: 48}
  );
  expect(Array.from(result.triangleCount)).toEqual([12]);
  const open = vi.spyOn(sessions, 'withCatalogGpuSession').mockResolvedValue(result as never);
  try {
    const output = await runGpuPolygonTriangulation(
      {device: {} as never},
      collection({type: 'Polygon', coordinates: rings.map(ring => [...ring, ring[0]])}),
      'planar'
    );
    const table = output.tables[0];
    expect(table.rowCount).toBe(12);
    let area = 0;
    for (let row = 0; row < 12; row++) {
      const value = table.columns['area'][row] as number;
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThan(0);
      area += value;
      const geometry = JSON.parse(table.columns['geometry'][row] as string);
      expect(geometry.type).toBe('Polygon');
      expect(table.columns['degeneracy'][row]).toBe('none');
      const corners = Array.from(
        result.indices.subarray(row * 3, row * 3 + 3),
        vertex => rings.flat()[vertex]
      );
      expect(geometry.coordinates).toEqual([[...corners, corners[0]]]);
    }
    expect(area).toBe(92);
  } finally {
    open.mockRestore();
  }
});

import {serializeCatalogTriangle} from '../src/analysis/catalog-triangle-output';
it('GEO-124 shared publication classifies literal translated, line and point rows', () => {
  // Hand worked: base 1 * height 1 / 2; collinear and coincident triples have zero area.
  const corners: [number, number][][] = [
    [
      [1e8, 1e8],
      [1e8 + 1, 1e8],
      [1e8, 1e8 + 1]
    ],
    [
      [0, 0],
      [1, 0],
      [2, 0]
    ],
    [
      [3, 7],
      [3, 7],
      [3, 7]
    ]
  ];
  corners.forEach((row, index) => {
    const result = serializeCatalogTriangle(row, (x, y) => [x, y]);
    expect(result.area).toBe([0.5, 0, 0][index]);
    expect(result.degeneracy).toBe(['none', 'line', 'point'][index]);
    expect(JSON.parse(result.geometry).type).toBe(['Polygon', 'LineString', 'Point'][index]);
  });
});

import {
  nodeCatalogLines,
  runGpuNetworkIsochrones,
  type CatalogIsochroneParameters
} from '../src/analysis/catalog-gpu-network';
import * as snapping from '../src/analysis/catalog-gpu-snapping';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';

it.each([
  0.25, 0.75
])('B1 S2 actual noding and isochrone uploads preserve complete XY for edge %s', async firstLength => {
  const device = createNullWebGPUDevice();
  const input = collection({
    type: 'MultiLineString',
    coordinates: [
      [
        [0, 0],
        [firstLength, 0]
      ],
      [
        [20000000, 0],
        [20000002, 0]
      ]
    ]
  });
  const reachedUpload = new Error('upload captured before GPU execution');
  let uploaded: Float32Array | undefined;
  let target = 'positions';
  const createBuffer = device.createBuffer.bind(device);
  const upload = vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    if (
      'id' in props &&
      props.id === `network-upload-${target}` &&
      'data' in props &&
      props.data instanceof Float32Array
    ) {
      uploaded = new Float32Array(props.data);
      throw reachedUpload;
    }
    return createBuffer(props);
  });
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _name, callback) => {
      const session = new sessions.CatalogGpuSession(device, 'network-upload');
      try {
        return await callback(session);
      } finally {
        await session.dispose();
      }
    });
  const snap = vi.spyOn(snapping, 'snapCatalogPoints').mockResolvedValue({
    snapped: [{edge: 0, fraction: 0, position: [0, 0], distance: 0}],
    engine: 'cpu-reference'
  });
  const parameters: CatalogIsochroneParameters = {
    coordinateSystem: 'planar',
    snapTolerance: 0,
    splitAtSharedVertices: false,
    costMode: 'length',
    forwardCostProperty: '',
    reverseCostProperty: '',
    onewayProperty: '',
    crossings: 'not-noded',
    maximumSnapDistance: 1,
    budgets: [1],
    bufferRadius: 1,
    rasterSize: 8
  };
  const assertUpload = (withOrigin: boolean) => {
    expect(uploaded).toBeDefined();
    // Independent endpoint differences are firstLength and 2; the snapped origin is [0,0].
    const positions = uploaded!;
    expect(Array.from(positions)).toEqual([
      0,
      0,
      firstLength,
      0,
      20000000,
      0,
      20000002,
      0,
      ...(withOrigin ? [0, 0] : [])
    ]);
    for (const [start, expected] of [
      [0, firstLength],
      [4, 2]
    ]) {
      const length = Math.hypot(
        positions[start + 2] - positions[start],
        positions[start + 3] - positions[start + 1]
      );
      expect(Number.isFinite(length)).toBe(true);
      expect(length).toBe(expected);
    }
  };
  try {
    await expect(nodeCatalogLines({device}, input, parameters)).rejects.toBe(reachedUpload);
    assertUpload(false);
    uploaded = undefined;
    target = 'node-positions';
    await expect(
      runGpuNetworkIsochrones(
        {device},
        input,
        collection({type: 'Point', coordinates: [0, 0]}),
        parameters,
        {maximumEdges: 10}
      )
    ).rejects.toBe(reachedUpload);
    assertUpload(true);
    // None of center, identity or minimum can represent both pairs in this input.
    const unsupported = collection({
      type: 'MultiLineString',
      coordinates: [
        [
          [0, 0],
          [0.25, 0]
        ],
        [
          [20000000, 0],
          [20000001, 0]
        ]
      ]
    });
    await expect(nodeCatalogLines({device}, unsupported, parameters)).rejects.toBeInstanceOf(
      sessions.CatalogGpuUnsupportedError
    );
    await expect(
      runGpuNetworkIsochrones(
        {device},
        unsupported,
        collection({type: 'Point', coordinates: [0, 0]}),
        parameters,
        {maximumEdges: 10}
      )
    ).rejects.toBeInstanceOf(sessions.CatalogGpuUnsupportedError);
  } finally {
    snap.mockRestore();
    open.mockRestore();
    upload.mockRestore();
    device.destroy();
  }
});

it('GEO-124 GPU publication retains every counted line and point row and identity', async () => {
  const open = vi.spyOn(sessions, 'withCatalogGpuSession').mockResolvedValue({
    indices: Uint32Array.of(0, 1, 2, 0, 1, 1, 0, 0, 0),
    valid: Uint32Array.of(1),
    triangleCount: Uint32Array.of(3),
    triangleOffsets: Uint32Array.of(0, 3)
  } as never);
  try {
    const input = collection({
      type: 'Polygon',
      coordinates: [
        [
          [1e8, 1e8],
          [1e8 + 1, 1e8],
          [1e8, 1e8 + 1],
          [1e8 - 1, 1e8 + 1],
          [1e8 - 1, 1e8],
          [1e8, 1e8]
        ]
      ]
    });
    // Five open vertices allow three triangles / nine indices in the production allocation.
    expect(
      getPolygonTriangulationIndexCount(
        getPolygonParts(input.features[0].geometry)[0][0].length - 1,
        1,
        1
      )
    ).toBe(9);
    input.features[0].id = 'original';
    const result = await runGpuPolygonTriangulation({device: {} as never}, input, 'planar');
    const table = result.tables[0];
    expect(table.rowCount).toBe(3);
    expect(table.columns['area']).toEqual([0.5, 0, 0]);
    expect(table.columns['degeneracy']).toEqual(['none', 'line', 'point']);
    expect(table.columns['triangle_id']).toEqual(['0:0:0', '0:0:1', '0:0:2']);
    expect(table.columns['triangle_index']).toEqual([0, 1, 2]);
    expect(table.columns['source_id']).toEqual(['original', 'original', 'original']);
    expect(table.columns['geometry'].map(text => JSON.parse(text as string).type)).toEqual([
      'Polygon',
      'LineString',
      'Point'
    ]);
  } finally {
    open.mockRestore();
  }
});

it('B7 CPU triangle publication preserves the original nonempty source part index', () => {
  const result = runCatalogOperation(
    getCatalogBinding('polygon-triangulation')!,
    {coordinateSystem: 'planar'},
    {
      polygons: collection({
        type: 'MultiPolygon',
        coordinates: [
          [],
          [
            [
              [0, 0],
              [1, 0],
              [0, 1],
              [0, 0]
            ]
          ]
        ]
      })
    }
  );
  expect(result.tables[0].columns['part_index']).toEqual([1]);
  expect(result.tables[0].columns['triangle_id']).toEqual(['0:1:0']);
  expect(result.tables[0].columns['area']).toEqual([0.5]);
});

import {getPolygonTriangulationIndexCount} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  appendCatalogTriangleRows,
  CATALOG_POLYGON_TRIANGLE_COLUMNS
} from '../src/analysis/catalog-triangle-output';
import {makeCatalogTable, getPolygonParts} from '../src/analysis/catalog-geojson';
it('S1 CPU row publisher retains counted line and point triples with original identities', () => {
  const table = makeCatalogTable(
    'polygon-triangles',
    'Polygon triangles',
    CATALOG_POLYGON_TRIANGLE_COLUMNS,
    'geometry'
  );
  // Literal counted triples: unit right triangle area 1/2, collinear line, coincident point.
  const triangles: [number, number][][] = [
    [
      [1e8, 1e8],
      [1e8 + 1, 1e8],
      [1e8, 1e8 + 1]
    ],
    [
      [0, 0],
      [1, 0],
      [2, 0]
    ],
    [
      [3, 7],
      [3, 7],
      [3, 7]
    ]
  ];
  appendCatalogTriangleRows(table, triangles, {id: 'original', geometry: null}, 4, 2, (x, y) => [
    x,
    y
  ]);
  expect(table.rowCount).toBe(3);
  expect(table.columns['area']).toEqual([0.5, 0, 0]);
  expect(table.columns['degeneracy']).toEqual(['none', 'line', 'point']);
  expect(table.columns['triangle_id']).toEqual(['4:2:0', '4:2:1', '4:2:2']);
  expect(table.columns['source_id']).toEqual(['original', 'original', 'original']);
  expect(table.columns['source_index']).toEqual([4, 4, 4]);
  expect(table.columns['part_index']).toEqual([2, 2, 2]);
  expect(table.columns['geometry'].map(text => JSON.parse(text as string).type)).toEqual([
    'Polygon',
    'LineString',
    'Point'
  ]);
});

it('S3 GPU-only public network bindings propagate unsupported frames', async () => {
  const lines = collection({
    type: 'MultiLineString',
    coordinates: [
      [
        [0, 0],
        [0.25, 0]
      ],
      [
        [20000000, 0],
        [20000001, 0]
      ]
    ]
  });
  for (const id of ['network-crossing-noding', 'network-isochrone-polygons']) {
    const binding = getCatalogBinding(id)!;
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    const inputs: Record<string, CatalogFeatureCollection> = {lines};
    if (id === 'network-isochrone-polygons')
      inputs['origins'] = collection({type: 'Point', coordinates: [0, 0]});
    await expect(
      runCatalogOperationAsync(binding, {coordinateSystem: 'planar'}, inputs, {device: {} as never})
    ).rejects.toBeInstanceOf(sessions.CatalogGpuUnsupportedError);
  }
});
