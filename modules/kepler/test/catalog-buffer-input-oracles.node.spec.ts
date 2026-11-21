// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {
  runGpuGeometryValidity,
  runGpuBuffer,
  type CatalogBufferParameters
} from '../src/analysis/catalog-gpu-geometry';
import {packCatalogLines, packCatalogPolygons} from '../src/analysis/catalog-gpu-pack';
import * as sessions from '../src/analysis/catalog-gpu-session';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';

const parameters: CatalogBufferParameters = {
  coordinateSystem: 'planar',
  distance: 1,
  quadSegments: 8,
  joinStyle: 'mitre',
  capStyle: 'flat',
  mitreLimit: 5,
  dissolve: true
};
const collection = (first: number): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features: [
    {
      geometry: {
        type: 'MultiLineString',
        coordinates: [
          [[first, 0]],
          [
            [0, 0],
            [4, 0]
          ]
        ]
      },
      properties: {}
    }
  ]
});

it('rejects nonfinite projected float32 input before opening a retry workspace', async () => {
  const device = createNullWebGPUDevice();
  const open = vi.spyOn(sessions, 'withCatalogGpuSession').mockResolvedValue({
    polygons: [],
    complete: false,
    requiredRings: 1,
    requiredBoundary: 4
  } as never);
  try {
    for (const invalid of [Number.NaN, Infinity, 1e100]) {
      await expect(runGpuBuffer({device}, 'line', collection(invalid), parameters)).rejects.toThrow(
        /input.*finite.*float32/i
      );
      expect(open).not.toHaveBeenCalled();
      const polygons: CatalogFeatureCollection = {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [invalid, 0],
                  [1, 0],
                  [0, 1]
                ]
              ]
            },
            properties: {}
          }
        ]
      };
      await expect(runGpuBuffer({device}, 'polygon', polygons, parameters)).rejects.toThrow(
        /input.*finite.*float32/i
      );
      expect(open).not.toHaveBeenCalled();
    }
    const polygon = [
      {
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [1, 0],
              [0, 1]
            ]
          ]
        },
        properties: {}
      }
    ];
    expect(() =>
      packCatalogPolygons(polygon, {closing: 'drop', project: () => [Infinity, 0]})
    ).toThrow(/input.*finite.*float32/i);
    expect(() => packCatalogLines(collection(0).features, {project: () => [1e100, 0]})).toThrow(
      /input.*finite.*float32/i
    );
  } finally {
    open.mockRestore();
    device.destroy();
  }
});

it('retries a genuine capacity result and stops when the arrangement fits', async () => {
  const device = createNullWebGPUDevice();
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockResolvedValueOnce({
      polygons: [],
      complete: false,
      requiredRings: 65,
      requiredBoundary: 1025
    } as never)
    .mockResolvedValueOnce({
      polygons: [],
      complete: true,
      requiredRings: 0,
      requiredBoundary: 0
    } as never);
  try {
    const result = await runGpuBuffer({device}, 'line', collection(0), parameters);
    expect(open).toHaveBeenCalledTimes(2);
    expect(result.tables[0].columns['geometry']).toEqual([]);
  } finally {
    open.mockRestore();
    device.destroy();
  }
});

// The first feature is usable: the second must be diagnosed before any workspace opens.
it('R11 rejects a bad later buffer feature before tolerance on either dissolve route', async () => {
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockRejectedValue(new Error('workspace opened'));
  try {
    for (const dissolve of [true, false])
      for (const invalid of [NaN, 1e100]) {
        const input: CatalogFeatureCollection = {
          type: 'FeatureCollection',
          features: [
            {
              properties: {},
              geometry: {
                type: 'LineString',
                coordinates: [
                  [0, 0],
                  [4, 0]
                ]
              }
            },
            {
              properties: {},
              geometry: {
                type: 'LineString',
                coordinates: [
                  [invalid, 0],
                  [5, 0]
                ]
              }
            }
          ]
        };
        await expect(
          runGpuBuffer({device: {} as never}, 'line', input, {...parameters, dissolve})
        ).rejects.toThrow(/input.*finite.*float32/i);
        expect(open).not.toHaveBeenCalled();
      }
  } finally {
    open.mockRestore();
  }
});

it('R11 validity keeps a valid first polygon and the bad later source diagnostic', async () => {
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockResolvedValue({mask: Uint32Array.of(0), overflow: 0} as never);
  try {
    for (const invalid of [NaN, 1e100]) {
      const input: CatalogFeatureCollection = {
        type: 'FeatureCollection',
        features: [
          {
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [1, 0],
                  [0, 1],
                  [0, 0]
                ]
              ]
            }
          },
          {
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [invalid, 0],
                  [1, 0],
                  [0, 1],
                  [invalid, 0]
                ]
              ]
            }
          }
        ]
      };
      const result = await runGpuGeometryValidity({device: {} as never}, input);
      expect(result.tables[0].columns['source_index']).toEqual([0, 1]);
      expect(result.tables[0].columns['status']).toEqual(['valid', 'invalid']);
      expect(result.tables[0].columns['issues']).toEqual(['', 'non-finite-coordinate']);
      const reversed = await runGpuGeometryValidity(
        {device: {} as never},
        {...input, features: [...input.features].reverse()}
      );
      expect(reversed.tables[0].columns['source_index']).toEqual([0, 1]);
      expect(reversed.tables[0].columns['status']).toEqual(['invalid', 'valid']);
      const allInvalid = await runGpuGeometryValidity(
        {device: {} as never},
        {...input, features: [input.features[1]]}
      );
      expect(allInvalid.tables[0].columns['issues']).toEqual(['non-finite-coordinate']);
    }
  } finally {
    open.mockRestore();
  }
});

it('B3 all-invalid validity reports host engine and no GPU dispatch', async () => {
  const input: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 3],
              [2 ** 140, 3],
              [0, 7],
              [0, 3]
            ]
          ]
        }
      }
    ]
  };
  const output = await runGpuGeometryValidity({device: {} as never}, input);
  expect(output).toHaveProperty('engine', 'cpu-reference');
  expect(output.dispatchedContributors).toEqual([]);
});

it('S4 buffer explicitly diagnoses its raw-coordinate restriction before projection', async () => {
  const input: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: [
            [1e40, 3],
            [1e40, 7]
          ]
        }
      }
    ]
  };
  const {createCatalogFrame} = await import('../src/analysis/catalog-gpu-pack');
  const frame = createCatalogFrame('planar', [input]);
  expect(Array.from(packCatalogLines(input.features, {project: frame.project}).positions)).toEqual([
    0, -2, 0, 2
  ]);
  await expect(runGpuBuffer({device: {} as never}, 'line', input, parameters)).rejects.toThrow(
    /source.*finite float32.*before projection/
  );
});

it('N1 validity upload preserves an open translated ring endpoint gap', async () => {
  const device = createNullWebGPUDevice();
  const captured = new Error('validity upload captured');
  let positions: Float32Array | undefined;
  const createBuffer = device.createBuffer.bind(device);
  const upload = vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    if (
      'id' in props &&
      props.id === 'validity-gap-positions' &&
      'data' in props &&
      props.data instanceof Float32Array
    ) {
      positions = new Float32Array(props.data);
      throw captured;
    }
    return createBuffer(props);
  });
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _name, callback) => {
      const session = new sessions.CatalogGpuSession(device, 'validity-gap');
      try {
        return await callback(session);
      } finally {
        await session.dispose();
      }
    });
  try {
    await expect(
      runGpuGeometryValidity(
        {device},
        {
          type: 'FeatureCollection',
          features: [
            {
              properties: {},
              geometry: {
                type: 'Polygon',
                coordinates: [
                  [
                    [20000000, 0],
                    [20000002, 0],
                    [20000002, 2],
                    [20000000, 2],
                    [20000000.25, 0]
                  ]
                ]
              }
            }
          ]
        }
      )
    ).rejects.toBe(captured);
    expect(positions).toBeDefined();
    expect(Array.from(positions!)).toEqual([-1, -1, 1, -1, 1, 1, -1, 1, -0.75, -1]);
    expect(positions![8] - positions![0]).toBe(0.25);
  } finally {
    open.mockRestore();
    upload.mockRestore();
    device.destroy();
  }
});

import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
// Regression: public CPU and GPU routes share raw-coordinate admission.
it('B2 public validity CPU and device routes agree on finite float64 overflow', async () => {
  const input: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [0, 3],
              [2 ** 140, 3],
              [0, 7],
              [0, 3]
            ]
          ]
        }
      }
    ]
  };
  const binding = getCatalogBinding('geometry-validity')!;
  for (const options of [{engine: 'cpu' as const}, {}, {device: {} as never}]) {
    const result = await runCatalogOperationAsync(binding, {}, {polygons: input}, options);
    expect(result.tables[0].columns['status']).toEqual(['invalid']);
    expect(result.tables[0].columns['issues']).toEqual(['non-finite-coordinate']);
  }
});
// Regression: public provenance follows the runner when no GPU work is needed.
it('B3 public all-invalid validity identifies host engine with empty dispatch', async () => {
  const result = await runCatalogOperationAsync(
    getCatalogBinding('geometry-validity')!,
    {},
    {
      polygons: {
        type: 'FeatureCollection',
        features: [
          {
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [NaN, 0],
                  [1, 0],
                  [0, 1],
                  [NaN, 0]
                ]
              ]
            }
          }
        ]
      }
    },
    {device: {} as never}
  );
  expect(result.engine).toBe('cpu-reference');
  expect(result.dispatchedContributors).toEqual([]);
});

import {getCatalogFloat32CoordinateDiagnostic} from '../src/analysis/catalog-geojson';
it('B2 shared validity source diagnostic catches finite float64 overflow', () => {
  expect(
    getCatalogFloat32CoordinateDiagnostic({
      type: 'Polygon',
      coordinates: [
        [
          [0, 3],
          [2 ** 140, 3],
          [0, 7],
          [0, 3]
        ]
      ]
    })
  ).toBe('non-finite-coordinate');
  expect(
    getCatalogFloat32CoordinateDiagnostic({
      type: 'Polygon',
      coordinates: [
        [
          [0, 3],
          [2, 3],
          [0, 7],
          [0, 3]
        ]
      ]
    })
  ).toBeUndefined();
});
