// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {CatalogGpuSession, type CatalogGpuOutput} from '../src/analysis/catalog-gpu-session';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {prepareTraversedEdges} from '../src/analysis/catalog-network-categorical-flows';
import {readCatalogFeaturesFromNativeDataset} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

// These probes keep real contributor construction and host result assembly, but replace graph
// execution/readback. They check bindings and native ingestion without a GPU or shader oracle.
const DEVICE = {type: 'webgpu', isLost: false, limits: {maxBufferSize: 1 << 24}} as Device;
const CAPTURED_INPUTS = new Map<string, Float32Array | Uint32Array | Int32Array>();

for (const candidateOverflow of [0, 1]) {
  test(`edge-volume adapter retains incompleteness status ${candidateOverflow}`, async () => {
    installSessionProbe();
    vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockImplementation(async output =>
      output.length === 3 ? Uint32Array.of(2, 2, 0) : Uint32Array.of(candidateOverflow)
    );
    const edges: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [0, 1, 2].map(row => ({
        id: `edge-${row}`,
        geometry: {
          type: 'LineString',
          coordinates: [
            [row, 0],
            [row + 1, 0]
          ]
        }
      }))
    };
    const tracks: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          geometry: {
            type: 'LineString',
            coordinates: [
              [0, 0],
              [2, 0]
            ]
          },
          properties: {traversedEdges: [0, 1, null, 0, 1]}
        }
      ]
    };
    const result = await runCatalogOperationAsync(
      getCatalogBinding('map-matched-edge-volumes')!,
      {},
      {edges, tracks},
      {device: DEVICE}
    );
    expect(result.tables[0].columns['volume']).toEqual([2, 2, 0]);
    expect(result.summary['candidateOverflow']).toBe(candidateOverflow);
    expect(result.summary).not.toHaveProperty('truncated');
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  CAPTURED_INPUTS.clear();
});

function createProbeView<Format extends GPUVectorFormat>(
  session: CatalogGpuSession,
  name: string,
  format: Format,
  length: number
): GraphDataView<Format> {
  return session.graph.createDataView(
    session.graph.createTransientBuffer({
      id: name,
      usage: Buffer.STORAGE,
      byteLength: Math.max(1, length) * getGPUVectorFormatInfo(format).byteLength
    }),
    {format, length}
  );
}

function installSessionProbe(stopAtCosts = false) {
  vi.spyOn(CatalogGpuSession.prototype, 'input').mockImplementation(function <
    Format extends GPUVectorFormat
  >(
    this: CatalogGpuSession,
    name: string,
    values: Float32Array | Uint32Array | Int32Array,
    format: Format
  ): GraphDataView<Format> {
    CAPTURED_INPUTS.set(name, values.slice());
    if (stopAtCosts && name === 'costs') throw new Error('Captured affine costs');
    return createProbeView(
      this,
      name,
      format,
      (values.length * 4) / getGPUVectorFormatInfo(format).byteLength
    );
  });
  vi.spyOn(CatalogGpuSession.prototype, 'output').mockImplementation(function <
    Format extends GPUVectorFormat
  >(
    this: CatalogGpuSession,
    name: string,
    format: Format,
    length: number
  ): CatalogGpuOutput<Format> {
    return {
      view: createProbeView(this, name, format, length),
      buffer: {} as Buffer,
      length,
      components: getGPUVectorFormatInfo(format).byteLength / 4
    };
  });
  vi.spyOn(GPUCommandGraph.prototype, 'add').mockImplementation(() => {});
  vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(
    async output => new Float32Array(output.length * output.components)
  );
}

function makeNetwork(withJunction: boolean, slope = 1): CatalogFeatureCollection {
  const features: CatalogFeatureCollection['features'] = [
    {
      id: 'route',
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [1, 0],
          [2, 0]
        ]
      },
      properties: {cost: 2, slope, oneway: 'forward'}
    }
  ];
  if (withJunction)
    features.push({
      id: 'unused',
      geometry: {
        type: 'LineString',
        coordinates: [
          [1, 0],
          [1, 1]
        ]
      },
      properties: {cost: 1, slope: 0, oneway: 'forward'}
    });
  return {type: 'FeatureCollection', features};
}

const NETWORK_VALUES = {
  coordinateSystem: 'planar',
  costMode: 'property',
  forwardCostProperty: 'cost',
  slopeProperty: 'slope',
  onewayProperty: 'oneway',
  departureCount: 1,
  percentiles: [100]
};

async function captureNetworkCosts(lines: CatalogFeatureCollection, parameters = {}) {
  let error = '';
  CAPTURED_INPUTS.clear();
  try {
    await runCatalogOperationAsync(
      getCatalogBinding('network-travel-time-percentiles')!,
      {...NETWORK_VALUES, ...parameters},
      {lines},
      {device: DEVICE}
    );
  } catch (failure) {
    error = (failure as Error).message;
  }
  return {costs: Array.from(CAPTURED_INPUTS.get('costs') ?? []), error};
}

test('nonzero affine models reject shared-vertex splits before GPU allocation', async () => {
  installSessionProbe(true);
  const original = await captureNetworkCosts(makeNetwork(false));
  vi.mocked(CatalogGpuSession.prototype.output).mockClear();
  const split = await captureNetworkCosts(makeNetwork(true));
  expect(CatalogGpuSession.prototype.output).not.toHaveBeenCalled();
  expect(original.costs).toEqual([2, 1]);
  expect(split.error).toMatch(/nonzero.*slope.*split/i);
  expect(split.costs).toEqual([]);
  const unsplit = await captureNetworkCosts(makeNetwork(true), {splitAtSharedVertices: false});
  expect(unsplit.costs.slice(0, 2)).toEqual(original.costs);
});

test('zero-slope models keep endpoint arrival unchanged when a junction splits the line', async () => {
  installSessionProbe(true);
  const original = await captureNetworkCosts(makeNetwork(false, 0));
  const split = await captureNetworkCosts(makeNetwork(true, 0));
  expect(original.costs).toEqual([2, 0]);
  expect(split.costs).toEqual([1, 0, 1, 0, 1, 0]);
  expect(split.costs[0] + split.costs[2]).toBe(original.costs[0]);
});

for (const kind of ['array', 'typed-array', 'arrow-list'] as const) {
  test(`native matched tracks retain ${kind} traversal values and breaks`, () => {
    const values =
      kind === 'array'
        ? [0, 1, null, 0]
        : kind === 'typed-array'
          ? new Uint32Array([0, 1, 0xffffffff, 0])
          : arrow
              .vectorFromArray(
                [[0, 1, null, 0]],
                new arrow.List(new arrow.Field('item', new arrow.Int32(), true))
              )
              .get(0);
    const geometry = JSON.stringify({
      type: 'LineString',
      coordinates: [
        [0, 0],
        [1, 0]
      ]
    });
    const features = readCatalogFeaturesFromNativeDataset(
      {
        fields: [{name: 'geometry'}, {name: 'traversedEdges'}],
        dataContainer: {
          numRows: () => 1,
          valueAt: (_, column) => (column === 0 ? geometry : values)
        }
      },
      'geometry'
    );
    const packed = prepareTraversedEdges(features, 'traversedEdges', 2, 4);
    expect(Array.from(packed.edges)).toEqual([0, 1, 0xffffffff, 0]);
    expect(Array.from(packed.counts)).toEqual([4]);
  });
}

for (const id of ['kernel-flow-smoothing', 'radiation-flow-model']) {
  test(`${id} publishes flat high-precision arc endpoints from singleton MultiPoints`, async () => {
    installSessionProbe();
    const endpoints = [
      [0.123456789, 0.987654321, 12],
      [1.123456789, 2.987654321, 13]
    ];
    const zones: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: endpoints.map((position, index) => ({
        id: index,
        geometry: {
          type: index === 0 ? 'MultiPoint' : 'Point',
          coordinates: index === 0 ? [position] : position
        },
        properties: {population: 1, outflow: 1}
      }))
    };
    const flows: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          geometry: {type: 'LineString', coordinates: endpoints},
          properties: {origin: 0, destination: 1, weight: 1}
        }
      ]
    };
    const output = await runCatalogOperationAsync(
      getCatalogBinding(id)!,
      {coordinateSystem: 'planar'},
      {zones, flows},
      {device: DEVICE}
    );
    const geometry = JSON.parse(output.tables[0].columns['geometry'][1] as string);
    expect(geometry).toEqual({type: 'LineString', coordinates: endpoints});
    for (const position of geometry.coordinates)
      expect(position.every((component: unknown) => typeof component === 'number')).toBe(true);
  });
}

test('bidirectional arcs and distinct MultiLineString parts do not falsely count as splits', async () => {
  installSessionProbe(true);
  const bidirectional = await captureNetworkCosts(makeNetwork(false), {onewayProperty: ''});
  expect(bidirectional.costs).toEqual([2, 1, 2, 1]);
  const multipart = makeNetwork(false);
  multipart.features[0].geometry = {
    type: 'MultiLineString',
    coordinates: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [2, 0],
        [3, 0]
      ]
    ]
  };
  expect((await captureNetworkCosts(multipart)).costs).toEqual([2, 1, 2, 1]);
});

test('native Arrow lists retain nullable unsigned edge ids without coercing null to zero', () => {
  const table = new arrow.Table({
    geometry: arrow.vectorFromArray(
      [
        JSON.stringify({
          type: 'LineString',
          coordinates: [
            [0, 0],
            [1, 0]
          ]
        })
      ],
      new arrow.Utf8()
    ),
    traversedEdges: arrow.vectorFromArray(
      [[0n, 1n, null, 0n]],
      new arrow.List(new arrow.Field('item', new arrow.Uint64(), true))
    ),
    exactIds: arrow.vectorFromArray(
      [[9007199254740993n]],
      new arrow.List(new arrow.Field('item', new arrow.Uint64(), true))
    )
  });
  const features = readCatalogFeaturesFromNativeDataset(
    {
      fields: table.schema.fields,
      dataContainer: {
        numRows: () => table.numRows,
        valueAt: (row, column) => table.getChildAt(column)!.get(row)
      }
    },
    'geometry'
  );
  expect(features.features[0].properties?.['traversedEdges']).toEqual([0, 1, null, 0]);
  expect(features.features[0].properties?.['exactIds']).toEqual(['9007199254740993']);
  expect(Array.from(prepareTraversedEdges(features, 'traversedEdges', 2, 4).edges)).toEqual([
    0, 1, 0xffffffff, 0
  ]);
});
