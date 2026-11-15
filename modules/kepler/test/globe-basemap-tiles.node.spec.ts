// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {_GlobeViewport as GlobeViewport, WebMercatorViewport} from '@deck.gl/core';
import {MVTLayer, Tileset2D} from '@deck.gl/geo-layers';
import {GeoJsonLayer, TextLayer} from '@deck.gl/layers';
import {GlobeBasemapPolygonLayer} from '../../../examples/deck/kepler-compute/src/globe-basemap-polygon-layer';
import {expect, test, vi} from 'vitest';
import {VectorTileLiveController} from '../../../examples/deck/kepler-compute/src/vector-tile-live-controller';
import {
  GlobeBasemapTileLayer,
  renderGlobeBasemapTile
} from '../../../examples/deck/kepler-compute/src/globe-basemap-tiles';

test('globe basemap keeps native tile coordinates, accessors, clipping and country labels', () => {
  const data = [
    {
      type: 'Feature',
      properties: {layerName: 'water'},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0]
          ]
        ]
      }
    },
    {
      type: 'Feature',
      properties: {layerName: 'place', class: 'country', 'name:en': 'Austria'},
      geometry: {type: 'Point', coordinates: [0.54, 0.35]}
    },
    {
      type: 'Feature',
      properties: {layerName: 'place_label', class: 'country', name_en: 'Germany'},
      geometry: {
        type: 'MultiPoint',
        coordinates: [
          [0.52, 0.32],
          [0.53, 0.33]
        ]
      }
    },
    {
      type: 'Feature',
      properties: {layerName: 'place', class: 'city', name: 'Vienna'},
      geometry: {type: 'Point', coordinates: [0.54, 0.35]}
    }
  ];
  const getFillColor = vi.fn(() => [20, 40, 100, 255]);
  const getLineWidth = vi.fn(() => 0.75);
  const admitTile = vi.fn();
  const controller = new VectorTileLiveController({
    getReserveMemory: () => null,
    getCreateBuffer: () => null,
    residentRenderer: {admitTile} as never
  });
  const original = new MVTLayer({
    id: 'globe-basemap-mvt-layer-carto-compute-background',
    data: 'https://example.invalid/{z}/{x}/{y}.pbf',
    binary: false,
    getFillColor,
    getLineWidth,
    config: {labels: true, labelsColor: [255, 200, 100, 255]},
    parameters: {depthCompare: 'less-equal', depthWriteEnabled: false, cullMode: 'none'}
  } as never);
  const wrapped = controller.wrapLayer(original);
  // Exercise the actual public MVT render method: it installs the local CRS/matrix/ClipExtension
  // before calling the replacement renderer. Resident lng/lat geometry must never see this data.
  Object.assign(wrapped, {state: {binary: false}});
  const tileLayer = wrapped.renderSubLayers({
    ...wrapped.props,
    id: `${wrapped.id}-0-0-0`,
    data,
    tile: {index: {x: 0, y: 0, z: 0}},
    _offset: 0
  } as unknown as Parameters<MVTLayer['renderSubLayers']>[0]) as GlobeBasemapTileLayer;
  expect(tileLayer).toBeInstanceOf(GlobeBasemapTileLayer);
  const layers = tileLayer.renderLayers() as [GeoJsonLayer, TextLayer];
  expect(wrapped.props.renderSubLayers).toBe(renderGlobeBasemapTile);
  const [geometry, labels] = layers;
  expect(geometry).toBeInstanceOf(GeoJsonLayer);
  expect(geometry.props.data).toBe(data);
  expect(geometry.props.coordinateSystem).toBe('OSMTile512');
  expect(geometry.props.coordinateOrigin).toEqual([0, 0, 0]);
  expect(geometry.props.modelMatrix?.[0]).toBe(512);
  expect((geometry.props as {clipBounds?: number[]}).clipBounds).toEqual([0, 0, 1, 1]);
  expect(geometry.props.extensions).toHaveLength(1);
  expect(geometry.props.getFillColor).toBe(getFillColor);
  expect(geometry.props.getLineWidth).toBe(getLineWidth);
  expect(geometry.props.parameters?.depthCompare).toBe('less-equal');
  expect(geometry.props._subLayerProps?.['polygons-fill']?.type).toBe(GlobeBasemapPolygonLayer);
  expect(labels).toBeInstanceOf(TextLayer);
  expect(labels.props.data).toEqual([
    {position: [0.54, 0.35], text: 'Austria'},
    {position: [0.52, 0.32], text: 'Germany'},
    {position: [0.53, 0.33], text: 'Germany'}
  ]);
  expect(labels.props.coordinateSystem).toBe('OSMTile512');
  expect(labels.props.extensions).toEqual(geometry.props.extensions);
  expect(labels.props.parameters).toEqual({
    cullMode: 'none',
    depthCompare: 'always',
    depthWriteEnabled: false
  });
  expect(labels.props._subLayerProps).toBeUndefined();
  expect(controller.getLayerIds()).toEqual([]);
  expect(admitTile).not.toHaveBeenCalled();
});

test('native globe traversal requests z0 at the acceptance view and hides cached z9 tiles', async () => {
  const viewport = new GlobeViewport({
    width: 800,
    height: 600,
    longitude: 0,
    latitude: 20,
    zoom: 1.2,
    resolution: 72
  });
  const tileset = new Tileset2D({
    zoomOffset: -2,
    minZoom: 0,
    maxZoom: 14,
    maxCacheSize: 100,
    extent: [-180, -85.051129, 180, 85.051129],
    getTileData: () => []
  });
  try {
    // This is the initial planar view before switching to the acceptance script's globe view.
    tileset.update(
      new WebMercatorViewport({width: 800, height: 600, longitude: 0, latitude: 20, zoom: 11})
    );
    await vi.waitFor(() => expect(tileset.isLoaded).toBe(true));
    expect(tileset.selectedTiles?.every(tile => tile.index.z === 9)).toBe(true);
    tileset.update(viewport);
    await vi.waitFor(() => expect(tileset.isLoaded).toBe(true));
    tileset.update(viewport);
    expect(tileset.selectedTiles?.map(tile => tile.index)).toMatchObject([{x: 0, y: 0, z: 0}]);
    expect(tileset.tiles.some(tile => tile.index.z === 9)).toBe(true);
    expect(tileset.tiles.filter(tile => tile.isVisible).map(tile => tile.index)).toMatchObject([
      {x: 0, y: 0, z: 0}
    ]);
  } finally {
    tileset.finalize();
  }
});

test('offline globe MVT fixture selects and fetches its z0 tile below minZoom with basemap bounds', async () => {
  const viewport = new GlobeViewport({
    width: 256,
    height: 256,
    longitude: -30,
    latitude: 20,
    zoom: 1.2,
    resolution: 72
  });
  const options = {minZoom: 0, maxZoom: 0, zoomOffset: -2};
  const unbounded = new Tileset2D(options);
  const bounded = new Tileset2D({...options, extent: [-180, -85.051129, 180, 85.051129]});
  try {
    // The broken fixture was "loaded" with zero selected tiles: minZoom only clamps upward
    // when a source extent is present. Keep this exact small-viewport configuration covered.
    expect(unbounded.getTileIndices({viewport, minZoom: 0, maxZoom: 0, zRange: null})).toEqual([]);
    for (const [longitude, latitude] of [
      [-30, 20],
      [150, -20]
    ]) {
      const indices = bounded.getTileIndices({
        viewport: new GlobeViewport({...viewport, longitude, latitude}),
        minZoom: 0,
        maxZoom: 0,
        zRange: null
      });
      expect(indices).toHaveLength(1);
      expect(indices[0]).toMatchObject({x: 0, y: 0, z: 0});
    }
    // MVTLayer does honor fetch for the memory URL and unwraps the GeoJSON table. The fixture
    // therefore retains native MVT transforms/renderSubLayers without a worker or tile server.
    const features = [
      {type: 'Feature', geometry: {type: 'Point', coordinates: [0.5, 0.5]}, properties: {}}
    ];
    const fetchTile = vi.fn(async () => ({shape: 'geojson-table', features}));
    const layer = new MVTLayer({
      data: 'memory://water/{z}/{x}/{y}.mvt',
      binary: false,
      fetch: fetchTile
    });
    Object.assign(layer, {state: {data: layer.props.data, binary: false}});
    const tile = {index: {x: 0, y: 0, z: 0}, signal: new AbortController().signal};
    const content = await layer.getTileData(tile as Parameters<MVTLayer['getTileData']>[0]);
    expect(content).toBe(features);
    expect(fetchTile).toHaveBeenCalledWith(
      'memory://water/0/0/0.mvt',
      expect.objectContaining({
        loadOptions: expect.objectContaining({
          mvt: expect.objectContaining({coordinates: 'local', tileIndex: tile.index})
        })
      })
    );
  } finally {
    unbounded.finalize();
    bounded.finalize();
  }
});
