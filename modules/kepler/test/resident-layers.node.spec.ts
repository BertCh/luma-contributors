// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {LayerClasses} from '@kepler.gl/layers';
import {GPUVector, type GPUVectorFormat} from '@luma-contributors/graph/gpu-data';
import type {Buffer} from '@luma.gl/core';
import type {PickingInfo} from '@deck.gl/core';
import {GPUScatterplotLayer} from '@deck.gl-community/gpu-layers';
import {
  createResidentLayerClasses,
  type ResidentDatasetBinding
} from '../../../examples/deck/kepler-compute/src/resident-layers';
import {
  createPreparedLayerClasses,
  PREPARE_DATA_MESSAGE
} from '../../../examples/deck/kepler-compute/src/prepared-layer-classes';

// Native Kepler's aggregate package eagerly loads WebGL-only dependencies. Keep the Node
// boundary test independent of that browser bootstrap; host smoke covers actual registration.
vi.mock('@kepler.gl/utils', () => ({
  DataRow: class {
    constructor(
      readonly _dataContainer: unknown,
      readonly _rowIndex: number
    ) {}
  }
}));

vi.mock('@kepler.gl/layers', () => {
  class NativeLayer {
    id: string;
    meta = {};
    config: {
      dataId: unknown;
      isVisible: boolean;
      columnMode: string;
      color: number[];
      columns: Record<string, {value: string; fieldIdx: number}>;
      textLabel: unknown[];
      visConfig: {
        filled: boolean;
        radius: number;
        opacity: number;
        thickness: number;
        billboard: boolean;
      };
    };
    constructor(props: Record<string, unknown>) {
      this.id = String(props['id'] ?? 'native');
      this.config = {
        dataId: props['dataId'],
        isVisible: true,
        columnMode: 'points',
        color: [1, 2, 3],
        columns: {},
        textLabel: [],
        visConfig: {filled: true, radius: 10, opacity: 0.8, thickness: 1, billboard: false},
        ...props
      };
    }
    assignColumn(key: string, field: {name: string; fieldIdx: number}) {
      return {...this.config['columns'], [key]: {value: field.name, fieldIdx: field.fieldIdx}};
    }
    getZoomFactor({zoom, zoomOffset = 0}: {zoom: number; zoomOffset?: number}) {
      return 2 ** Math.max(14 - zoom + zoomOffset, 0);
    }
    hasLayerData(layerData: {data?: {length?: number; numRows?: number}}) {
      return Boolean(layerData?.data && (layerData.data.length || layerData.data.numRows));
    }
    hasAllColumns() {
      return Object.keys(this.config.columns).length > 0;
    }
    shouldRenderLayer(layerData: {data?: {length?: number; numRows?: number}}) {
      return (
        this.hasAllColumns() &&
        this.hasLayerData(layerData) &&
        typeof this.renderLayer === 'function'
      );
    }
    static findDefaultLayerProps() {
      throw new Error('Unexpected native default geometry scan');
    }
    formatLayerData() {
      throw new Error('Unexpected native source scan');
    }
    updateLayerMeta() {
      throw new Error('Unexpected native metadata scan');
    }
    setInitialLayerConfig() {
      throw new Error('Unexpected native configuration scan');
    }
    getHoverData() {
      throw new Error('Unexpected native tooltip scan');
    }
    renderLayer(): unknown[] {
      throw new Error('Unexpected native accessor render');
    }
    updateLayerDomain() {
      throw new Error('Unexpected native domain scan');
    }
    updateAnimationDomain(domain: unknown) {
      Object.assign(this.config, {animation: {domain}});
    }
    getDeckRenderCallbackData() {
      throw new Error('Unexpected native animation source scan');
    }
  }
  return {
    LayerClasses: Object.fromEntries(
      [
        'point',
        'line',
        'arc',
        'geojson',
        'trip',
        'grid',
        'hexagon',
        'icon',
        'hexagonId',
        's2',
        'a5',
        'geohash'
      ].map(type => [
        type,
        class extends NativeLayer {
          get type() {
            return type;
          }
        }
      ])
    )
  };
});

function vector<Format extends GPUVectorFormat>(format: Format): GPUVector<Format> {
  return new GPUVector({
    type: 'buffer',
    name: 'resident-test-vector',
    format,
    length: 4,
    buffer: {byteLength: 64, destroy: vi.fn(), readAsync: vi.fn()} as unknown as Buffer,
    ownsBuffer: false
  });
}
function setup(onRenderError?: Parameters<typeof createResidentLayerClasses>[0]['onRenderError']) {
  const binding: ResidentDatasetBinding = {
    point: {positions: vector('float32x2'), columns: {lat: 'latitude', lng: 'longitude'}},
    connections: {
      sourcePositions: vector('float32x2'),
      targetPositions: vector('float32x2'),
      columns: {
        lat0: 'latitude',
        lng0: 'longitude',
        lat1: 'targetLatitude',
        lng1: 'targetLongitude'
      }
    },
    selectionMask: vector('uint32'),
    bounds: [-2, -1, 2, 1],
    onHover: vi.fn(),
    resolveSourceRow: index => 100 + index
  };
  const onUnsupported = vi.fn();
  const classes = createResidentLayerClasses({
    isResidentDataset: id => id === 'resident',
    resolveDataset: () => binding,
    onUnsupported,
    onRenderError
  });
  return {binding, classes, onUnsupported};
}
function setColumns(
  layer: InstanceType<
    | typeof LayerClasses.point
    | typeof LayerClasses.arc
    | typeof LayerClasses.icon
    | typeof LayerClasses.hexagonId
  >,
  columns: Record<string, string>
): void {
  for (const [key, value] of Object.entries(columns))
    layer.config.columns = layer.assignColumn(key, {
      name: value,
      fieldIdx: 0
    }) as typeof layer.config.columns;
}

function inaccessibleSource<Source extends object>(): Source {
  // Every property read fails, so this fixture can stand in for any native source interface.
  return new Proxy({} as Source, {
    get(_target, property) {
      // The guard may only inspect the dataset type to exempt tile-3d tilesets.
      if (property === 'dataset') return undefined;
      throw new Error('CPU source access');
    }
  });
}

describe('Kepler resident layer overrides', () => {
  test('point altitude requires matching XYZ and restores flat XY rendering when removed', () => {
    const {binding, classes, onUnsupported} = setup();
    const layer = new classes.point({id: 'altitude-points', dataId: 'resident'});
    setColumns(layer, {lat: 'latitude', lng: 'longitude', altitude: 'height'});
    expect(layer.renderLayer({})).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith(
      layer.id,
      'Point altitude requires matching resident XYZ positions.'
    );

    const positions = vector('float32x3');
    binding.point = {
      positions,
      columns: {lat: 'latitude', lng: 'longitude', altitude: 'height'}
    };
    const rendered = layer.renderLayer({mapState: {zoom: 12, dragRotate: false}});
    expect(rendered).toHaveLength(1);
    expect(rendered[0].props.getPosition).toBe(positions);
    expect(rendered[0].props.selectionMask).toBe(binding.selectionMask);
    expect(rendered[0].props.parameters).toMatchObject({
      depthCompare: 'less-equal',
      depthWriteEnabled: true
    });
    expect(
      layer.renderLayer({mapState: {layerParameters: {depthWriteEnabled: false}}})[0].props
        .parameters.depthWriteEnabled
    ).toBe(false);

    setColumns(layer, {altitude: 'differentHeight'});
    expect(layer.renderLayer({})).toEqual([]);
    setColumns(layer, {altitude: ''});
    binding.point = {
      positions: vector('float32x2'),
      columns: {lat: 'latitude', lng: 'longitude'}
    };
    const flat = layer.renderLayer({mapState: {zoom: 12, dragRotate: true}});
    expect(flat[0].props.getPosition).toBe(binding.point.positions);
    expect(flat[0].props.parameters).toMatchObject({
      depthCompare: 'always',
      depthWriteEnabled: false
    });
  });

  test('point labels share rendered radius, survive hidden markers and retire on removal', () => {
    const {binding, onUnsupported} = setup();
    binding.channels = {sizes: vector('float32')};
    const label = new GPUScatterplotLayer({
      id: 'native-point-label-0',
      getPosition: binding.point!.positions,
      pickable: true
    });
    const resolvePointLabels = vi.fn<
      NonNullable<Parameters<typeof createResidentLayerClasses>[0]['resolvePointLabels']>
    >(layer => ({layers: layer.config.textLabel.some(item => item.field) ? [label] : []}));
    const classes = createResidentLayerClasses({
      isResidentDataset: () => true,
      resolveDataset: () => binding,
      resolvePointLabels,
      onUnsupported
    });
    const layer = new classes.point({id: 'labelled-points', dataId: 'resident'});
    setColumns(layer, binding.point!.columns);
    Object.assign(layer.config, {textLabel: [{field: {name: 'caption'}}]});
    const rendered = layer.renderLayer({mapState: {zoom: 12}});
    expect(rendered).toHaveLength(2);
    expect(resolvePointLabels.mock.lastCall?.[1]).toBe(binding);
    expect(resolvePointLabels.mock.lastCall?.[3]).toMatchObject({
      radius: rendered[0].props.getRadius,
      radiusScale: rendered[0].props.radiusScale,
      radiusMaxPixels: rendered[0].props.radiusMaxPixels
    });
    expect(rendered[1].id).toBe(label.id);
    expect(rendered[1].props.pickable).toBe(false);
    expect(rendered[1].props.nativeLayerId).toBe(layer.id);
    layer.config.isVisible = false;
    const hidden = layer.renderLayer({mapState: {zoom: 12}});
    expect(hidden.every(renderedLayer => renderedLayer.props.visible === false)).toBe(true);
    expect(resolvePointLabels).toHaveBeenCalledTimes(1);
    layer.config.isVisible = true;
    layer.config.visConfig.filled = false;
    layer.config.visConfig.outline = false;
    const labelsOnly = layer.renderLayer({visible: false, mapState: {zoom: 12}});
    expect(labelsOnly).toHaveLength(1);
    expect(labelsOnly[0].props.visible).toBe(false);
    layer.config.textLabel = [];
    expect(layer.renderLayer({})).toEqual([]);
    expect(resolvePointLabels).toHaveBeenCalledTimes(3);
    expect(onUnsupported).toHaveBeenLastCalledWith(layer.id, null);
  });

  test('unprepared native uploads never construct GLSL layers on the WebGPU host', () => {
    let prepared = false;
    const onPreparationRequired = vi.fn();
    const classes = createPreparedLayerClasses(LayerClasses, {
      isPrepared: () => prepared,
      onPreparationRequired
    });
    for (const [name, LayerClass] of Object.entries(classes)) {
      const layer = new LayerClass({id: name, dataId: 'uploaded-csv'});
      expect(
        layer.renderLayer(inaccessibleSource<Parameters<typeof layer.renderLayer>[0]>())
      ).toEqual([]);
      expect(
        layer.renderLayer(inaccessibleSource<Parameters<typeof layer.renderLayer>[0]>())
      ).toEqual([]);
      expect(onPreparationRequired).toHaveBeenCalledWith(name, PREPARE_DATA_MESSAGE);
    }
    expect(onPreparationRequired).toHaveBeenCalledTimes(Object.keys(classes).length);
    prepared = true;
    const layer = new classes.point({id: 'point', dataId: 'uploaded-csv'});
    // Once prepared, the guard delegates to the renderer and clears its own diagnostic.
    expect(() => layer.renderLayer({})).toThrow('Unexpected native accessor render');
    expect(onPreparationRequired).toHaveBeenLastCalledWith('point', null);
  });

  test('H3 column branch preserves source identity and native scale controls', () => {
    const {binding, classes, onUnsupported} = setup();
    const columns = {
      positions: vector('float32x2'),
      vertices: vector('float32x2').data[0]!,
      edgeDistance: 70,
      selectionMask: vector('uint32'),
      colors: vector('unorm8x4'),
      coverages: vector('float32'),
      elevations: vector('float32'),
      resolveSourceRow: (index: number) => [2, 8, 9, 12][index]!
    };
    binding.h3 = {column: 'cell', branch: 'columns', columns};
    const layer = new classes.hexagonId({id: 'cells', dataId: 'resident'});
    setColumns(layer, {hex_id: 'cell'});
    Object.assign(layer.config.visConfig, {
      coverage: 0.7,
      enable3d: true,
      outline: true,
      elevationScale: 5,
      thickness: 2
    });
    const dataset = new Proxy(
      {},
      {
        get: () => {
          throw new Error('CPU source access');
        }
      }
    );
    expect(layer.formatLayerData(dataset, undefined)).toEqual({resident: true});
    expect(layer.updateLayerDomain(dataset, undefined)).toBe(layer);
    const rendered = layer.renderLayer({mapState: {zoom: 6}});
    expect(rendered).toHaveLength(1);
    const properties = rendered[0].props;
    expect(properties.id).toBe('cells-resident-hexagonId-columns');
    expect(properties.nativeLayerId).toBe('cells');
    expect(properties.vertices).toBe(columns.vertices);
    expect(properties.selectionMask).toBe(columns.selectionMask);
    expect(properties.getCoverage).toBe(columns.coverages);
    expect(properties.coverage).toBe(0.7);
    expect(properties.getElevation).toBe(columns.elevations);
    expect(properties.elevationScale).toBe(20);
    expect(properties.stroked).toBe(false);
    expect(properties.flatShading).toBe(true);
    expect(properties.parameters.depthWriteEnabled).toBe(true);
    properties.onHover({index: 1} as PickingInfo, {});
    expect(binding.onHover).toHaveBeenCalledWith(expect.objectContaining({sourceRowIndex: 8}));
    expect(onUnsupported).toHaveBeenLastCalledWith('cells', null);
  });

  test('H3 polygon branch switches physical renderer and uses native high-precision outline policy', () => {
    const {binding, classes, onUnsupported} = setup();
    const polygons = {
      positions: vector('vertex-list<float32x4>'),
      indices: vector('vertex-list<uint32>'),
      rowIndices: vector('vertex-list<uint32>'),
      getFillColor: vector('unorm8x4'),
      getElevation: vector('float32'),
      selectionMask: vector('uint32'),
      extrusionRings: {
        positions: vector('vertex-list<float32x3>'),
        sourceRowIds: vector('uint32'),
        windingOrder: 'CCW' as const
      }
    };
    binding.h3 = {
      column: 'cell',
      branch: 'polygons',
      polygons,
      polygonOutlines: {getPath: vector('vertex-list<float32x3>'), selectionMask: vector('uint32')}
    };
    const layer = new classes.hexagonId({id: 'cells', dataId: 'resident'});
    setColumns(layer, {hex_id: 'cell'});
    Object.assign(layer.config.visConfig, {
      outline: true,
      opacity: 0.6,
      strokeOpacity: 0.2,
      thickness: 2
    });
    let rendered = layer.renderLayer({mapState: {zoom: 6}});
    expect(rendered.map(value => value.props.id)).toEqual([
      'cells-resident-hexagonId-polygons',
      'cells-resident-hexagonId-polygon-outlines'
    ]);
    expect(rendered[1].props.opacity).toBe(0.6);
    expect(rendered[1].props.widthScale).toBe(2);
    layer.config.visConfig.enable3d = true;
    rendered = layer.renderLayer({mapState: {zoom: 6}});
    expect(rendered).toHaveLength(1);
    expect(rendered[0].props.extrusionRings).toBe(polygons.extrusionRings);
    expect(rendered[0].props.getElevation).toBe(polygons.getElevation);
    expect(rendered[0].props.extruded).toBe(true);
    expect(rendered[0].props.parameters.depthWriteEnabled).toBe(true);
    expect(
      layer.renderLayer({mapState: {zoom: 6, layerParameters: {depthWriteEnabled: false}}})[0].props
        .parameters.depthWriteEnabled
    ).toBe(false);
    binding.h3.branch = 'empty';
    expect(layer.renderLayer({})).toEqual([]);
    binding.h3.column = 'other';
    expect(layer.renderLayer({})).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith(
      'cells',
      expect.stringContaining('does not match')
    );
  });

  test('native mesh icons preserve grouped identity and native radius controls', () => {
    const {binding, classes, onUnsupported} = setup();
    const group = {
      iconId: 'marker/one',
      iconGeometry: vector('float32x3').data[0]!,
      positions: vector('float32x2'),
      selectionMask: vector('uint32'),
      colors: vector('unorm8x4'),
      radii: vector('float32'),
      resolveSourceRow: (index: number) => [3, 7, 11, 15][index]!
    };
    binding.icon = {
      columns: {lat: 'latitude', lng: 'longitude', icon: 'symbol'},
      groups: [group, {...group, iconId: 'marker%2Fone'}]
    };
    const layer = new classes.icon({id: 'symbols', dataId: 'resident'});
    setColumns(layer, binding.icon.columns);
    layer.config.visConfig.radius = 10;
    const poisonedDataset = new Proxy(
      {},
      {
        get: () => {
          throw new Error('CPU source access');
        }
      }
    );
    expect(layer.formatLayerData(poisonedDataset, undefined)).toEqual({resident: true});
    expect(layer.updateLayerDomain(poisonedDataset, undefined)).toBe(layer);
    const onRedrawNeeded = vi.fn();
    const rendered = layer.renderLayer({mapState: {zoom: 12}, layerCallbacks: {onRedrawNeeded}});
    expect(layer.onRedrawNeeded).toBe(onRedrawNeeded);
    expect(rendered[0].props.id).not.toBe(rendered[1].props.id);
    expect(rendered[0].props.nativeLayerId).toBe('symbols');
    expect(rendered[0].props.iconGeometry).toBe(group.iconGeometry);
    expect(rendered[0].props.selectionMask).toBe(group.selectionMask);
    expect(rendered[0].props.getFillColor).toBe(group.colors);
    expect(rendered[0].props.getRadius).toBe(group.radii);
    expect(rendered[0].props.radiusScale).toBe(40);
    expect(rendered[0].props.radiusMaxPixels).toBe(500);
    rendered[0].props.onHover({index: 1} as PickingInfo, {});
    expect(binding.onHover).toHaveBeenCalledWith(expect.objectContaining({sourceRowIndex: 7}));
    layer.config.visConfig.fixedRadius = true;
    expect(layer.renderLayer({mapState: {zoom: 12}})[0].props.radiusScale).toBe(1);
    layer.config.columns.icon.value = 'otherSymbol';
    expect(layer.renderLayer({})).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith('symbols', expect.stringContaining('columns'));
  });
  test('native Trips use prepared timestamps and animation domains without source access', () => {
    const {binding, classes, onUnsupported} = setup();
    const originMilliseconds = 1700000000000;
    binding.trip = {
      column: 'geometry',
      getPath: vector('vertex-list<float32x3>'),
      getTimestamps: vector('vertex-list<float32>'),
      selectionMask: vector('uint32'),
      resolveSourceRow: index => 100 + index,
      originMilliseconds,
      maximumTimeErrorMilliseconds: 0,
      animationDomainMilliseconds: [originMilliseconds, originMilliseconds + 5000]
    };
    const layer = new classes.trip({id: 'routes', dataId: 'resident'});
    layer.config.columnMode = 'geojson';
    layer.config.columns.geojson = {value: 'geometry', fieldIdx: 0};
    layer.config.visConfig.thickness = 0.5;
    layer.config.visConfig.billboard = true;
    layer.config.visConfig.trailLength = 2;
    layer.config.visConfig.fadeTrail = true;
    const source = new Proxy(
      {},
      {
        get() {
          throw new Error('Unexpected source read');
        }
      }
    );
    expect(layer.formatLayerData(source, undefined)).toEqual({resident: true});
    expect(layer.config.animation.domain).toEqual(binding.trip.animationDomainMilliseconds);
    expect(layer.getDeckRenderCallbackData({} as never)).toBeNull();
    expect(layer.renderLayer({mapState: {zoom: 12}})).toEqual([]);
    const options = {
      mapState: {zoom: 12},
      animationConfig: {
        domain: [originMilliseconds - 1000, originMilliseconds + 5000],
        currentTime: originMilliseconds + 1500
      }
    };
    const rendered = layer.renderLayer(options)[0];
    expect(rendered.props.getPath).toBe(binding.trip.getPath);
    expect(rendered.props.getTimestamps).toBe(binding.trip.getTimestamps);
    expect(rendered.props.selectionMask).toBe(binding.trip.selectionMask);
    expect(rendered.props.currentTime).toBe(1500);
    expect(rendered.props.trailLength).toBe(2000);
    expect(rendered.props.fadeTrail).toBe(true);
    expect(rendered.props.timestampMode).toBe('trips');
    expect(rendered.props.capRounded).toBe(true);
    expect(rendered.props.jointRounded).toBe(true);
    expect(rendered.props.widthUnits).toBe('meters');
    expect(rendered.props.widthScale).toBeCloseTo((0.5 * 4 * 8 * 2) / 3);
    rendered.props.onHover({index: 1} as PickingInfo, {});
    expect(binding.onHover).toHaveBeenCalledWith(expect.objectContaining({sourceRowIndex: 101}));
    layer.config.visConfig.scenegraphEnabled = true;
    expect(layer.renderLayer(options)).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith('routes', expect.stringContaining('scenegraph'));
    layer.config.visConfig.scenegraphEnabled = false;
    layer.config.columnMode = 'table';
    expect(layer.renderLayer(options)).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith(
      'routes',
      expect.stringContaining('column mode')
    );
  });
  test('native Trips scenegraph markers share the trip buffers and follow the resolver state', () => {
    const {binding, onUnsupported} = setup();
    const originMilliseconds = 1700000000000;
    binding.trip = {
      column: 'geometry',
      getPath: vector('vertex-list<float32x3>'),
      getTimestamps: vector('vertex-list<float32>'),
      selectionMask: vector('uint32'),
      resolveSourceRow: index => 100 + index,
      originMilliseconds,
      maximumTimeErrorMilliseconds: 0,
      animationDomainMilliseconds: [originMilliseconds, originMilliseconds + 5000]
    };
    const scenegraph = {scenes: [], nodes: [], meshes: []};
    let marker: ReturnType<
      NonNullable<Parameters<typeof createResidentLayerClasses>[0]['resolveTripMarker']>
    > = {status: 'loading'};
    const classes = createResidentLayerClasses({
      isResidentDataset: id => id === 'resident',
      resolveDataset: () => binding,
      resolveTripMarker: () => marker,
      onUnsupported
    });
    const layer = new classes.trip({id: 'routes', dataId: 'resident'});
    layer.config.columnMode = 'geojson';
    layer.config.columns.geojson = {value: 'geometry', fieldIdx: 0};
    layer.config.visConfig.trailLength = 2;
    layer.config.visConfig.scenegraphEnabled = true;
    layer.config.visConfig.scenegraphColorEnabled = true;
    layer.config.visConfig.scenegraphUseTrailColor = false;
    layer.config.visConfig.scenegraphColor = [255, 0, 255];
    const options = {
      mapState: {zoom: 12},
      animationConfig: {
        domain: [originMilliseconds, originMilliseconds + 5000],
        currentTime: originMilliseconds + 1500
      }
    };
    // Trails keep drawing while the model loads.
    expect(layer.renderLayer(options).map(rendered => rendered.constructor.name)).toEqual([
      'GPUPathLayer'
    ]);
    marker = {status: 'ready', scenegraph, sizeScale: 2, baseRotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]};
    const rendered = layer.renderLayer(options);
    expect(rendered.map(item => item.constructor.name)).toEqual([
      'GPUPathLayer',
      'GPUTripMarkerLayer'
    ]);
    expect(rendered[1].props.getPath).toBe(binding.trip.getPath);
    expect(rendered[1].props.getTimestamps).toBe(binding.trip.getTimestamps);
    expect(rendered[1].props.selectionMask).toBe(binding.trip.selectionMask);
    expect(rendered[1].props.currentTime).toBe(1500);
    expect(rendered[1].props.scenegraph).toBe(scenegraph);
    expect(rendered[1].props.vertexSelection).toBeUndefined();
    expect(rendered[1].props.getColor).toEqual([255, 0, 255, 255]);
    marker = {status: 'failed', reason: 'Scenegraph model failed to load: HTTP 404'};
    expect(layer.renderLayer(options).map(item => item.constructor.name)).toEqual(['GPUPathLayer']);
    expect(onUnsupported).toHaveBeenLastCalledWith('routes', expect.stringContaining('HTTP 404'));
    marker = {status: 'ready', scenegraph, sizeScale: 2, baseRotation: [1, 0, 0, 0, 1, 0, 0, 0, 1]};
    // GeoJSON trips have no per-vertex datums, so a roll field leaves the marker unrotated.
    layer.config.rollField = {name: 'roll', fieldIdx: 1} as never;
    expect(layer.renderLayer(options)).toHaveLength(2);
    // Table-mode datum fields and per-timestamp colors need resident per-vertex channels.
    binding.trip.grouping = {
      timestampColumn: 'time',
      longitudeColumn: 'lon',
      latitudeColumn: 'lat'
    };
    layer.config.columnMode = 'table';
    layer.config.columns = {
      id: {value: 'geometry', fieldIdx: 0},
      timestamp: {value: 'time', fieldIdx: 1},
      lng: {value: 'lon', fieldIdx: 2},
      lat: {value: 'lat', fieldIdx: 3}
    } as never;
    expect(layer.renderLayer(options)).toHaveLength(1);
    expect(onUnsupported).toHaveBeenLastCalledWith('routes', expect.stringContaining('roll'));
    const roll = vector('float32');
    binding.trip.markerChannels = {vertexSelection: vector('uint32'), roll};
    layer.config.visConfig.invertRoll = true;
    const tableRendered = layer.renderLayer(options);
    expect(tableRendered).toHaveLength(2);
    expect(tableRendered[1].props.getVertexRoll).toBe(roll);
    expect(tableRendered[1].props.getVertexPitch).toBeUndefined();
    expect(tableRendered[1].props.invertRoll).toBe(true);
    layer.config.rollField = null;
    layer.config.colorField = {name: 'level', fieldIdx: 4} as never;
    layer.config.visConfig.scenegraphUseTrailColor = true;
    binding.trip.getColor = vector('unorm8x4');
    expect(layer.renderLayer(options)).toHaveLength(1);
    expect(onUnsupported).toHaveBeenLastCalledWith('routes', expect.stringContaining('per-vertex'));
    const vertexColors = vector('unorm8x4');
    binding.trip.markerChannels = {vertexSelection: vector('uint32'), colors: vertexColors};
    expect(layer.renderLayer(options)[1].props.getVertexColor).toBe(vertexColors);
  });
  test('maps native point styles and masks without formatting source rows', () => {
    const {binding, classes} = setup();
    const layer = new classes.point({id: 'points', dataId: 'resident', color: [10, 20, 30]});
    setColumns(layer, binding.point!.columns);
    layer.config.visConfig.filled = true;
    layer.config.visConfig.radius = 10;
    layer.config.visConfig.billboard = false;
    const poisonedDataset = inaccessibleSource<Parameters<typeof layer.updateLayerMeta>[0]>();
    const layerData = layer.formatLayerData(poisonedDataset, undefined);
    expect(layerData).toEqual({resident: true});
    expect(layer.shouldRenderLayer(layerData)).toBe(true);
    expect(layer.meta['bounds']).toEqual(binding.bounds);
    layer.meta = {};
    layer.setInitialLayerConfig(poisonedDataset);
    expect(layer.meta['bounds']).toEqual(binding.bounds);
    layer.updateLayerMeta(poisonedDataset);
    expect(layer.meta['bounds']).toEqual(binding.bounds);
    const rendered = layer.renderLayer({mapState: {zoom: 12}})[0];
    expect(rendered.props.getPosition).toBe(binding.point!.positions);
    expect(rendered.props.selectionMask).toBe(binding.selectionMask);
    expect(rendered.props.getFillColor).toEqual([10, 20, 30, 255]);
    expect(rendered.props.radiusUnits).toBe('meters');
    expect(rendered.props.radiusScale).toBe(40);
    expect(rendered.props.billboard).toBe(false);
    expect(layer.renderLayer({mapState: {zoom: 12}, visible: false})[0].props.visible).toBe(false);
    expect(rendered.props.onHover({index: 2} as PickingInfo, {})).toBe(false);
    const hoverInformation = {index: 2, layer: rendered};
    expect(
      layer.getHoverData(
        {index: 2},
        inaccessibleSource<Parameters<typeof layer.getHoverData>[1]>(),
        [],
        inaccessibleSource<Parameters<typeof layer.getHoverData>[3]>(),
        hoverInformation
      )
    ).toMatchObject({sourceRowIndex: 102, _dataContainer: null});
    expect(binding.onHover).toHaveBeenCalledWith(
      expect.objectContaining({index: 2, sourceRowIndex: 102, datasetId: 'resident'})
    );
  });
  test('makes native line and arc types use borrowed source geometry', () => {
    const {binding, classes} = setup();
    for (const kind of ['line', 'arc'] as const) {
      const layer = new classes[kind]({id: kind, dataId: 'resident'});
      setColumns(layer, binding.connections!.columns);
      layer.config.visConfig.thickness = 6;
      layer.config.visConfig.targetColor = null as never;
      const rendered = layer.renderLayer({})[0];
      expect(rendered.props.getSourcePosition).toBe(binding.connections!.sourcePositions);
      expect(rendered.props.getTargetPosition).toBe(binding.connections!.targetPositions);
      expect(rendered.props.selectionMask).toBe(binding.selectionMask);
      expect(rendered.props.widthScale).toBe(4);
      if (kind === 'arc') expect(rendered.props.arcMode).toBe('deck');
    }
  });
  test('native layer type changes cannot inherit another GPU renderer model', () => {
    const {binding, classes} = setup();
    const point = new classes.point({id: 'same-native-layer', dataId: 'resident'});
    const line = new classes.line({id: 'same-native-layer', dataId: 'resident'});
    const arc = new classes.arc({id: 'same-native-layer', dataId: 'resident'});
    setColumns(point, binding.point!.columns);
    setColumns(line, binding.connections!.columns);
    setColumns(arc, binding.connections!.columns);
    const rendered = [point, line, arc].map(layer => layer.renderLayer({})[0]);
    expect(new Set(rendered.map(layer => layer.id)).size).toBe(3);
    expect(
      rendered.map(layer =>
        'nativeLayerId' in layer.props ? layer.props.nativeLayerId : undefined
      )
    ).toEqual(['same-native-layer', 'same-native-layer', 'same-native-layer']);
    expect(point.renderLayer({})[0].id).toBe(rendered[0].id);
  });
  test('routes renderer failures to the captured native layer identity', () => {
    const onRenderError = vi.fn();
    const {binding, classes, onUnsupported} = setup(onRenderError);
    const layer = new classes.point({id: 'native-point', dataId: 'resident'});
    setColumns(layer, binding.point!.columns);
    const rendered = layer.renderLayer({})[0];
    const failure = new Error('Injected renderer failure');
    expect(rendered.props.onError?.(failure)).toBe(true);
    expect(onRenderError).toHaveBeenCalledWith(layer, failure);
    expect(onUnsupported).toHaveBeenLastCalledWith('native-point', failure.message);
    expect(rendered.id).not.toBe(layer.id);
  });
  test('maps independent connection colors and gives analysis overrides precedence at both ends', () => {
    const {binding, classes, onUnsupported} = setup();
    for (const kind of ['line', 'arc'] as const) {
      const layer = new classes[kind]({id: kind, dataId: 'resident', color: [10, 20, 30]});
      setColumns(layer, binding.connections!.columns);
      layer.config.visConfig.targetColor = [40, 50, 60];
      let rendered = layer.renderLayer({})[0];
      expect(rendered.props[kind === 'line' ? 'getColor' : 'getSourceColor']).toEqual([
        10, 20, 30, 255
      ]);
      expect(rendered.props.getTargetColor).toEqual([40, 50, 60, 255]);
      expect(onUnsupported).toHaveBeenLastCalledWith(kind, null);

      const sourceColors = vector('unorm8x4');
      const targetColors = vector('unorm8x4');
      const override = vector('unorm8x4');
      binding.channels = {colors: sourceColors, targetColors};
      rendered = layer.renderLayer({})[0];
      expect(rendered.props[kind === 'line' ? 'getColor' : 'getSourceColor']).toBe(sourceColors);
      expect(rendered.props.getTargetColor).toBe(targetColors);
      binding.colorOverride = override;
      rendered = layer.renderLayer({})[0];
      expect(rendered.props[kind === 'line' ? 'getColor' : 'getSourceColor']).toBe(override);
      expect(rendered.props.getTargetColor).toBe(override);
      binding.colorOverride = undefined;
      binding.channels = undefined;
    }
  });
  test('hides unsupported channels, geometry changes, and layer types with a diagnostic', () => {
    const {binding, classes, onUnsupported} = setup();
    const layer = new classes.point({id: 'points', dataId: 'resident'});
    setColumns(layer, binding.point!.columns);
    layer.config.visConfig.filled = true;
    // Neighbor-on-hover is supported (resident neighbor overlay), so it no longer hides the layer.
    layer.config.visConfig.showNeighborOnHover = true;
    expect(layer.renderLayer({})).toHaveLength(1);
    layer.config.visConfig.showNeighborOnHover = false;
    layer.config.columns = layer.assignColumn('lat', {
      name: 'other',
      fieldIdx: 3
    }) as typeof layer.config.columns;
    expect(layer.renderLayer({})).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith('points', expect.stringContaining('columns'));
    expect(classes.grid.findDefaultLayerProps({id: 'resident'} as never)).toEqual({props: []});
    const grid = new classes.grid({id: 'grid', dataId: 'resident'});
    expect(
      grid.formatLayerData(
        new Proxy(
          {},
          {
            get() {
              throw new Error('CPU source access');
            }
          }
        ),
        undefined
      )
    ).toEqual({resident: true});
    expect(grid.renderLayer({})).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith('grid', expect.stringContaining('no resident'));
  });
  test('passes source-feature masks to GeoJSON fill, paths and outlines with native stroke scaling', () => {
    const {binding, classes} = setup();
    const paths = vector('vertex-list<float32x2>');
    const positions = vector('vertex-list<float32x4>');
    const indices = vector('vertex-list<uint32>');
    binding.geojson = {
      column: 'geometry',
      paths,
      polygonOutlines: paths,
      polygons: {positions, indices, rowIndices: indices}
    };
    const layer = new classes.geojson({id: 'geometry', dataId: 'resident'});
    layer.config.columns.geojson = {value: 'geometry', fieldIdx: 0};
    layer.config.visConfig.filled = true;
    layer.config.visConfig.stroked = true;
    layer.config.visConfig.thickness = 2;
    layer.config.visConfig.opacity = 0.4;
    layer.config.visConfig.strokeOpacity = 0.7;
    const rendered = layer.renderLayer({mapState: {zoom: 12}});
    expect(rendered).toHaveLength(3);
    for (const child of rendered) expect(child.props.selectionMask).toBe(binding.selectionMask);
    expect(rendered[0].props.positions).toBe(positions);
    expect(rendered[0].props.opacity).toBe(0.4);
    expect(rendered[0].props.getFillColor).toEqual([1, 2, 3, 255]);
    expect(rendered[1].props.getPath).toBe(paths);
    expect(rendered[1].props.opacity).toBe(0.7);
    expect(rendered[1].props.getColor).toEqual([1, 2, 3, 255]);
    expect(rendered[1].props.widthUnits).toBe('meters');
    expect(rendered[1].props.widthScale).toBe(64);
    expect(rendered[1].props.billboard).toBe(false);
    layer.config.visConfig.enable3d = true;
    expect(layer.renderLayer({})).toEqual([]);
  });
  test('reuses GPU vectors when native controls change and clears unsupported state', () => {
    const {binding, classes, onUnsupported} = setup();
    const layer = new classes.point({id: 'points', dataId: 'resident'});
    setColumns(layer, binding.point!.columns);
    layer.config.visConfig.filled = true;
    layer.config.colorField = {name: 'value'} as never;
    expect(layer.renderLayer({})).toEqual([]);
    expect(onUnsupported).toHaveBeenLastCalledWith(
      'points',
      expect.stringContaining('color channel')
    );
    binding.channels = {colors: vector('unorm8x4'), sizes: vector('float32')};
    layer.config.sizeField = {name: 'value'} as never;
    layer.config.visConfig.fixedRadius = true;
    const first = layer.renderLayer({})[0];
    expect(first.props.radiusScale).toBe(1);
    expect(first.props.getRadius).toBe(binding.channels.sizes);
    expect(first.props.getFillColor).toBe(binding.channels.colors);
    expect(onUnsupported).toHaveBeenLastCalledWith('points', null);
    layer.config.visConfig.opacity = 0.2;
    const second = layer.renderLayer({})[0];
    expect(second.props.getPosition).toBe(first.props.getPosition);
    expect(second.props.selectionMask).toBe(first.props.selectionMask);
    expect(second.props.opacity).toBe(0.2);
    for (const chunk of binding.point!.positions.data) {
      expect(chunk.buffer.readAsync).not.toHaveBeenCalled();
      expect(chunk.buffer.destroy).not.toHaveBeenCalled();
    }
    binding.enabled = false;
    expect(layer.renderLayer({})).toEqual([]);
  });
  test('GeoJSON vertical modes preserve paths and native polygon order, scales and depth', () => {
    const {binding, classes} = setup();
    const paths = {getPath: vector('vertex-list<float32x3>')};
    const polygons = {
      positions: vector('vertex-list<float32x4>'),
      indices: vector('vertex-list<uint32>'),
      rowIndices: vector('vertex-list<uint32>'),
      getElevation: vector('float32'),
      getElevationOffset: vector('float32'),
      getLineColor: vector('unorm8x4'),
      extrusionRings: {
        positions: vector('vertex-list<float32x3>'),
        sourceRowIds: vector('uint32'),
        windingOrder: 'CCW' as const
      }
    };
    binding.geojson = {
      column: 'geometry',
      polygons,
      paths,
      polygonOutlines: paths,
      points: {positions: vector('float32x2'), positions3D: vector('float32x3')}
    };
    const layer = new classes.geojson({id: 'vertical', dataId: 'resident'});
    layer.config.columns.geojson = {value: 'geometry', fieldIdx: 0};
    Object.assign(layer.config.visConfig, {enable3d: true, stroked: true, elevationScale: 2});
    let rendered = layer.renderLayer({mapState: {zoom: 6}});
    expect(rendered.map(child => child.props.id)).toEqual([
      'vertical-paths',
      'vertical-points',
      'vertical-fill'
    ]);
    expect(rendered[0].props.getPath).toBe(paths.getPath);
    expect(rendered[0].props.miterLimit).toBe(2);
    expect(rendered[1].props.getPosition).toBe(binding.geojson.points!.positions3D);
    expect(rendered[1].props.stroked).toBe(true);
    expect(rendered[2].props.getElevation).toBe(polygons.getElevation);
    expect(rendered[2].props.getElevationOffset).toBe(polygons.getElevationOffset);
    expect(rendered[2].props.elevationScale).toBe(8);
    expect(rendered[2].props.parameters.depthWriteEnabled).toBe(true);
    Object.assign(layer.config.visConfig, {
      enable3d: false,
      elevationOffset: 10,
      filled: false,
      wireframe: true
    });
    rendered = layer.renderLayer({
      mapState: {zoom: 6, layerParameters: {depthWriteEnabled: false}}
    });
    expect(rendered.map(child => child.props.id)).toEqual(['vertical-paths', 'vertical-fill']);
    expect(rendered[1].props.getElevation).toBe(0);
    expect(rendered[1].props.getLineColor).toBe(polygons.getLineColor);
    expect(rendered[1].props.filled).toBe(false);
    expect(rendered[1].props.wireframe).toBe(true);
    expect(rendered[1].props.parameters.depthWriteEnabled).toBe(true);
    Object.assign(layer.config.visConfig, {elevationOffset: -10, filled: true, wireframe: false});
    rendered = layer.renderLayer({mapState: {zoom: 6}});
    expect(rendered.map(child => child.props.id)).toEqual([
      'vertical-fill',
      'vertical-paths',
      'vertical-outlines',
      'vertical-points'
    ]);
    expect(rendered[0].props.extruded).toBe(false);
    expect(rendered[0].props.getElevationOffset).toBe(polygons.getElevationOffset);
  });
  test('keeps multipart geometry masks, styles and picking in each primitive index space', () => {
    const {binding, classes} = setup();
    const pathMask = vector('uint32');
    const outlineMask = vector('uint32');
    const pointMask = vector('uint32');
    const pathColors = vector('unorm8x4');
    const pointColors = vector('unorm8x4');
    binding.geojson = {
      column: 'geometry',
      points: {
        positions: vector('float32x2'),
        positions3D: vector('float32x3'),
        selectionMask: pointMask,
        colors: pointColors,
        strokeColors: pathColors,
        lineWidths: vector('float32'),
        sizes: vector('float32'),
        resolveSourceRow: index => index + 10
      },
      paths: {
        getPath: vector('vertex-list<float32x3>'),
        selectionMask: pathMask,
        getColor: pathColors,
        getWidth: vector('float32'),
        resolveSourceRow: index => index + 20
      },
      polygonOutlines: {
        getPath: vector('vertex-list<float32x3>'),
        selectionMask: outlineMask,
        getColor: pathColors,
        getWidth: vector('float32'),
        resolveSourceRow: index => index + 30
      },
      polygons: {
        positions: vector('vertex-list<float32x4>'),
        indices: vector('vertex-list<uint32>'),
        rowIndices: vector('vertex-list<uint32>'),
        resolveSourceRow: index => index,
        getFillColor: vector('vertex-list<unorm8x4>')
      }
    };
    const layer = new classes.geojson({id: 'parts', dataId: 'resident'});
    layer.config.columns.geojson = {value: 'geometry', fieldIdx: 0};
    layer.config.visConfig.stroked = true;
    layer.config.visConfig.thickness = 2;
    layer.config.radiusField = {name: 'radius'} as never;
    layer.config.sizeField = {name: 'width'} as never;
    layer.config.colorField = {name: 'fill'} as never;
    layer.config.strokeColorField = {name: 'stroke'} as never;
    const inaccessibleSource = new Proxy(
      {},
      {
        get() {
          throw new Error('Native feature discovery must not inspect resident source rows');
        }
      }
    );
    layer.setInitialLayerConfig(inaccessibleSource as never);
    expect(layer.meta.featureTypes).toEqual({point: true, line: true, polygon: true});
    layer.meta = {};
    layer.formatLayerData(inaccessibleSource as never, undefined);
    expect(layer.meta.featureTypes).toEqual({point: true, line: true, polygon: true});
    const rendered = layer.renderLayer({mapState: {zoom: 12}});
    expect(rendered).toHaveLength(4);
    expect(rendered.map(child => child.props.selectionMask)).toEqual([
      binding.selectionMask,
      pathMask,
      outlineMask,
      pointMask
    ]);
    expect(rendered[1].props.getColor).toBe(pathColors);
    expect(rendered[3].props.getFillColor).toBe(pointColors);
    expect(rendered[3].props.getPosition).toBe(binding.geojson.points!.positions3D);
    expect(rendered[3].props.radiusScale).toBe(4);
    for (const [index, child] of rendered.entries()) {
      child.props.onHover({index: 1} as PickingInfo, {});
      expect(binding.onHover).toHaveBeenLastCalledWith(
        expect.objectContaining({
          sourceRowIndex: [1, 21, 31, 11][index]
        })
      );
      child.props.onHover({index: -1} as PickingInfo, {});
      expect(binding.onHover).toHaveBeenLastCalledWith(
        expect.objectContaining({sourceRowIndex: -1})
      );
    }
    binding.geojson.points!.colors = undefined;
    expect(layer.renderLayer({})).toEqual([]);
  });
  test('uses native zoom scaling for fixed radius without a mapped field', () => {
    const {binding, classes} = setup();
    const layer = new classes.point({id: 'fixed', dataId: 'resident'});
    setColumns(layer, binding.point!.columns);
    layer.config.visConfig.fixedRadius = true;
    layer.config.visConfig.radius = 10;
    expect(layer.renderLayer({mapState: {zoom: 12}})[0].props.radiusScale).toBe(40);
  });
  test('supports native hollow point outlines without requiring the disabled fill channel', () => {
    const {binding, classes, onUnsupported} = setup();
    const layer = new classes.point({id: 'outline', dataId: 'resident'});
    setColumns(layer, binding.point!.columns);
    layer.config.colorField = {name: 'unused-fill'} as never;
    layer.config.strokeColorField = {name: 'stroke'} as never;
    layer.config.visConfig.filled = false;
    layer.config.visConfig.outline = true;
    layer.config.visConfig.thickness = 4;
    binding.channels = {strokeColors: vector('unorm8x4')};
    const rendered = layer.renderLayer({mapState: {zoom: 12}})[0];
    expect(rendered.props.filled).toBe(false);
    expect(rendered.props.stroked).toBe(true);
    expect(rendered.props.getLineColor).toBe(binding.channels.strokeColors);
    expect(rendered.props.lineWidthUnits).toBe('pixels');
    expect(rendered.props.lineWidthScale).toBe(4);
    expect(onUnsupported).toHaveBeenLastCalledWith('outline', null);
  });
  test('leaves nonresident native methods intact and waits for pending resident uploads', () => {
    const format = vi
      .spyOn(LayerClasses.point.prototype, 'formatLayerData')
      .mockReturnValue({native: true});
    const render = vi.spyOn(LayerClasses.point.prototype, 'renderLayer').mockReturnValue([]);
    try {
      const classes = createResidentLayerClasses({
        isResidentDataset: id => id === 'resident',
        resolveDataset: () => undefined,
        onUnsupported: vi.fn()
      });
      const native = new classes.point({dataId: 'ordinary'});
      expect(native.formatLayerData({}, undefined)).toEqual({native: true});
      native.renderLayer({});
      expect(format).toHaveBeenCalledTimes(1);
      expect(render).toHaveBeenCalledTimes(1);
      const pending = new classes.point({dataId: 'resident'});
      expect(pending.formatLayerData({}, undefined)).toEqual({resident: true});
      expect(pending.renderLayer({})).toEqual([]);
      expect(format).toHaveBeenCalledTimes(1);
      expect(render).toHaveBeenCalledTimes(1);
    } finally {
      format.mockRestore();
      render.mockRestore();
    }
  });
});
