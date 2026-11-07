// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  collectVectorTileStyleFields,
  deriveNativeStyleLayerFromDeckProps,
  deriveVectorTileConstants,
  deriveVectorTileDrawParameters,
  deriveVectorTileFiltersFromDeckProps,
  getVectorTileDeckStyleSignature
} from '../../../examples/deck/kepler-compute/src/vector-tile-style-derive';
import {
  evaluateVectorTileScalarReference,
  evaluateVectorTileStyleReference,
  packVectorTileColor,
  packVectorTileParameters,
  planVectorTileStyle,
  VECTOR_TILE_KERNEL_SOURCE,
  VECTOR_TILE_PARAMETER_WORD_COUNT
} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-style';
import {computeVectorTileCacheSize} from '../../../examples/deck/kepler-compute/src/vector-tile-cache-budget';
import {resolveVectorTilePick} from '../../../examples/deck/kepler-compute/src/vector-tile-picking';
import {VectorTileFeatureRegistry} from '../../../examples/deck/kepler-compute/src/vector-tile-feature-ids';
import {VectorTileLiveController} from '../../../examples/deck/kepler-compute/src/vector-tile-live-controller';
import type {VectorTileResidentRenderer} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-layers';

const BLACK = packVectorTileColor(0, 0, 0);
const WHITE = packVectorTileColor(255, 255, 255);

function makeDeckProps(overrides: Record<string, unknown> = {}) {
  return {
    opacity: 0.6,
    stroked: true,
    pointRadiusUnits: 'pixels',
    pointRadiusScale: 4,
    pointRadiusMinPixels: 1,
    filterRange: [
      [0, 0],
      [10, 20],
      [0, 0],
      [0, 0]
    ],
    _subLayerProps: {'polygons-stroke': {opacity: 0.3}},
    updateTriggers: {
      getFilterValue: {
        gpuFilter_0: null,
        gpuFilter_1: {name: 'height', domain0: 100},
        gpuFilter_2: null,
        gpuFilter_3: null
      },
      getFillColor: {
        color: [10, 20, 30],
        colorField: {name: 'height', type: 'real'},
        colorScale: 'linear',
        colorDomain: [0, 100],
        colorRange: {colors: ['#000000', '#ffffff']}
      },
      getLineColor: {strokeColor: null, strokeColorField: null},
      getLineWidth: {
        sizeRange: [1, 5],
        strokeWidth: 2,
        sizeField: {name: 'lanes'},
        sizeScale: 'linear'
      },
      getPointRadius: {radius: 6, radiusField: null}
    },
    ...overrides
  };
}

describe('vector tile live style derivation', () => {
  it('rebuilds a native layer description from Kepler channel update triggers', () => {
    const layer = deriveNativeStyleLayerFromDeckProps('tiles', makeDeckProps());
    expect(layer?.config.colorField).toEqual({name: 'height'});
    expect(layer?.config.colorScale).toBe('linear');
    expect(layer?.config.sizeField).toEqual({name: 'lanes'});
    expect(layer?.config.visConfig.sizeRange).toEqual([1, 5]);
    expect(layer?.config.visConfig.opacity).toBe(0.6);
    expect(layer?.config.visConfig.strokeOpacity).toBe(0.3);
    expect(deriveNativeStyleLayerFromDeckProps('tiles', {})).toBeNull();
  });

  it('converts GPU filter ranges back to absolute property ranges', () => {
    expect(deriveVectorTileFiltersFromDeckProps(makeDeckProps())).toEqual([
      {kind: 'range', field: 'height', min: 110, max: 120}
    ]);
    expect(deriveVectorTileFiltersFromDeckProps({})).toEqual([]);
  });

  it('reads draw parameters and constants from the Deck layer', () => {
    expect(deriveVectorTileDrawParameters(makeDeckProps())).toMatchObject({
      opacity: 0.6,
      strokeOpacity: 0.3,
      stroked: true,
      radiusUnits: 'pixels',
      radiusScale: 4,
      radiusMinPixels: 1,
      lineWidthUnits: 'pixels'
    });
    expect(deriveVectorTileConstants(makeDeckProps())).toEqual({lineWidth: 2, radius: 6});
  });

  it('collects numeric, categorical and filter fields', () => {
    const layer = deriveNativeStyleLayerFromDeckProps(
      'tiles',
      makeDeckProps({
        updateTriggers: {
          ...makeDeckProps().updateTriggers,
          getFillColor: {
            ...makeDeckProps().updateTriggers.getFillColor,
            colorField: {name: 'kind'},
            colorScale: 'ordinal'
          }
        }
      })
    );
    const fields = collectVectorTileStyleFields(layer, [
      {kind: 'range', field: 'height', min: 0, max: 1}
    ]);
    expect(fields.categorical).toEqual(['kind']);
    expect([...fields.numeric].sort()).toEqual(['height', 'lanes']);
  });

  it('changes its signature only when a feeding prop changes', () => {
    const first = getVectorTileDeckStyleSignature(makeDeckProps());
    expect(getVectorTileDeckStyleSignature(makeDeckProps())).toBe(first);
    expect(getVectorTileDeckStyleSignature(makeDeckProps({opacity: 0.7}))).not.toBe(first);
    expect(getVectorTileDeckStyleSignature(makeDeckProps({onTileLoad: () => {}}))).toBe(first);
  });
});

describe('vector tile data-driven channels', () => {
  const layer = deriveNativeStyleLayerFromDeckProps('tiles', makeDeckProps())!;

  it('plans color, width and radius channels together', () => {
    const result = planVectorTileStyle(layer, {
      layerId: 'tiles',
      zoom: 4,
      constants: deriveVectorTileConstants(makeDeckProps()),
      getFieldValues: fieldName => (fieldName === 'lanes' ? [1, 3, 5] : [0, 50, 100])
    });
    expect(result.supported).toBe(true);
    if (!result.supported) return;
    expect(result.channels.width).toMatchObject({
      kind: 'field',
      fieldName: 'lanes',
      scale: 'linear',
      domain: [1, 5],
      range: [1, 5]
    });
    // No radius field: the constant radius from the Deck layer.
    expect(result.channels.radius).toEqual({kind: 'constant', value: 6});
    // No stroke field: stroke follows the constant base color.
    expect(result.channels.stroke).toEqual({
      kind: 'constant',
      color: packVectorTileColor(10, 20, 30)
    });
    if (result.channels.width?.kind !== 'field') throw new Error('expected field width');
    expect(evaluateVectorTileScalarReference(result.channels.width, 1)).toBe(1);
    expect(evaluateVectorTileScalarReference(result.channels.width, 5)).toBe(5);
    expect(evaluateVectorTileScalarReference(result.channels.width, 3)).toBe(3);
    // Outside the sampled domain clamps; missing values use the null value.
    expect(evaluateVectorTileScalarReference(result.channels.width, 99)).toBe(5);
    expect(evaluateVectorTileScalarReference(result.channels.width, Number.NaN)).toBe(0);
  });

  it('maps sqrt and log size scales and rejects non-positive log domains', () => {
    const sqrtLayer = {
      ...layer,
      config: {...layer.config, sizeScale: 'sqrt', sizeDomain: undefined}
    };
    const result = planVectorTileStyle(sqrtLayer, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: fieldName => (fieldName === 'lanes' ? [0, 4] : [0, 1])
    });
    if (!result.supported || result.channels.width?.kind !== 'field')
      throw new Error('expected sqrt width');
    expect(evaluateVectorTileScalarReference(result.channels.width, 1)).toBe(1 + 4 * 0.5);
    const logLayer = {...layer, config: {...layer.config, sizeScale: 'log'}};
    const logResult = planVectorTileStyle(logLayer, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: fieldName => (fieldName === 'lanes' ? [0, 4] : [0, 1])
    });
    expect(logResult.supported).toBe(false);
  });

  it('supports categorical colors from the layer dictionary', () => {
    const ordinal = {
      ...layer,
      config: {
        ...layer.config,
        colorField: {name: 'kind'},
        colorScale: 'ordinal',
        colorDomain: undefined,
        visConfig: {
          ...layer.config.visConfig,
          colorRange: {colors: ['#ff0000', '#00ff00', '#0000ff']}
        }
      }
    };
    const result = planVectorTileStyle(ordinal, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: () => [1, 3],
      getFieldCategories: () => ['road', 'rail', 'path']
    });
    expect(result.supported).toBe(true);
    if (!result.supported || result.style.kind !== 'field') throw new Error('expected field style');
    expect(result.style.scale).toBe('ordinal');
    const first = evaluateVectorTileStyleReference(result.style, 0);
    // Native ordinal order is the sorted dictionary: path, rail, road.
    expect(first).toBe(evaluateVectorTileStyleReference(result.style, 0));
    expect(
      new Set([0, 1, 2].map(code => evaluateVectorTileStyleReference(result.style as never, code)))
        .size
    ).toBe(3);
    expect(evaluateVectorTileStyleReference(result.style, 3)).toBe(0);
    expect(evaluateVectorTileStyleReference(result.style, Number.NaN)).toBe(0);
    expect(evaluateVectorTileStyleReference(result.style, -1)).toBe(0);
  });

  it('packs stroke, width and radius blocks and keeps the kernel bindings', () => {
    const words = packVectorTileParameters({
      rowCount: 3,
      fieldIndexByName: new Map([['lanes', 2]]),
      filters: [],
      style: {kind: 'constant', color: WHITE},
      channels: {
        stroke: {kind: 'constant', color: BLACK},
        width: {
          kind: 'field',
          fieldName: 'lanes',
          scale: 'sqrt',
          domain: [0, 4],
          range: [1, 5],
          nullValue: 0
        },
        radius: {kind: 'constant', value: 7}
      }
    });
    expect(words.length).toBe(VECTOR_TILE_PARAMETER_WORD_COUNT);
    const floats = new Float32Array(words.buffer);
    const scalarStart = words.length - 20;
    expect(words[scalarStart]).toBe(1);
    expect(words[scalarStart + 1]).toBe(2);
    expect(words[scalarStart + 2]).toBe(1);
    expect(floats[scalarStart + 5]).toBe(4);
    expect(words[scalarStart + 10]).toBe(0);
    expect(floats[scalarStart + 13]).toBe(7);
    // Without a stroke channel the kernel copies the fill color.
    const copied = packVectorTileParameters({
      rowCount: 1,
      fieldIndexByName: new Map(),
      filters: [],
      style: {kind: 'constant', color: WHITE}
    });
    expect(copied[words.length - 20 - 257 - 256 - 9]).toBe(2);
    for (const binding of ['strokeOutput', 'widthOutput', 'radiusOutput'])
      expect(VECTOR_TILE_KERNEL_SOURCE).toContain(`var<storage, read_write> ${binding}:`);
  });
});

describe('vector tile picking, cache budget and ids', () => {
  it('maps a primitive pick to the tile feature through the source row map', () => {
    const features = [
      {id: 7, properties: {name: 'a'}, geometry: null},
      {properties: {name: 'b'}, geometry: null}
    ];
    const pick = resolveVectorTilePick({
      layerId: 'tiles',
      tileKey: '3/1/2',
      primitive: 'paths',
      primitiveIndex: 4,
      features,
      resolveSourceRow: index => (index === 4 ? 1 : -1),
      featureIds: ['tiles:i:7', 'tiles:h:abc']
    });
    expect(pick).toMatchObject({sourceRow: 1, featureId: 'tiles:h:abc', properties: {name: 'b'}});
    expect(
      resolveVectorTilePick({
        layerId: 'tiles',
        tileKey: '3/1/2',
        primitive: 'points',
        primitiveIndex: 0,
        features
      })?.featureId
    ).toBe('tiles:i:7');
    expect(
      resolveVectorTilePick({
        layerId: 'tiles',
        tileKey: 't',
        primitive: 'points',
        primitiveIndex: -1,
        features
      })
    ).toBeNull();
    expect(
      resolveVectorTilePick({
        layerId: 'tiles',
        tileKey: 't',
        primitive: 'polygons',
        primitiveIndex: 9,
        features,
        resolveSourceRow: () => -1
      })
    ).toBeNull();
  });

  it('bounds the Deck tile cache by the working set budget', () => {
    expect(
      computeVectorTileCacheSize({
        budgetBytes: 1000,
        usedBytes: 100,
        residentTileCount: 2,
        visibleTileCount: 1
      })
    ).toBe(20);
    // Never below what is on screen, never above the hard maximum.
    expect(
      computeVectorTileCacheSize({
        budgetBytes: 10,
        usedBytes: 1000,
        residentTileCount: 1,
        visibleTileCount: 9
      })
    ).toBe(9);
    expect(
      computeVectorTileCacheSize({
        budgetBytes: 1e12,
        usedBytes: 10,
        residentTileCount: 1,
        visibleTileCount: 1
      })
    ).toBe(512);
    expect(
      computeVectorTileCacheSize({
        budgetBytes: 1000,
        usedBytes: 100,
        residentTileCount: 2,
        visibleTileCount: 1,
        existingMaxCacheSize: 6
      })
    ).toBe(6);
  });

  it('separates distinct features that share properties inside one tile', () => {
    const registry = new VectorTileFeatureRegistry({sourceLayer: 'roads'});
    const duplicate = {properties: {name: 'Main St'}};
    const ids = registry.registerTile('3/1/1', [duplicate, duplicate, {properties: {name: 'Elm'}}]);
    expect(new Set(ids).size).toBe(3);
    expect(ids[1]).toBe(`${ids[0]}#1`);
    expect(registry.summarize()).toMatchObject({
      uniqueFeatureCount: 3,
      ambiguousFragmentCount: 1,
      hasCollisionRisk: true
    });
    // The same content reloads under the same ids.
    expect(
      registry.registerTile('3/1/1', [duplicate, duplicate, {properties: {name: 'Elm'}}])
    ).toEqual(ids);
    registry.evictTile('3/1/1');
    expect(registry.summarize().ambiguousFragmentCount).toBe(0);
    // A point and a polygon with equal properties no longer collide.
    const [point, polygon] = registry.registerTile('3/2/1', [
      {properties: {name: 'x'}, geometry: {type: 'Point'}},
      {properties: {name: 'x'}, geometry: {type: 'Polygon'}}
    ]);
    expect(point).not.toBe(polygon);
  });
});

class FakeMVTLayer {
  static layerName = 'MVTLayer';
  constructor(
    readonly id: string,
    readonly props: Record<string, unknown> = {}
  ) {}
  clone(props: Record<string, unknown>): FakeMVTLayer {
    return new FakeMVTLayer(this.id, {...this.props, ...props});
  }
}

describe('vector tile controller automatic style', () => {
  function makeController() {
    const calls: string[] = [];
    const filterCalls: unknown[] = [];
    const renderer = {
      syncLayer: () => calls.push('sync'),
      setZoom: () => calls.push('zoom'),
      admitTile: () => calls.push('admit'),
      removeTile: () => calls.push('remove'),
      refreshStyle: () => calls.push('refresh'),
      requireFilterFields: () => calls.push('require'),
      setStyleFromDeckProps: () => calls.push('style'),
      setFilters: (_layerId: string, filters: unknown) => {
        calls.push('filters');
        filterCalls.push(filters);
      }
    } as unknown as VectorTileResidentRenderer;
    const controller = new VectorTileLiveController({
      getReserveMemory: () => null,
      getCreateBuffer: () => null,
      residentRenderer: renderer
    });
    return {calls, filterCalls, controller};
  }

  it('derives style and filters on first wrap and only again when props change', () => {
    const {calls, filterCalls, controller} = makeController();
    const wrapped = controller.wrapLayer(
      new FakeMVTLayer('tiles', makeDeckProps())
    ) as FakeMVTLayer;
    expect(calls.filter(call => call === 'style')).toHaveLength(1);
    expect(filterCalls[0]).toEqual([{kind: 'range', field: 'height', min: 110, max: 120}]);
    expect(typeof wrapped.props['maxCacheSize']).toBe('number');
    // Same props on the next render: nothing is re-derived, so a hook-set filter survives.
    controller.setFilters('tiles', []);
    controller.wrapLayer(new FakeMVTLayer('tiles', makeDeckProps()));
    expect(calls.filter(call => call === 'style')).toHaveLength(1);
    expect(filterCalls[filterCalls.length - 1]).toEqual([]);
    controller.wrapLayer(new FakeMVTLayer('tiles', makeDeckProps({opacity: 0.9})));
    expect(calls.filter(call => call === 'style')).toHaveLength(2);
  });

  it('adds required filter columns to resident CPU batches', () => {
    const {controller} = makeController();
    const wrapped = controller.wrapLayer(new FakeMVTLayer('tiles', {})) as FakeMVTLayer;
    const tile = {
      index: {x: 1, y: 2, z: 3},
      content: [
        {
          id: 1,
          properties: {a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, height: 115},
          geometry: {type: 'Point', coordinates: [0, 0]}
        }
      ]
    };
    (wrapped.props['onTileLoad'] as (value: unknown) => void)(tile);
    // Field discovery stops at eight numeric columns, so the filter field is added on demand.
    expect(controller.getState()[0].numericFieldNames).not.toContain('height');
    controller.wrapLayer(new FakeMVTLayer('tiles', makeDeckProps()));
    const state = controller.getState()[0];
    expect(state.numericFieldNames).toContain('height');
    expect(state.passingFeatureCount).toBe(1);
  });
});
