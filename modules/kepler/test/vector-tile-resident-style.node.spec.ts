// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  evaluateVectorTileFilterReference,
  evaluateVectorTileStyleReference,
  packVectorTileColor,
  packVectorTileParameters,
  planVectorTileStyle,
  VECTOR_TILE_KERNEL_SOURCE,
  VECTOR_TILE_PARAMETER_WORD_COUNT
} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-style';
import {computeDecodedTileFilterMask} from '../../../examples/deck/kepler-compute/src/vector-tile-decoded-batch';
import {VectorTileLiveController} from '../../../examples/deck/kepler-compute/src/vector-tile-live-controller';
import type {VectorTileResidentRenderer} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-layers';

const BLACK = packVectorTileColor(0, 0, 0);
const WHITE = packVectorTileColor(255, 255, 255);

function makeLayer(colorScale: string) {
  return {
    id: 'tiles',
    type: 'vectorTile',
    config: {
      dataId: 'ignored',
      color: [10, 20, 30],
      colorField: {name: 'height'},
      colorScale,
      colorDomain: [0, 100],
      visConfig: {
        opacity: 0.5,
        thickness: 1,
        colorRange: {colors: ['#000000', '#ffffff']}
      }
    }
  };
}

describe('vector tile resident style', () => {
  it('translates a linear color scale through the native style plan', () => {
    const result = planVectorTileStyle(makeLayer('linear') as never, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: () => [10, 20, 30, Number.NaN]
    });
    expect(result.supported).toBe(true);
    if (!result.supported || result.style.kind !== 'field') throw new Error('expected field style');
    expect(result.style.domain).toEqual([10, 30]);
    expect(result.render.opacity).toBe(0.5);
    expect(evaluateVectorTileStyleReference(result.style, 10)).toBe(BLACK);
    expect(evaluateVectorTileStyleReference(result.style, 30)).toBe(WHITE);
    expect(evaluateVectorTileStyleReference(result.style, Number.NaN)).toBe(0);
    const middle = evaluateVectorTileStyleReference(result.style, 20);
    expect(middle & 255).toBe(128);
  });

  it('resolves quantile edges from loaded values and classifies by inner edges', () => {
    const result = planVectorTileStyle(makeLayer('quantile') as never, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: () => [1, 2, 3, 4]
    });
    if (!result.supported || result.style.kind !== 'field') throw new Error('expected field style');
    expect(result.style.domain.length).toBe(3);
    expect(evaluateVectorTileStyleReference(result.style, 1)).toBe(BLACK);
    expect(evaluateVectorTileStyleReference(result.style, 4)).toBe(WHITE);
  });

  it('uses a constant style when no color field is set', () => {
    const layer = makeLayer('linear');
    (layer.config as {colorField: unknown}).colorField = null;
    const result = planVectorTileStyle(layer as never, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: () => []
    });
    expect(result.supported && result.style).toEqual({
      kind: 'constant',
      color: packVectorTileColor(10, 20, 30)
    });
  });

  it('reports unsupported categorical styles instead of approximating them', () => {
    const result = planVectorTileStyle(makeLayer('ordinal') as never, {
      layerId: 'tiles',
      zoom: 4,
      getFieldValues: () => [1]
    });
    expect(result.supported).toBe(false);
  });

  it('packs kernel parameters that agree with the CPU filter mask planning', () => {
    const filters = [
      {kind: 'range', field: 'height', min: 2, max: 3},
      {kind: 'select', field: 'class', values: [1, 2]}
    ] as const;
    const heights = Float32Array.from([1, 2, 3, 4, Number.NaN]);
    const classes = Float32Array.from([1, 2, 3, 1, 1]);
    const batch = {
      family: 'point' as const,
      featureIds: ['a', 'b', 'c', 'd', 'e'],
      positions: new Float32Array(10),
      vertexCounts: new Uint32Array(5).fill(1),
      numericColumns: new Map([
        ['height', heights],
        ['class', classes]
      ])
    };
    const expected = [...computeDecodedTileFilterMask(batch, filters)];
    const reference = heights.map((_, row) =>
      evaluateVectorTileFilterReference(
        filters,
        field => (field === 'height' ? heights : classes)[row]
      )
    );
    expect([...reference]).toEqual(expected);
    const words = packVectorTileParameters({
      rowCount: 5,
      fieldIndexByName: new Map([
        ['height', 0],
        ['class', 1]
      ]),
      filters,
      style: {kind: 'constant', color: WHITE}
    });
    expect(words.length).toBe(VECTOR_TILE_PARAMETER_WORD_COUNT);
    expect(words[0]).toBe(2);
    expect(words[10]).toBe(5);
    expect(words[8]).toBe(WHITE);
    // Missing fields address the NaN slot.
    const missing = packVectorTileParameters({
      rowCount: 1,
      fieldIndexByName: new Map(),
      filters: [{kind: 'range', field: 'absent', min: 0, max: 1}],
      style: {kind: 'constant', color: WHITE}
    });
    expect(missing[12]).toBe(0xffffffff);
  });

  it('keeps the kernel bindings and parameter layout stable', () => {
    for (const binding of ['parameters', 'columns', 'maskOutput', 'colorOutput'])
      expect(VECTOR_TILE_KERNEL_SOURCE).toContain(
        `var<storage, ${binding === 'parameters' || binding === 'columns' ? 'read' : 'read_write'}> ${binding}:`
      );
    expect(VECTOR_TILE_KERNEL_SOURCE).toContain('@workgroup_size(64)');
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

describe('vector tile controller resident routing', () => {
  it('suppresses Deck sublayers and forwards tiles and filters to the resident renderer', () => {
    const calls: string[] = [];
    const renderer = {
      syncLayer: () => calls.push('sync'),
      setZoom: () => calls.push('zoom'),
      admitTile: (_layer: string, tileKey: string) => calls.push(`admit:${tileKey}`),
      removeTile: (_layer: string, tileKey: string) => calls.push(`remove:${tileKey}`),
      setFilters: () => calls.push('filters')
    } as unknown as VectorTileResidentRenderer;
    const controller = new VectorTileLiveController({
      getReserveMemory: () => null,
      getCreateBuffer: () => null,
      residentRenderer: renderer
    });
    const wrapped = controller.wrapLayer(new FakeMVTLayer('tiles')) as FakeMVTLayer;
    expect((wrapped.props['renderSubLayers'] as () => unknown[])()).toEqual([]);
    const tile = {
      index: {x: 1, y: 2, z: 3},
      content: [{id: 1, properties: {height: 5}, geometry: {type: 'Point', coordinates: [0, 0]}}]
    };
    (wrapped.props['onTileLoad'] as (value: unknown) => void)(tile);
    expect(calls).toContain('admit:3/1/2');
    controller.setFilters('tiles', [{kind: 'range', field: 'height', min: 0, max: 10}]);
    expect(calls).toContain('filters');
    (wrapped.props['onTileUnload'] as (value: unknown) => void)(tile);
    expect(calls).toContain('remove:3/1/2');
  });

  it('keeps Deck sublayers when no resident renderer exists', () => {
    const controller = new VectorTileLiveController({
      getReserveMemory: () => null,
      getCreateBuffer: () => null
    });
    const wrapped = controller.wrapLayer(new FakeMVTLayer('tiles')) as FakeMVTLayer;
    expect(wrapped.props['renderSubLayers']).toBeUndefined();
  });
});
