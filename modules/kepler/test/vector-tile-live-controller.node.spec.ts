// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  VectorTileLiveController,
  type VectorTileGpuBuffer
} from '../../../examples/deck/kepler-compute/src/vector-tile-live-controller';

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

function makeTile(x: number, ids: number[]) {
  return {
    index: {x, y: 0, z: 3},
    content: ids.map(id => ({
      id,
      properties: {height: id * 10},
      geometry: {type: 'Point', coordinates: [x, id]}
    }))
  };
}

function makeController(budgetBytes?: number) {
  const reservations: number[] = [];
  const released: number[] = [];
  const buffers: {destroyed: boolean}[] = [];
  const controller = new VectorTileLiveController({
    budgetBytes,
    getReserveMemory: () => byteLength => {
      reservations.push(byteLength);
      return {resize() {}, release: () => released.push(byteLength)};
    },
    getCreateBuffer: () => data => {
      const state = {destroyed: false};
      buffers.push(state);
      const buffer: VectorTileGpuBuffer = {
        byteLength: data.byteLength,
        write() {},
        destroy: () => {
          state.destroyed = true;
        }
      };
      return buffer;
    }
  });
  return {controller, reservations, released, buffers};
}

describe('VectorTileLiveController', () => {
  it('ignores non-MVT layers and wraps MVT layers once', () => {
    const {controller} = makeController();
    const other = {id: 'x', props: {}, clone: () => null, constructor: {layerName: 'Other'}};
    expect(controller.wrapLayer(other)).toBe(other);
    const wrapped = controller.wrapLayer(new FakeMVTLayer('tiles'));
    expect(controller.wrapLayer(wrapped)).toBe(wrapped);
  });

  it('admits tiles, assigns stable ids, filters and releases on unload', () => {
    const {controller, reservations, released, buffers} = makeController();
    const original: unknown[] = [];
    const wrapped = controller.wrapLayer(
      new FakeMVTLayer('tiles', {onTileLoad: (t: unknown) => original.push(t)})
    );
    const props = (wrapped as FakeMVTLayer).props as Record<string, (argument: unknown) => void>;
    const tileA = makeTile(1, [1, 2, 3]);
    const tileB = makeTile(2, [3, 4]);
    props['onTileLoad'](tileA);
    props['onTileLoad'](tileB);
    props['onViewportLoad']([tileA, tileB]);
    expect(original).toHaveLength(2);
    const state = controller.getState()[0];
    expect(state.workingSet.tileCount).toBe(2);
    expect(state.featureSummary?.uniqueFeatureCount).toBe(4);
    expect(state.featureSummary?.splitFeatureCount).toBe(1);
    expect(state.gpuByteLength).toBeGreaterThan(0);
    expect(reservations.length).toBe(4);
    expect(state.scope?.kind).toBe('viewport-working-set');
    expect(state.completeDatasetDecision?.allowed).toBe(false);

    controller.setFilters('tiles', [{kind: 'range', field: 'height', min: 20, max: 30}]);
    expect(controller.getState()[0].passingFeatureCount).toBe(3);
    props['onTileUnload'](tileA);
    expect(controller.getState()[0].workingSet.tileCount).toBe(1);
    expect(buffers.filter(buffer => buffer.destroyed).length).toBeGreaterThan(0);
    expect(released.length).toBeGreaterThan(0);
    expect(controller.decideAnalysis('tiles', {requiresCompleteDataset: true})?.allowed).toBe(
      false
    );
  });

  it('rejects tiles larger than the budget and marks coverage incomplete', () => {
    const {controller} = makeController(10);
    const wrapped = controller.wrapLayer(new FakeMVTLayer('tiles'));
    const props = (wrapped as FakeMVTLayer).props as Record<string, (argument: unknown) => void>;
    const tile = makeTile(1, [1]);
    props['onTileLoad'](tile);
    props['onViewportLoad']([tile]);
    const state = controller.getState()[0];
    expect(state.workingSet.rejectedTileKeys).toEqual(['3/1/0']);
    expect(state.scope?.isTileCoverageComplete).toBe(false);
  });
});
