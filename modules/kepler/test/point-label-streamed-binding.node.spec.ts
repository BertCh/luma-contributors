// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, beforeEach, describe, expect, it, vi} from 'vitest';

import * as arrow from 'apache-arrow';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import type {KeplerArrowDataset, KeplerComputeSession} from '../src/session/compute-session';
import type {GPUVector} from '@luma.gl/graph/gpu-data';

const mocked = vi.hoisted(() => {
  vi.resetModules();
  return {
    streamed: [] as Array<{
      props: {
        chunks: Array<{rowCount: number; getText(rowIndex: number): string}>;
        collisionPositions: unknown;
      };
      complete(): void;
      destroy: ReturnType<typeof vi.fn>;
    }>,
    runtimes: 0
  };
});
vi.mock('../../../examples/deck/kepler-compute/gpu-layers/index', () => ({
  GPUPointLabelLayer: class {
    constructor(readonly props: object) {}
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/label-runtime', () => ({
  makePointLabelRuntime: () => {
    mocked.runtimes++;
    throw new Error('The capped runtime must not be used for streamed sources.');
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/label-streamed-runtime', () => ({
  StreamedPointLabelRuntime: class {
    ready: Promise<void>;
    complete!: () => void;
    destroy = vi.fn(async (completion: PromiseLike<unknown>) => {
      await completion;
    });
    constructor(
      _device: unknown,
      readonly props: {
        chunks: Array<{rowCount: number; getText(rowIndex: number): string}>;
        collisionPositions: unknown;
      }
    ) {
      this.ready = new Promise<void>(resolve => {
        this.complete = resolve;
      });
      mocked.streamed.push(this as never);
    }
    resolve({label}: {label: {collisionEnabled?: boolean}}) {
      return {layerProps: {collisionEnabled: label.collisionEnabled}};
    }
  }
}));

import {PointLabelBindingRuntime} from '../../../examples/deck/kepler-compute/src/point-label-binding-runtime';

const ROW_COUNT = 100001;

function fixture() {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64 << 20});
  const source = new arrow.Table({
    label: arrow.vectorFromArray(
      Array.from({length: ROW_COUNT}, (_value, index) => `row ${index}`),
      new arrow.Utf8()
    )
  });
  const dataset = registry.registerDataset({
    id: 'labels',
    value: {source} as KeplerArrowDataset,
    batches: [{id: 'source', rowCount: ROW_COUNT}],
    resources: [
      {
        id: 'source-memory',
        value: source,
        byteLength: 128,
        memoryKind: 'cpu',
        ownership: 'owned',
        destroy: vi.fn()
      }
    ]
  });
  const device = {
    isLost: false,
    lost: new Promise(() => {}),
    createFence: () => ({signaled: Promise.resolve(), destroy: vi.fn()})
  };
  const session = {registry, device, deviceLost: false} as unknown as KeplerComputeSession;
  const changed = vi.fn();
  const owner = new PointLabelBindingRuntime(session, changed);
  const positions = {format: 'float32x2'} as GPUVector<'float32x2'>;
  const options = {
    dataset,
    layerId: 'point',
    labels: [{field: {name: 'label'}, size: 18, collisionEnabled: true}],
    positions,
    selectionMask: {format: 'uint32'} as GPUVector<'uint32'>,
    radius: 1,
    radiusScale: 1,
    mapState: {longitude: 0, latitude: 0, zoom: 2, width: 800, height: 600}
  };
  return {owner, options, changed, positions};
}

describe('streamed point labels above the glyph cap', () => {
  beforeEach(() => {
    mocked.streamed.length = 0;
    mocked.runtimes = 0;
  });

  it('streams a GPU-collision source above 100,000 rows instead of refusing it', async () => {
    const {owner, options, changed, positions} = fixture();
    expect(owner.resolve(options)).toEqual({layers: [], pending: true});
    expect(mocked.streamed).toHaveLength(1);
    expect(mocked.runtimes).toBe(0);
    const [streamed] = mocked.streamed;
    // The source is read lazily and unchanged: no row, glyph or byte budget applies to it.
    expect(streamed.props.chunks.map(chunk => chunk.rowCount)).toEqual([ROW_COUNT]);
    expect(streamed.props.chunks[0].getText(99999)).toBe('row 99999');
    expect(streamed.props.collisionPositions).toBe(positions);
    streamed.complete();
    await new Promise(resolve => setTimeout(resolve));
    expect(changed).toHaveBeenCalledTimes(1);
    const result = owner.resolve(options);
    expect(result.layers).toHaveLength(1);
    expect(result.reason).toBeUndefined();
    // GPU collision owns visibility, so the label layer receives no per-row CPU mask.
    expect(
      (result.layers[0] as unknown as {props: {collisionEnabled: boolean}}).props.collisionEnabled
    ).toBe(false);
    await owner.destroy(Promise.resolve());
    expect(streamed.destroy).toHaveBeenCalledTimes(1);
  });

  it('still refuses a source above the cap when collision is off', () => {
    const {owner, options} = fixture();
    const result = owner.resolve({
      ...options,
      labels: [{field: {name: 'label'}, size: 18, collisionEnabled: false}]
    });
    expect(result.layers).toEqual([]);
    expect(mocked.streamed).toHaveLength(0);
    expect(result.reason ?? '').toMatch(/at most 100000/);
  });

  it('streams with globe map state so collision can project on the sphere', async () => {
    const {owner, options} = fixture();
    owner.resolve({
      ...options,
      mapState: {...options.mapState, globe: {enabled: true}}
    });
    expect(mocked.streamed).toHaveLength(1);
    mocked.streamed[0].complete();
    await new Promise(resolve => setTimeout(resolve));
    expect(
      owner.resolve({...options, mapState: {...options.mapState, globe: {enabled: true}}}).layers
    ).toHaveLength(1);
    await owner.destroy(Promise.resolve());
  });
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('../../../examples/deck/kepler-compute/gpu-layers/index');
  vi.doUnmock('../../../examples/deck/kepler-compute/src/label-runtime');
  vi.doUnmock('../../../examples/deck/kepler-compute/src/label-streamed-runtime');
  vi.resetModules();
});
