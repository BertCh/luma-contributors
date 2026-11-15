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
    runtimes: [] as Array<{
      ready: Promise<void>;
      complete(): void;
      destroy: ReturnType<typeof vi.fn>;
    }>
  };
});
vi.mock('../../../examples/deck/kepler-compute/gpu-layers/index', () => ({
  GPUPointLabelLayer: class {
    constructor(readonly props: object) {}
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/label-runtime', () => ({
  makePointLabelRuntime: () => {
    let complete!: () => void;
    const ready = new Promise<void>(resolve => {
      complete = resolve;
    });
    const runtime = {
      ready,
      complete,
      destroy: vi.fn(async (completion: PromiseLike<unknown>) => {
        await completion;
      }),
      resolve: ({label}: {label: {size?: number}}) => ({layerProps: {size: label.size}})
    };
    mocked.runtimes.push(runtime);
    return runtime;
  }
}));

import {PointLabelBindingRuntime} from '../../../examples/deck/kepler-compute/src/point-label-binding-runtime';

function fixture() {
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 64 << 20});
  const source = new arrow.Table({
    label: arrow.vectorFromArray(['one', 'two'], new arrow.Utf8()),
    alternate: arrow.vectorFromArray(['three', 'four'], new arrow.Utf8())
  });
  const sourceDestroyed = vi.fn();
  const dataset = registry.registerDataset({
    id: 'labels',
    value: {source} as KeplerArrowDataset,
    batches: [{id: 'source', rowCount: 2}],
    resources: [
      {
        id: 'source-memory',
        value: source,
        byteLength: 128,
        memoryKind: 'cpu',
        ownership: 'owned',
        destroy: sourceDestroyed
      }
    ]
  });
  let completeFence!: () => void;
  const signaled = new Promise<void>(resolve => {
    completeFence = resolve;
  });
  const fenceDestroyed = vi.fn();
  const device = {
    isLost: false,
    lost: new Promise(() => {}),
    createFence: () => ({signaled, destroy: fenceDestroyed})
  };
  const session = {registry, device, deviceLost: false} as unknown as KeplerComputeSession;
  const changed = vi.fn();
  const owner = new PointLabelBindingRuntime(session, changed);
  const options = {
    dataset,
    layerId: 'point',
    labels: [{field: {name: 'label'}, size: 18}],
    positions: {format: 'float32x2'} as GPUVector<'float32x2'>,
    selectionMask: {format: 'uint32'} as GPUVector<'uint32'>,
    radius: 1,
    radiusScale: 1,
    mapState: {longitude: 0, latitude: 0, zoom: 2}
  };
  return {owner, options, registry, changed, sourceDestroyed, completeFence, fenceDestroyed};
}

describe('point label host ownership', () => {
  beforeEach(() => {
    mocked.runtimes.length = 0;
  });

  it('reuses a field/font variant across display controls and retires replacement after submission', async () => {
    const {owner, options, changed, completeFence, fenceDestroyed} = fixture();
    expect(owner.resolve(options)).toEqual({layers: [], pending: true});
    mocked.runtimes[0].complete();
    await Promise.resolve();
    expect(changed).toHaveBeenCalledTimes(1);
    expect(
      owner.resolve({...options, labels: [{field: {name: 'label'}, size: 22}]}).layers
    ).toHaveLength(1);
    expect(mocked.runtimes).toHaveLength(1);
    owner.resolve({...options, labels: [{field: {name: 'alternate'}, size: 22}]});
    expect(mocked.runtimes).toHaveLength(2);
    expect(mocked.runtimes[0].destroy).not.toHaveBeenCalled();
    owner.afterRender();
    expect(mocked.runtimes[0].destroy).toHaveBeenCalledTimes(1);
    expect(fenceDestroyed).not.toHaveBeenCalled();
    completeFence();
    await owner.destroy(Promise.resolve());
    expect(fenceDestroyed).toHaveBeenCalledTimes(1);
  });

  it('admits several native label entries as independent variants and retires only removed ones', async () => {
    const {owner, options, completeFence} = fixture();
    const labels = [
      {field: {name: 'label'}, size: 18, anchor: 'start', alignment: 'top'},
      {field: {name: 'alternate'}, size: 12, anchor: 'end', alignment: 'bottom'}
    ];
    expect(owner.resolve({...options, labels})).toEqual({layers: [], pending: true});
    expect(mocked.runtimes).toHaveLength(2);
    mocked.runtimes.forEach(runtime => runtime.complete());
    await Promise.resolve();
    const result = owner.resolve({...options, labels});
    expect(
      result.layers.map(layer => (layer as unknown as {props: {id: string}}).props.id)
    ).toEqual(['point-resident-point-label-0', 'point-resident-point-label-1']);
    owner.resolve({...options, labels: [labels[0]]});
    owner.afterRender();
    expect(mocked.runtimes[0].destroy).not.toHaveBeenCalled();
    expect(mocked.runtimes[1].destroy).toHaveBeenCalledTimes(1);
    completeFence();
    await owner.destroy(Promise.resolve());
  });

  it('ignores retired atlas completion and retains source ownership until the render fence', async () => {
    const {owner, options, registry, changed, sourceDestroyed, completeFence} = fixture();
    owner.resolve(options);
    owner.resolve({...options, labels: []});
    owner.afterRender();
    registry.removeDataset(options.dataset.id);
    mocked.runtimes[0].complete();
    await Promise.resolve();
    expect(changed).not.toHaveBeenCalled();
    expect(sourceDestroyed).not.toHaveBeenCalled();
    completeFence();
    await owner.destroy(Promise.resolve());
    expect(sourceDestroyed).toHaveBeenCalledTimes(1);
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('../../../examples/deck/kepler-compute/gpu-layers/index');
  vi.doUnmock('../../../examples/deck/kepler-compute/src/label-runtime');
  vi.resetModules();
});
