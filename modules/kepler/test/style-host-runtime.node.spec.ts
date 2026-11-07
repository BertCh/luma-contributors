// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {CommandEncoder} from '@luma.gl/core';
import {NativeStyleRuntime} from '../../../examples/deck/kepler-compute/src/style-host-runtime';
import type {NativeStyleLayer} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import type {ResidentDatasetBinding} from '../../../examples/deck/kepler-compute/src/resident-layers';
import type {KeplerComputeSession} from '../src/session/compute-session';
import {KeplerResidentStyleCache} from '../src/styles/style-cache';
import type {KeplerResidentStyle, KeplerResidentStyleLegend} from '../src/styles/resident-style';
import type {KeplerResidentStylePlan} from '../src/styles/style-plan';

afterEach(() => vi.restoreAllMocks());

function makeReadableVector(format: string, values: Float32Array | Uint32Array) {
  return {
    format,
    length: values.length,
    data: [
      {
        length: values.length,
        byteOffset: 0,
        byteStride: 4,
        buffer: {readAsync: vi.fn(async () => new Uint8Array(values.slice().buffer))}
      }
    ]
  };
}

function setup() {
  const reserveMemory = vi.fn(() => ({release: vi.fn()}));
  const naturalBreaks = vi.fn((values: number[]) => [values.length]);
  const source = new Proxy(
    {},
    {
      get() {
        throw new Error('Source rows must not be touched');
      }
    }
  );
  const dataset = {
    id: 'sample',
    revision: 1,
    rowCount: 3,
    value: {
      source,
      columns: {
        value: makeReadableVector('float32', new Float32Array([1, NaN, 4])),
        category: {format: 'uint32'},
        unsupported: {format: undefined}
      },
      analytics: {
        dictionaries: {},
        validity: {value: makeReadableVector('uint32', new Uint32Array([1, 1, 1]))}
      },
      prepared: {dictionaries: {}}
    }
  };
  const fence = {signaled: Promise.resolve(), destroy: vi.fn()};
  const session = {
    registry: {getDataset: vi.fn(() => dataset), reserveMemory},
    device: {createFence: vi.fn(() => fence)}
  } as unknown as KeplerComputeSession;
  const output = {format: 'unorm8x4', length: 3, data: []} as unknown as GPUVector<'unorm8x4'>;
  const sourceMask = {format: 'uint32', length: 3, data: []} as unknown as GPUVector<'uint32'>;
  const binding: ResidentDatasetBinding = {selectionMask: sourceMask};
  const styles = new Map<
    string,
    {
      plan: KeplerResidentStylePlan;
      revision: number;
      style: KeplerResidentStyle;
      mask?: GPUVector<'uint32'>;
    }
  >();
  const encoded = vi.fn();
  const readLegend = vi.fn(
    async (
      plan: KeplerResidentStylePlan,
      revision: number
    ): Promise<KeplerResidentStyleLegend | null> => ({
      styleId: plan.id,
      revision,
      source: plan.source,
      scale: plan.scale,
      domain: [0, 10],
      domainStatus: 'valid',
      finiteCount: 3,
      numericalPolicy: 'float32',
      ...(plan.kind === 'color' ? {palette: plan.palette} : {}),
      ...(plan.domain.kind === 'selection'
        ? {selectionRevision: plan.domain.revision}
        : plan.domain.kind === 'fixed' && plan.domain.selectionRevision !== undefined
          ? {selectionRevision: plan.domain.selectionRevision}
          : {})
    })
  );
  const set = vi
    .spyOn(KeplerResidentStyleCache.prototype, 'set')
    .mockImplementation((plan, options) => {
      const previous = styles.get(plan.id);
      const record = {
        plan,
        revision: (previous?.revision ?? 0) + 1,
        style: undefined as unknown as KeplerResidentStyle,
        mask: options?.selectionMask
      };
      record.style = {
        colors: plan.kind === 'color' ? output : undefined,
        get currentRevision() {
          return record.revision;
        },
        get currentPlan() {
          return record.plan;
        },
        isCurrent: true,
        canUpdate: (next: KeplerResidentStylePlan, mask?: GPUVector<'uint32'>) =>
          next.column === record.plan.column &&
          next.domain.kind === record.plan.domain.kind &&
          mask === record.mask,
        encode: encoded,
        discardEncoding: vi.fn(),
        readLegend: () => readLegend(record.plan, record.revision)
      } as unknown as KeplerResidentStyle;
      styles.set(plan.id, record);
      return record.style;
    });
  const remove = vi.spyOn(KeplerResidentStyleCache.prototype, 'remove').mockResolvedValue();
  const destroy = vi.spyOn(KeplerResidentStyleCache.prototype, 'destroy').mockResolvedValue();
  const requestUpdate = vi.fn();
  const onUnsupported = vi.fn();
  const runtime = new NativeStyleRuntime({session, requestUpdate, onUnsupported, naturalBreaks});
  const layer: NativeStyleLayer = {
    id: 'points',
    type: 'point',
    config: {
      dataId: 'sample',
      color: [10, 20, 30],
      colorField: {name: 'value'},
      colorScale: 'quantile',
      colorDomain: [0, 10],
      visConfig: {colorRange: {colors: ['#000000', '#ffffff']}, radius: 3}
    }
  };
  return {
    runtime,
    naturalBreaks,
    reserveMemory,
    session,
    dataset,
    binding,
    layer,
    set,
    remove,
    destroy,
    encoded,
    readLegend,
    output,
    requestUpdate,
    onUnsupported,
    fence
  };
}

test('host translates numeric source domains without source scans or repeated style uploads', async () => {
  const state = setup();
  const listener = vi.fn();
  const unsubscribe = state.runtime.subscribe(listener);
  const first = state.runtime.resolve(state.layer, state.binding);
  expect(first.channels?.colors).toBe(state.output);
  expect(state.set.mock.calls[0][0].domain).toEqual({kind: 'source'});
  for (let index = 0; index < 5; index++)
    state.runtime.resolve(state.layer, state.binding, {zoom: index});
  expect(state.set).toHaveBeenCalledTimes(1);
  expect(state.runtime.getLegend('points', 'fillColor')?.legend).toBeUndefined();
  state.runtime.encode({} as CommandEncoder);
  await state.runtime.readLegends();
  await state.runtime.readLegends();
  expect(state.readLegend).toHaveBeenCalledTimes(1);
  expect(state.runtime.getLegend('points', 'fillColor')?.legend?.domain).toEqual([0, 10]);
  expect(listener).toHaveBeenCalled();
  unsubscribe();
  await state.runtime.destroy(Promise.resolve());
});

test('native Jenks requires current explicit controls without reading source rows or native empty domains', async () => {
  const state = setup();
  state.layer.config.colorScale = 'jenks';
  state.layer.config.colorDomain = [];
  const controls = {
    source: {datasetId: 'sample', revision: 1},
    column: 'value',
    classCount: 2,
    values: [4]
  };
  const missing = state.runtime.resolve(state.layer, state.binding);
  // Missing explicit controls derive thresholds after submission via one bounded accounted read.
  expect(missing.enabled).toBe(false);
  expect(missing.unsupportedReason).toBeUndefined();
  expect(state.set).not.toHaveBeenCalled();
  await state.runtime.readLegends();
  await state.runtime.readLegends();
  expect(state.naturalBreaks).toHaveBeenCalledTimes(1);
  expect(state.naturalBreaks).toHaveBeenCalledWith([1, 4], 2);
  expect(state.reserveMemory).toHaveBeenCalledWith(3 * 16 + 3 * 16, 'cpu', 'readback');
  expect(state.runtime.getDerivedJenksThresholds('points', 'fillColor')).toMatchObject({
    column: 'value',
    classCount: 2,
    values: [2]
  });
  expect(state.runtime.getDerivedJenksThresholds('points', 'strokeColor')).toBeUndefined();
  const derived = state.runtime.resolve(state.layer, state.binding);
  expect(derived.enabled).not.toBe(false);
  expect(state.set.mock.calls.at(-1)?.[0].domain).toEqual({kind: 'fixed', values: [2]});
  state.runtime.resolve(state.layer, state.binding);
  expect(state.naturalBreaks).toHaveBeenCalledTimes(1);
  const global = state.runtime.resolve(state.layer, state.binding, {
    jenksThresholds: {fillColor: controls}
  });
  expect(global.enabled).not.toBe(false);
  expect(state.set.mock.calls.at(-1)?.[0]).toMatchObject({
    scale: 'threshold',
    domain: {kind: 'fixed', values: [4]}
  });
  const domainMask = {format: 'uint32'} as GPUVector<'uint32'>;
  const selectedOptions = {
    domainMask,
    domainRevision: 2,
    jenksThresholds: {fillColor: {...controls, selectionRevision: 2}}
  };
  state.runtime.resolve(state.layer, state.binding, selectedOptions);
  expect(state.set.mock.calls.at(-1)?.[0].domain.kind).toBe('fixed');
  await state.runtime.readLegends();
  expect(state.runtime.getLegend('points', 'fillColor')).toMatchObject({
    expected: {selectionRevision: 2},
    legend: {selectionRevision: 2}
  });
  const stale = state.runtime.resolve(state.layer, state.binding, {
    ...selectedOptions,
    domainRevision: 3
  });
  expect(stale.enabled).toBe(false);
  expect(stale.unsupportedReason).toContain('domain selection revision');
  expect(state.runtime.getLegend('points', 'fillColor')?.legend).toBeUndefined();
  const recovered = state.runtime.resolve(state.layer, state.binding, {
    domainMask,
    domainRevision: 3,
    jenksThresholds: {fillColor: {...controls, values: [7], selectionRevision: 3}}
  });
  expect(recovered.enabled).not.toBe(false);
  expect(state.set.mock.calls.at(-1)?.[0].domain).toEqual({
    kind: 'fixed',
    values: [7],
    selectionRevision: 3
  });
  await state.runtime.destroy(Promise.resolve());
});

test('domain masks remain independent from display masks and invalidate bounded legends by revision', async () => {
  const state = setup();
  const domainMask = {format: 'uint32'} as GPUVector<'uint32'>;
  const first = state.runtime.resolve(state.layer, state.binding, {domainMask, domainRevision: 1});
  expect(first.selectionMask).toBe(state.binding.selectionMask);
  expect(state.set.mock.calls[0][1]?.selectionMask).toBe(domainMask);
  expect(state.set.mock.calls[0][0].domain).toEqual({kind: 'selection', revision: 1});
  state.runtime.encode({} as CommandEncoder);
  await state.runtime.readLegends();
  state.runtime.resolve(state.layer, state.binding, {domainMask, domainRevision: 2});
  expect(state.runtime.getLegend('points', 'fillColor')?.legend).toBeUndefined();
  expect(state.runtime.getLegend('points', 'fillColor')?.expected.selectionRevision).toBe(2);
  await state.runtime.readLegends();
  expect(state.readLegend).toHaveBeenCalledTimes(2);
  expect(state.runtime.getLegend('points', 'fillColor')?.legend?.selectionRevision).toBe(2);
  expect(state.fence.destroy).toHaveBeenCalled();
  await state.runtime.destroy(Promise.resolve());
});

test('derived scalar log domains report invalid status and recover without blocking GPU encoding', async () => {
  const state = setup();
  state.layer.config.colorField = null;
  state.layer.config.sizeField = {name: 'value'};
  state.layer.config.sizeScale = 'log';
  state.layer.config.sizeDomain = [-10, 10];
  const domainMask = {format: 'uint32'} as GPUVector<'uint32'>;
  state.readLegend.mockResolvedValueOnce({
    styleId: 'points:size',
    revision: 1,
    source: {datasetId: 'sample', revision: 1},
    selectionRevision: 1,
    scale: 'log',
    domain: [-10, 10],
    domainStatus: 'invalid-log-domain',
    finiteCount: 3,
    range: [0, 50],
    numericalPolicy: 'float32'
  });
  const options = {domainMask, domainRevision: 1};
  expect(state.runtime.resolve(state.layer, state.binding, options).enabled).not.toBe(false);
  expect(state.set.mock.calls[0][0].domain).toEqual({kind: 'selection', revision: 1});
  state.runtime.encode({} as CommandEncoder);
  await state.runtime.readLegends();
  expect(state.readLegend).toHaveBeenCalledTimes(1);
  expect(state.runtime.getLegend('points', 'size')?.legend?.domainStatus).toBe(
    'invalid-log-domain'
  );
  const invalid = state.runtime.resolve(state.layer, state.binding, options);
  expect(invalid.enabled).toBe(false);
  expect(invalid.unsupportedReason).toContain('size logarithmic domain crosses or touches zero');
  await Promise.resolve();
  expect(state.onUnsupported).toHaveBeenCalledWith(
    'points',
    expect.stringContaining('size logarithmic domain')
  );
  state.runtime.encode({} as CommandEncoder);
  expect(state.encoded).toHaveBeenCalledTimes(2);
  state.readLegend.mockResolvedValueOnce({
    styleId: 'points:size',
    revision: 2,
    source: {datasetId: 'sample', revision: 1},
    selectionRevision: 2,
    scale: 'log',
    domain: [-100, -1],
    domainStatus: 'valid',
    finiteCount: 2,
    range: [0, 50],
    numericalPolicy: 'float32'
  });
  state.runtime.resolve(state.layer, state.binding, {...options, domainRevision: 2});
  await state.runtime.readLegends();
  const recovered = state.runtime.resolve(state.layer, state.binding, {
    ...options,
    domainRevision: 2
  });
  expect(recovered.enabled).not.toBe(false);
  expect(recovered.unsupportedReason).toBeUndefined();
  expect(state.runtime.getLegend('points', 'size')?.legend?.domain).toEqual([-100, -1]);
  await Promise.resolve();
  expect(state.onUnsupported.mock.calls.at(-1)).toEqual(['points', null]);
  await state.runtime.destroy(Promise.resolve());
});

test('analysis fill overrides keep precedence over invalid native logarithmic color domains', async () => {
  const state = setup();
  state.layer.config.colorScale = 'log';
  const binding = {...state.binding, colorOverride: state.output};
  state.readLegend.mockResolvedValueOnce({
    styleId: 'points:fillColor',
    revision: 1,
    source: {datasetId: 'sample', revision: 1},
    scale: 'log',
    domain: [-1, 1],
    domainStatus: 'invalid-log-domain',
    finiteCount: 3,
    palette: [0xff000000, 0xffffffff],
    numericalPolicy: 'float32'
  });
  state.runtime.resolve(state.layer, binding);
  expect(state.set.mock.calls[0][0].domain).toEqual({kind: 'source'});
  await state.runtime.readLegends();
  const resolved = state.runtime.resolve(state.layer, binding);
  expect(resolved.colorOverride).toBe(state.output);
  expect(resolved.enabled).not.toBe(false);
  expect(resolved.unsupportedReason).toBeUndefined();
  await state.runtime.destroy(Promise.resolve());
});

test('unsupported source precision hides a previous legend and preserves analysis override precedence', async () => {
  const state = setup();
  const override = {format: 'unorm8x4'} as GPUVector<'unorm8x4'>;
  const first = state.runtime.resolve(state.layer, {...state.binding, colorOverride: override});
  expect(first.colorOverride).toBe(override);
  await state.runtime.readLegends();
  state.layer.config.colorField = {name: 'unsupported'};
  const unsupported = state.runtime.resolve(state.layer, state.binding);
  expect(unsupported.enabled).toBe(false);
  expect(unsupported.unsupportedReason).toContain('No resident source column');
  expect(state.runtime.getLegend('points', 'fillColor')?.legend).toBeUndefined();
  await Promise.resolve();
  expect(state.onUnsupported).toHaveBeenCalledWith(
    'points',
    expect.stringContaining('No resident source column')
  );
  state.layer.config.colorField = {name: 'category'};
  state.layer.config.colorScale = 'ordinal';
  state.layer.config.colorDomain = [2, 0, 1];
  const categorical = state.runtime.resolve(state.layer, state.binding, {
    dictionaries: {category: [0, 1, 2]}
  });
  expect(categorical.enabled).not.toBe(false);
  expect(state.set.mock.calls.at(-1)?.[0].legend?.codes).toEqual([2, 0, 1]);
  await state.runtime.destroy(Promise.resolve());
});

test('obsolete readbacks cannot publish and removed layers retire against caller completion', async () => {
  const state = setup();
  let complete!: (legend: KeplerResidentStyleLegend | null) => void;
  state.readLegend.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        complete = resolve;
      })
  );
  state.runtime.resolve(state.layer, state.binding);
  const pending = state.runtime.readLegends();
  const completion = Promise.resolve();
  state.runtime.retainLayers([], completion);
  complete({
    styleId: 'points:fillColor',
    revision: 1,
    source: {datasetId: 'sample', revision: 1},
    scale: 'quantile',
    domain: [0, 10],
    domainStatus: 'valid',
    finiteCount: 3,
    numericalPolicy: 'float32'
  });
  await pending;
  expect(state.runtime.getLegend('points', 'fillColor')).toBeNull();
  expect(state.remove).toHaveBeenCalledWith('points:fillColor', completion);
  await state.runtime.destroy(completion);
  expect(state.destroy).toHaveBeenCalledWith(completion);
});

test('a not-yet-submitted legend retries once submitted and clears its pending reason', async () => {
  const state = setup();
  state.readLegend.mockResolvedValueOnce(null);
  state.runtime.resolve(state.layer, state.binding);
  await state.runtime.readLegends();
  expect(state.runtime.getLegend('points', 'fillColor')?.reason).toContain('pending');
  state.runtime.encode({} as CommandEncoder);
  await state.runtime.readLegends();
  expect(state.runtime.getLegend('points', 'fillColor')?.reason).toBeUndefined();
  expect(state.runtime.getLegend('points', 'fillColor')?.legend?.revision).toBe(1);
  await state.runtime.readLegends();
  expect(state.readLegend).toHaveBeenCalledTimes(2);
  await state.runtime.destroy(Promise.resolve());
});

test('dataset retirement waits its resource fence while preserving other dataset styles', async () => {
  const state = setup();
  vi.mocked(state.session.registry.getDataset).mockImplementation(id =>
    id === 'sample' ? (state.dataset as never) : ({...state.dataset, id} as never)
  );
  state.runtime.resolve(state.layer, state.binding);
  const otherLayer = {
    ...state.layer,
    id: 'other-points',
    config: {...state.layer.config, dataId: 'other'}
  };
  state.runtime.resolve(otherLayer, state.binding);
  let complete!: () => void;
  const completion = new Promise<void>(resolve => {
    complete = resolve;
  });
  state.remove.mockImplementation((_id, fence) => Promise.resolve(fence).then(() => undefined));
  let finished = false;
  const retirement = state.runtime.retireDataset('sample', completion).then(() => {
    finished = true;
  });
  await Promise.resolve();
  expect(finished).toBe(false);
  expect(state.runtime.getLegend('points', 'fillColor')).toBeNull();
  expect(state.runtime.getLegend('other-points', 'fillColor')).not.toBeNull();
  expect(state.remove).toHaveBeenCalledWith('points:fillColor', completion);
  expect(state.remove).not.toHaveBeenCalledWith('other-points:fillColor', expect.anything());
  complete();
  await retirement;
  expect(finished).toBe(true);
  await state.runtime.destroy(Promise.resolve());
});

test('explicit native threshold maps keep fixed controls across dynamic-domain revisions', async () => {
  const state = setup();
  state.layer.config.colorScale = 'custom';
  state.layer.config.colorDomain = [-100, 100];
  state.layer.config.visConfig.colorRange = {
    colors: ['#ff0000', '#00ff00', '#0000ff'],
    colorMap: [
      [10, '#ff0000'],
      [20, '#00ff00'],
      [null, '#0000ff']
    ]
  };
  const domainMask = {format: 'uint32'} as GPUVector<'uint32'>;
  const resolved = state.runtime.resolve(state.layer, state.binding, {
    domainMask,
    domainRevision: 1
  });
  expect(resolved.enabled).not.toBe(false);
  const plan = state.set.mock.calls[0][0];
  expect(plan.domain).toEqual({kind: 'fixed', values: [10, 10, 20, 20]});
  expect(plan.kind).toBe('color');
  if (plan.kind !== 'color') throw new Error('Expected color plan');
  expect(plan.scale).toBe('quantile');
  expect(plan.interpolation).toBe('step');
  expect(plan.palette).toEqual([0xff0000ff, 0xff00ff00, 0xffff0000]);
  state.runtime.resolve(state.layer, state.binding, {domainMask, domainRevision: 2});
  expect(state.set).toHaveBeenCalledTimes(1);
  expect(
    state.runtime.getLegend('points', 'fillColor')?.expected.selectionRevision
  ).toBeUndefined();
  await state.runtime.destroy(Promise.resolve());
});
