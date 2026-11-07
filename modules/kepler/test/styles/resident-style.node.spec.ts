// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {
  KeplerResidentStyle,
  KeplerResidentStyleCache,
  validateKeplerResidentStylePlan,
  type KeplerResidentStylePlan
} from '../../src/styles';

function fixture() {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'data',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array())})
  });
  const plan: KeplerResidentStylePlan = {
    id: 'point-fill',
    source: {datasetId: dataset.id, revision: dataset.revision},
    column: 'value',
    kind: 'color',
    scale: 'linear',
    palette: [0xff000000, 0xffffffff],
    domain: {kind: 'fixed', values: [0, 10]},
    outsideDomain: 'extrapolate'
  };
  return {
    device,
    session,
    dataset,
    plan,
    cleanup: async () => {
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}

test('resident style validates explicit precision, native descending color domains and category order', () => {
  const source = {datasetId: 'data', revision: 1};
  const plan: KeplerResidentStylePlan = {
    id: 'radius',
    source,
    column: 'value',
    kind: 'scalar',
    scale: 'point',
    range: [0, 10],
    domain: {kind: 'fixed', values: [2, 0, 1]}
  };
  expect(() => validateKeplerResidentStylePlan(plan)).not.toThrow();
  expect(() =>
    validateKeplerResidentStylePlan({...plan, domain: {kind: 'fixed', values: [0, 0]}})
  ).toThrow(/domain/);
  expect(() =>
    validateKeplerResidentStylePlan({
      ...plan,
      scale: 'linear',
      domain: {kind: 'fixed', values: [0, 0.1]}
    })
  ).toThrow(/normalization/);
  expect(() =>
    validateKeplerResidentStylePlan({
      id: 'fill',
      source,
      column: 'value',
      kind: 'color',
      scale: 'sqrt',
      domain: {kind: 'fixed', values: [10, 0, -10]},
      palette: [0, 1, 2],
      outsideDomain: 'extrapolate'
    })
  ).not.toThrow();
});

test('explicit threshold controls retain boundaries and provenance in bounded legends', async () => {
  const context = fixture();
  const plan: KeplerResidentStylePlan = {
    ...context.plan,
    kind: 'color',
    scale: 'threshold',
    domain: {kind: 'fixed', values: [4, 4], selectionRevision: 7},
    palette: [0xff000000, 0xff0000ff, 0xffffffff],
    outsideDomain: 'clamp'
  };
  const style = new KeplerResidentStyle({...context, plan, workingMemoryBudgetBytes: 16384});
  style.encode(context.device.commandEncoder);
  expect(await style.readLegend()).toMatchObject({
    scale: 'threshold',
    domain: [4, 4],
    palette: plan.palette,
    selectionRevision: 7
  });
  expect(style.readbackByteLength).toBe(16);
  style.update({...plan, domain: {kind: 'fixed', values: []}, palette: [0xffffffff]});
  style.encode(context.device.commandEncoder);
  expect(await style.readLegend()).toMatchObject({domain: [], palette: [0xffffffff]});
  for (const invalid of [
    {...plan, domain: {kind: 'source'} as const},
    {...plan, palette: [0xffffffff]},
    {...plan, domain: {kind: 'fixed', values: [4, 3]} as const},
    {...plan, domain: {kind: 'fixed', values: [4, 4], selectionRevision: 0} as const}
  ])
    expect(() => validateKeplerResidentStylePlan(invalid)).toThrow();
  await style.destroy(Promise.resolve());
  await context.cleanup();
});

test('style capacity rejection happens before allocating controls or outputs', async () => {
  const context = fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  const allocate = vi.spyOn(context.device, 'createBuffer');
  expect(
    () => new KeplerResidentStyle({...context, workingMemoryBudgetBytes: 2 ** 20 + 1})
  ).toThrow(/budget/);
  expect(allocate).not.toHaveBeenCalled();
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  allocate.mockRestore();
  await context.cleanup();
});

test('scalar log styles reject zero and mixed-sign domains before allocating GPU resources', async () => {
  const context = fixture();
  const plan: KeplerResidentStylePlan = {
    id: 'radius',
    source: context.plan.source,
    column: 'value',
    kind: 'scalar',
    scale: 'log',
    range: [1, 10],
    domain: {kind: 'fixed', values: [1, 100]}
  };
  const baseline = context.session.registry.memoryUsage.totalBytes;
  for (const domain of [{kind: 'source'} as const, {kind: 'selection', revision: 1} as const]) {
    expect(() => validateKeplerResidentStylePlan({...plan, domain})).not.toThrow();
    expect(() =>
      validateKeplerResidentStylePlan({...context.plan, scale: 'log', domain})
    ).not.toThrow();
  }
  const allocate = vi.spyOn(context.device, 'createBuffer');
  for (const values of [
    [0, 10],
    [-10, 0],
    [-10, 10],
    [10, -10]
  ]) {
    expect(
      () =>
        new KeplerResidentStyle({
          ...context,
          plan: {...plan, domain: {kind: 'fixed', values}},
          workingMemoryBudgetBytes: 16384
        })
    ).toThrow(/same-sign/);
  }
  for (const values of [
    [1, 100],
    [100, 1],
    [-100, -1],
    [-1, -100],
    [2, 2],
    [-2, -2]
  ]) {
    expect(() =>
      validateKeplerResidentStylePlan({
        ...plan,
        domain: {kind: 'fixed', values}
      })
    ).not.toThrow();
  }
  expect(allocate).not.toHaveBeenCalled();
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  allocate.mockRestore();
  await context.cleanup();
});

test('empty styles replay stable controls, bound legends and reject stale legend publication', async () => {
  const context = fixture();
  const style = new KeplerResidentStyle({...context, workingMemoryBudgetBytes: 16384});
  const colors = style.colors;
  expect(await style.readLegend()).toBeNull();
  style.encode(context.device.commandEncoder);
  const pending = style.readLegend();
  const excess = style.readLegend();
  style.update({...context.plan, domain: {kind: 'fixed', values: [1, 20]}});
  expect(await excess).toBeNull();
  expect(await pending).toBeNull();
  expect(style.colors).toBe(colors);
  style.encode(context.device.commandEncoder);
  expect((await style.readLegend())?.domain).toEqual([1, 20]);
  expect(style.readbackByteLength).toBe(16);
  const revision = style.currentRevision;
  expect(() => style.update({...context.plan, domain: {kind: 'fixed', values: [1, 0.1]}})).toThrow(
    /normalization/
  );
  expect(style.currentRevision).toBe(revision);
  expect(style.currentPlan.domain).toEqual({kind: 'fixed', values: [1, 20]});
  await style.destroy(Promise.resolve());
  await context.cleanup();
});

test('style outputs and source lease survive retirement until consumer fences settle', async () => {
  const context = fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  const style = new KeplerResidentStyle({...context, workingMemoryBudgetBytes: 16384});
  const output = style.acquireOutputs();
  let resolveFence!: () => void;
  const fence = new Promise<void>(resolve => {
    resolveFence = resolve;
  });
  const completion = style.destroy(Promise.resolve());
  const release = output.release(fence);
  expect(output.isCurrent()).toBe(false);
  await Promise.resolve();
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline + 16384);
  resolveFence();
  await Promise.all([completion, release]);
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  await context.cleanup();
});

test('style cache reuses controls and requires an explicit fence for topology replacement', async () => {
  const context = fixture();
  const cache = new KeplerResidentStyleCache({
    session: context.session,
    workingMemoryBudgetBytesPerStyle: 16384
  });
  const first = cache.set(context.plan);
  const changed = {...context.plan, domain: {kind: 'fixed' as const, values: [2, 8]}};
  expect(cache.set(changed)).toBe(first);
  const derived = {...context.plan, domain: {kind: 'source' as const}};
  expect(() => cache.set(derived)).toThrow(/fence/);
  expect(cache.get(context.plan.id)).toBe(first);
  const second = cache.set(derived, {retireAfter: Promise.resolve()});
  expect(second).not.toBe(first);
  expect(first.isCurrent).toBe(false);
  await cache.destroy(Promise.resolve());
  await context.cleanup();
});

test('dataset replacement invalidates cached style publication without destroying a leased source early', async () => {
  const context = fixture();
  const style = new KeplerResidentStyle({...context, workingMemoryBudgetBytes: 16384});
  context.session.importArrowDataset({
    id: 'data',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array())})
  });
  expect(style.isCurrent).toBe(false);
  expect(() => style.encode(context.device.commandEncoder)).toThrow(/stale/);
  expect(await style.readLegend()).toBeNull();
  await style.destroy(Promise.resolve());
  await context.cleanup();
});

test('style source and population mismatches fail before reserving or allocating style resources', async () => {
  const context = fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  expect(
    () =>
      new KeplerResidentStyle({
        ...context,
        plan: {...context.plan, source: {...context.plan.source, revision: 99}},
        workingMemoryBudgetBytes: 16384
      })
  ).toThrow(/revision/);
  expect(
    () =>
      new KeplerResidentStyle({
        ...context,
        plan: {...context.plan, domain: {kind: 'selection', revision: 1}},
        workingMemoryBudgetBytes: 16384
      })
  ).toThrow(/mask/);
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  await context.cleanup();
});

test('cache retains asynchronous retirement errors while still cleaning every source and style reservation', async () => {
  const context = fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  const cache = new KeplerResidentStyleCache({
    session: context.session,
    workingMemoryBudgetBytesPerStyle: 16384
  });
  cache.set(context.plan);
  cache.set(
    {...context.plan, domain: {kind: 'source'}},
    {retireAfter: Promise.reject(new Error('Injected submission failure'))}
  );
  await new Promise<void>(resolve => {
    setTimeout(resolve, 0);
  });
  await expect(cache.destroy(Promise.resolve())).rejects.toThrow('Injected submission failure');
  expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
  await context.cleanup();
});

test('scalar identity inversion is explicit and supports GPU-derived endpoints', () => {
  const plan: KeplerResidentStylePlan = {
    id: 'height',
    source: {datasetId: 'data', revision: 1},
    column: 'height',
    kind: 'scalar',
    scale: 'identity',
    invert: true,
    range: [0, 1],
    domain: {kind: 'source'}
  };
  expect(() => validateKeplerResidentStylePlan(plan)).not.toThrow();
  expect(() =>
    validateKeplerResidentStylePlan({...plan, domain: {kind: 'selection', revision: 2}})
  ).not.toThrow();
  expect(() => validateKeplerResidentStylePlan({...plan, scale: 'linear'})).toThrow(/identity/);
});
