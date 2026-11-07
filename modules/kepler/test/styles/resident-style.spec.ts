// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {KeplerResidentStyle, type KeplerResidentStylePlan} from '../../src/styles';

const RED = 0xff0000ff;
const GREEN = 0xff00ff00;
const BLUE = 0xffff0000;

async function fixture() {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for resident style verification');
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const source = new arrow.Table({
    value: arrow.vectorFromArray([0, 1, null, 2], new arrow.Float32()),
    category: arrow.vectorFromArray(['z', 'a', null, 'z'], new arrow.Utf8())
  }).concat(
    new arrow.Table({
      value: arrow.vectorFromArray([100, 101, 102, 103], new arrow.Float32()),
      category: arrow.vectorFromArray(['a', 'm', 'm', 'z'], new arrow.Utf8())
    })
  );
  const dataset = session.importArrowDataset({id: 'observations', table: source});
  const reference = {datasetId: dataset.id, revision: dataset.revision};
  return {device, session, dataset, reference};
}

async function readUnsigned(vector: GPUVector): Promise<number[]> {
  const rows: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    rows.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return rows;
}
async function readFloat(vector: GPUVector): Promise<number[]> {
  const rows: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    rows.push(...new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return rows;
}

test('resident styles use global chunked quantiles, preserve nulls and cache unchanged domain work', async () => {
  const context = await fixture();
  const plan: KeplerResidentStylePlan = {
    id: 'fill',
    source: context.reference,
    column: 'value',
    kind: 'color',
    scale: 'quantile',
    palette: [RED, BLUE],
    domain: {kind: 'source'},
    outsideDomain: 'extrapolate'
  };
  const writes = context.dataset.value.columns['value'].data.map(chunk =>
    vi.spyOn(chunk.buffer, 'write')
  );
  const style = new KeplerResidentStyle({
    session: context.session,
    plan,
    maximumDomainCount: 8,
    maximumPaletteCount: 8,
    maximumLegendBytes: 128,
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  try {
    const submit = () => {
      const encoder = context.device.createCommandEncoder();
      style.encode(encoder);
      context.device.submit(encoder.finish());
    };
    submit();
    expect(style.colors!.data.map(chunk => chunk.length)).toEqual([4, 4]);
    expect(await readUnsigned(style.colors!)).toEqual([RED, RED, 0, RED, BLUE, BLUE, BLUE, BLUE]);
    const legend = await style.readLegend();
    expect(legend?.domain).toEqual([0, 100, 103]);
    expect(legend?.finiteCount).toBe(7);
    // NATIVEUI5: the GPU legend reports the one null row of eight.
    expect(legend?.missingCount).toBe(1);
    expect(style.readbackByteLength).toBe(16);
    const colors = style.colors;
    style.update({...plan, palette: [BLUE, RED]});
    submit();
    expect(style.colors).toBe(colors);
    expect(style.executionCounts).toEqual({domain: 1, style: 2});
    expect(await readUnsigned(style.colors!)).toEqual([BLUE, BLUE, 0, BLUE, RED, RED, RED, RED]);
    style.update({...plan, palette: [RED, GREEN, BLUE]});
    submit();
    expect((await style.readLegend())?.domain).toEqual([0, 2, 101, 103]);
    expect(style.executionCounts).toEqual({domain: 2, style: 3});
    submit();
    expect(style.executionCounts).toEqual({domain: 2, style: 3});
    for (const write of writes) expect(write).not.toHaveBeenCalled();
  } finally {
    for (const write of writes) write.mockRestore();
    await style.destroy(Promise.resolve());
    await context.session.destroy(Promise.resolve());
  }
});

test('selected style populations intersect null validity, retain quantile ties and never hide unselected output rows', async () => {
  const context = await fixture();
  const maskBuffers = [new Uint32Array([0, 0, 1, 0]), new Uint32Array([1, 0, 0, 0])].map(data =>
    context.device.createBuffer({data, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST})
  );
  const mask = new GPUVector({
    name: 'style-selection-mask',
    type: 'data',
    format: 'uint32',
    data: maskBuffers.map(
      buffer => new GPUData({buffer, format: 'uint32', length: 4, ownsBuffer: false})
    ),
    ownsData: true
  });
  const plan: KeplerResidentStylePlan = {
    id: 'selected-fill',
    source: context.reference,
    column: 'value',
    kind: 'color',
    scale: 'quantile',
    palette: [RED, BLUE],
    domain: {kind: 'selection', revision: 1},
    outsideDomain: 'extrapolate'
  };
  const style = new KeplerResidentStyle({
    session: context.session,
    plan,
    selectionMask: mask,
    maximumDomainCount: 4,
    maximumPaletteCount: 4,
    maximumLegendBytes: 64,
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  try {
    let encoder = context.device.createCommandEncoder();
    style.encode(encoder);
    context.device.submit(encoder.finish());
    expect((await style.readLegend())?.domain).toEqual([100, 100, 100]);
    expect((await style.readLegend())?.finiteCount).toBe(1);
    expect(await readUnsigned(style.colors!)).toEqual([RED, RED, 0, RED, BLUE, BLUE, BLUE, BLUE]);
    for (const buffer of maskBuffers) buffer.write(new Uint32Array(4));
    style.update({...plan, domain: {kind: 'selection', revision: 2}});
    encoder = context.device.createCommandEncoder();
    style.encode(encoder);
    context.device.submit(encoder.finish());
    const legend = await style.readLegend();
    expect(legend?.domain).toBeNull();
    expect(legend?.finiteCount).toBe(0);
    expect(legend?.selectionRevision).toBe(2);
    expect(await readUnsigned(style.colors!)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  } finally {
    await style.destroy(Promise.resolve());
    for (const buffer of maskBuffers) buffer.destroy();
    mask.destroy();
    await context.session.destroy(Promise.resolve());
  }
});

test('native categorical order maps independent color/scalar channels without reordering source chunks', async () => {
  const context = await fixture();
  expect(context.dataset.value.prepared.dictionaries['category'].values).toEqual(['z', 'a', 'm']);
  const common = {
    source: context.reference,
    column: 'category',
    domain: {kind: 'fixed' as const, values: [1, 2, 0]},
    legend: {labels: ['a', 'm', 'z'], codes: [1, 2, 0]}
  };
  const color = new KeplerResidentStyle({
    session: context.session,
    workingMemoryBudgetBytes: 1024 * 1024,
    plan: {
      ...common,
      id: 'fill',
      kind: 'color',
      scale: 'ordinal',
      palette: [BLUE, RED, GREEN],
      outsideDomain: 'reject'
    }
  });
  const size = new KeplerResidentStyle({
    session: context.session,
    workingMemoryBudgetBytes: 1024 * 1024,
    plan: {...common, id: 'radius', kind: 'scalar', scale: 'point', range: [1, 9]}
  });
  try {
    const encoder = context.device.createCommandEncoder();
    color.encode(encoder);
    size.encode(encoder);
    context.device.submit(encoder.finish());
    expect(await readUnsigned(color.colors!)).toEqual([
      BLUE,
      RED,
      0,
      BLUE,
      RED,
      GREEN,
      GREEN,
      BLUE
    ]);
    expect(await readFloat(size.values!)).toEqual([9, 1, 0, 9, 1, 5, 5, 9]);
    expect(await readUnsigned(size.validity!)).toEqual([1, 1, 0, 1, 1, 1, 1, 1]);
    expect(size.values!.data.map(chunk => chunk.length)).toEqual([4, 4]);
    expect((await color.readLegend())?.labels).toEqual(['a', 'm', 'z']);
    expect((await size.readLegend())?.domain).toEqual([1, 2, 0]);
  } finally {
    await Promise.all([color.destroy(Promise.resolve()), size.destroy(Promise.resolve())]);
    await context.session.destroy(Promise.resolve());
  }
});

test('numeric scalar styles extrapolate with source null fallback and replay into stable outputs', async () => {
  const context = await fixture();
  const plan: KeplerResidentStylePlan = {
    id: 'radius',
    source: context.reference,
    column: 'value',
    kind: 'scalar',
    scale: 'sqrt',
    range: [0, 10],
    domain: {kind: 'fixed', values: [0, 100]}
  };
  const style = new KeplerResidentStyle({
    session: context.session,
    plan,
    workingMemoryBudgetBytes: 1024 * 1024
  });
  try {
    let encoder = context.device.createCommandEncoder();
    style.encode(encoder);
    context.device.submit(encoder.finish());
    const values = await readFloat(style.values!);
    const expected = [0, 1, 0, Math.sqrt(2), 10, Math.sqrt(101), Math.sqrt(102), Math.sqrt(103)];
    for (let index = 0; index < values.length; index++)
      expect(values[index]).toBeCloseTo(expected[index], 5);
    expect(await readUnsigned(style.validity!)).toEqual([1, 1, 0, 1, 1, 1, 1, 1]);
    const output = style.values;
    style.update({
      ...plan,
      scale: 'linear',
      range: [2, 6],
      domain: {kind: 'fixed', values: [0, 2]}
    });
    encoder = context.device.createCommandEncoder();
    style.encode(encoder);
    context.device.submit(encoder.finish());
    expect(style.values).toBe(output);
    expect(await readFloat(style.values!)).toEqual([2, 4, 0, 6, 202, 204, 206, 208]);
  } finally {
    await style.destroy(Promise.resolve());
    await context.session.destroy(Promise.resolve());
  }
});

test('two style revisions encoded before one submission preserve ordered control snapshots', async () => {
  const context = await fixture();
  const plan: KeplerResidentStylePlan = {
    id: 'ordered-fill',
    source: context.reference,
    column: 'value',
    kind: 'color',
    scale: 'linear',
    palette: [RED, BLUE],
    domain: {kind: 'fixed', values: [0, 100]},
    outsideDomain: 'extrapolate'
  };
  const style = new KeplerResidentStyle({
    session: context.session,
    plan,
    maximumPaletteCount: 4,
    maximumDomainCount: 2,
    workingMemoryBudgetBytes: 1024 * 1024
  });
  const snapshots = [0, 1].map(() =>
    context.device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST | Buffer.COPY_SRC})
  );
  try {
    const encoder = context.device.createCommandEncoder();
    style.encode(encoder);
    const chunk = style.colors!.data[0];
    const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
    encoder.copyBufferToBuffer({
      sourceBuffer: buffer,
      sourceOffset: chunk.byteOffset,
      destinationBuffer: snapshots[0],
      size: 4
    });
    style.update({...plan, palette: [GREEN, BLUE]});
    style.encode(encoder);
    encoder.copyBufferToBuffer({
      sourceBuffer: buffer,
      sourceOffset: chunk.byteOffset,
      destinationBuffer: snapshots[1],
      size: 4
    });
    context.device.submit(encoder.finish());
    const bytes = await Promise.all(snapshots.map(snapshot => snapshot.readAsync(0, 4)));
    expect(bytes.map(value => new Uint32Array(value.buffer, value.byteOffset, 1)[0])).toEqual([
      RED,
      GREEN
    ]);
    expect(style.uploadedControlByteLength).toBe(style.controlSnapshotByteLength * 2);
  } finally {
    await style.destroy(Promise.resolve());
    for (const snapshot of snapshots) snapshot.destroy();
    await context.session.destroy(Promise.resolve());
  }
});

test('native inverted identity heights use global GPU domains and nullable selected populations', async () => {
  const context = await fixture();
  const maskBuffer = context.device.createBuffer({
    data: new Uint32Array([0, 1, 1, 1, 0, 0, 0, 0]),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const mask = new GPUVector({
    name: 'style-selection-mask',
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: [0, 16].map(
      byteOffset =>
        new GPUData({
          buffer: maskBuffer,
          format: 'uint32',
          length: 4,
          byteOffset,
          ownsBuffer: false
        })
    )
  });
  const plan: KeplerResidentStylePlan = {
    id: 'height',
    source: context.reference,
    column: 'value',
    kind: 'scalar',
    scale: 'identity',
    invert: true,
    range: [0, 1],
    nullValue: 0,
    domain: {kind: 'selection', revision: 1}
  };
  const style = new KeplerResidentStyle({
    session: context.session,
    plan,
    selectionMask: mask,
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const sourceReads = context.dataset.value.columns['value'].data.map(chunk =>
    vi.spyOn(chunk.buffer, 'readAsync')
  );
  try {
    const submit = () => {
      const encoder = context.device.createCommandEncoder();
      style.encode(encoder);
      context.device.submit(encoder.finish());
    };
    submit();
    expect((await style.readLegend())?.domain).toEqual([1, 2]);
    const outputs = await readFloat(style.values!);
    [3, 2, 0, 1, -97, -98, -99, -100].forEach((expected, index) =>
      expect(outputs[index]).toBeCloseTo(expected, 4)
    );
    const values = style.values;
    style.update({...plan, invert: false});
    submit();
    expect(await readFloat(style.values!)).toEqual([0, 1, 0, 2, 100, 101, 102, 103]);
    expect(style.values).toBe(values);
    expect(style.executionCounts.domain).toBe(1);
    for (const spy of sourceReads) expect(spy).not.toHaveBeenCalled();
  } finally {
    sourceReads.forEach(spy => spy.mockRestore());
    await style.destroy(Promise.resolve());
    mask.destroy();
    maskBuffer.destroy();
    await context.session.destroy(Promise.resolve());
  }
});
