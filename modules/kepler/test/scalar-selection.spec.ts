// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getInt64TimeWords} from '@luma-contributors/graph/gpu-dataframe';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {GPUKeplerScalarSelection} from '../src/selection/gpu-scalar-selection';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';

it('native filter IDs retain independent nullable masks on the same field and replay controls', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = makeVector(device, 'float32', [[5, 15, 25], [], [35, 45, Number.NaN]]);
  const validity = makeVector(device, 'uint32', [[1, 0, 1], [], [1, 1, 1]]);
  const categories = makeVector(device, 'uint32', [[0, 1, 2], [], [0, 2, 0xffffffff]]);
  const readSpies = [values, validity, categories].flatMap(vector =>
    vector.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'))
  );
  const selection = new GPUKeplerScalarSelection({
    device,
    columns: {values, categories},
    validity: {values: validity},
    datasetRevision: 9,
    filters: [
      {filterId: 'lower', kind: 'range', column: 'values'},
      {filterId: 'upper', kind: 'range', column: 'values', exclusiveMaximum: true},
      {filterId: 'classes', kind: 'category', column: 'categories', capacity: 64}
    ]
  });
  const masks = {...selection.predicateMasks};
  try {
    expect(selection.isCurrent(selection.acquireRevision())).toBe(false);
    submit(device, selection);
    for (const mask of Object.values(masks)) {
      expect(mask.data.map(chunk => chunk.length)).toEqual([3, 0, 3]);
      expect(await readMask(mask)).toEqual([1, 1, 1, 1, 1, 1]);
      for (const chunk of mask.data) expect(chunk.buffer.usage & Buffer.VERTEX).toBeTruthy();
    }
    const initialRevision = selection.acquireRevision();
    selection.setRange('lower', 20, 50);
    selection.setRange('upper', 0, 35);
    selection.setCategories('classes', [0, 0xffffffff]);
    expect(selection.isCurrent(initialRevision)).toBe(false);
    submit(device, selection);
    expect(await readMask(masks['lower'])).toEqual([0, 0, 1, 1, 1, 0]);
    expect(await readMask(masks['upper'])).toEqual([1, 0, 1, 0, 0, 0]);
    expect(await readMask(masks['classes'])).toEqual([1, 0, 0, 1, 0, 1]);
    const saved = JSON.parse(JSON.stringify(selection.getControls()));
    selection.setEmpty('lower');
    selection.setCategories('classes', []);
    selection.clear('upper');
    submit(device, selection);
    expect(await readMask(masks['lower'])).toEqual([0, 0, 0, 0, 0, 0]);
    expect(await readMask(masks['classes'])).toEqual([0, 0, 0, 0, 0, 0]);
    expect(await readMask(masks['upper'])).toEqual([1, 1, 1, 1, 1, 1]);
    selection.setControls(saved);
    submit(device, selection);
    expect(await readMask(masks['lower'])).toEqual([0, 0, 1, 1, 1, 0]);
    expect(await readMask(masks['upper'])).toEqual([1, 0, 1, 0, 0, 0]);
    expect(await readMask(masks['classes'])).toEqual([1, 0, 0, 1, 0, 1]);
    const beforeRejected = selection.getControls();
    expect(() =>
      selection.setControls([
        {...saved[0], minimum: 0, maximum: 100},
        {...saved[1], minimum: 10, maximum: 0},
        saved[2]
      ])
    ).toThrow('Reversed');
    expect(selection.getControls()).toEqual(beforeRejected);
    values.data[2].buffer.write(new Float32Array([10, 45, 30]));
    const previousLower = selection.getFilterRevision('lower');
    const previousUpper = selection.getFilterRevision('upper');
    selection.invalidate();
    expect(selection.getFilterRevision('lower')).toBe(previousLower + 1);
    expect(selection.getFilterRevision('upper')).toBe(previousUpper + 1);
    submit(device, selection);
    expect(await readMask(masks['lower'])).toEqual([0, 0, 1, 0, 1, 1]);
    expect(selection.isCurrent(selection.acquireRevision())).toBe(true);
    selection.discardEncoding();
    expect(selection.isCurrent(selection.acquireRevision())).toBe(false);
    for (const [filterId, mask] of Object.entries(masks))
      expect(selection.getMask(filterId)).toBe(mask);
    expect(selection.readbackByteLength).toBe(0);
    for (const spy of readSpies) expect(spy).not.toHaveBeenCalled();
  } finally {
    await closeSelection(device, selection);
    readSpies.forEach(spy => spy.mockRestore());
    for (const vector of [values, validity, categories]) {
      for (const chunk of vector.data) expect(chunk.buffer.destroyed).toBe(false);
      vector.destroy();
    }
  }
});

it('addFilter and removeFilter change one slot and match a full rebuild', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = makeVector(device, 'float32', [[5, 15, 25], [], [35, 45, Number.NaN]]);
  const validity = makeVector(device, 'uint32', [[1, 0, 1], [], [1, 1, 1]]);
  const categories = makeVector(device, 'uint32', [[0, 1, 2], [], [0, 2, 0xffffffff]]);
  const columns = {values, categories};
  const rangeFilter = {filterId: 'lower', kind: 'range', column: 'values'} as const;
  const classFilter = {
    filterId: 'classes',
    kind: 'category',
    column: 'categories',
    capacity: 8
  } as const;
  const incremental = new GPUKeplerScalarSelection({
    device,
    columns,
    validity: {values: validity},
    filters: [rangeFilter]
  });
  const rebuilt = new GPUKeplerScalarSelection({
    device,
    columns,
    validity: {values: validity},
    filters: [rangeFilter, classFilter]
  });
  try {
    const rangeMask = incremental.getMask('lower');
    const classMask = incremental.addFilter(classFilter, {columns, validity: {values: validity}});
    expect(incremental.getMask('lower')).toBe(rangeMask);
    expect(incremental.filterIds).toEqual(['lower', 'classes']);
    for (const selection of [incremental, rebuilt]) {
      selection.setRange('lower', 20, 50);
      selection.setCategories('classes', [0, 0xffffffff]);
      submit(device, selection);
    }
    expect(await readMask(classMask)).toEqual(await readMask(rebuilt.getMask('classes')));
    expect(await readMask(rangeMask)).toEqual(await readMask(rebuilt.getMask('lower')));
    expect(() => incremental.addFilter(classFilter, {columns})).toThrow('Duplicate');
    incremental.removeFilter('classes');
    expect(incremental.filterIds).toEqual(['lower']);
    expect(classMask.data[0].buffer.destroyed).toBe(true);
    expect(() => incremental.getMask('classes')).toThrow('Unknown');
    submit(device, incremental);
    expect(await readMask(rangeMask)).toEqual([0, 0, 1, 1, 1, 0]);
    expect(incremental.isCurrent(incremental.acquireRevision())).toBe(true);
  } finally {
    await closeSelection(device, incremental);
    await closeSelection(device, rebuilt);
    for (const vector of [values, validity, categories]) vector.destroy();
  }
});

it('exact time masks preserve null bypass, intervals, JSON replay and encoded window snapshots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const epoch = 9_007_199_254_740_993n;
  const timestamps = makeTimeVector(device, [[epoch, epoch + 1n], [], [epoch + 2n, -10n]]);
  const endTimestamps = makeTimeVector(device, [[epoch, epoch + 1n], [], [epoch + 3n, -5n]]);
  const validity = makeVector(device, 'uint32', [[1, 0], [], [1, 1]]);
  const values = makeVector(device, 'sint32', [[0, 1], [], [2, 3]]);
  const selection = new GPUKeplerScalarSelection({
    device,
    columns: {values},
    filters: [
      {filterId: 'number', kind: 'range', column: 'values'},
      {
        filterId: 'instant',
        kind: 'time',
        timestamps,
        validity,
        unit: 'microsecond',
        window: {start: epoch, end: epoch, unit: 'microsecond'}
      },
      {
        filterId: 'interval',
        kind: 'time',
        timestamps,
        endTimestamps,
        endValidity: validity,
        unit: 'microsecond',
        window: {start: epoch, end: epoch, unit: 'microsecond'}
      }
    ]
  });
  const snapshots: Buffer[] = [];
  try {
    submit(device, selection);
    expect(await readMask(selection.getMask('instant'))).toEqual([1, 1, 1, 1]);
    selection.setTimeWindow('instant', {start: epoch + 1n, end: epoch + 2n, unit: 'microsecond'});
    selection.setTimeWindow('interval', {start: epoch + 3n, end: epoch + 3n, unit: 'microsecond'});
    selection.setRange('number', 0, 1);
    const saved = JSON.parse(JSON.stringify(selection.getControls()));
    const encoder = device.createCommandEncoder();
    selection.encode(encoder);
    snapshots.push(copyMask(device, encoder, selection.getMask('instant')));
    snapshots.push(copyMask(device, encoder, selection.getMask('number')));
    selection.setTimeWindow('instant', {start: -10n, end: -5n, unit: 'microsecond'});
    selection.setRange('number', 2, 3);
    selection.encode(encoder);
    snapshots.push(copyMask(device, encoder, selection.getMask('instant')));
    snapshots.push(copyMask(device, encoder, selection.getMask('number')));
    device.submit(encoder.finish());
    expect(await Promise.all(snapshots.map(readWords))).toEqual([
      [0, 0, 1, 0],
      [1, 1, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 1, 1]
    ]);
    expect(await readMask(selection.getMask('interval'))).toEqual([0, 0, 1, 0]);
    selection.setControls(saved);
    submit(device, selection);
    expect(await readMask(selection.getMask('instant'))).toEqual([0, 0, 1, 0]);
    selection.setEmpty('instant');
    submit(device, selection);
    expect(await readMask(selection.getMask('instant'))).toEqual([0, 0, 0, 0]);
    selection.clear('instant');
    submit(device, selection);
    expect(await readMask(selection.getMask('instant'))).toEqual([1, 1, 1, 1]);
    expect(() =>
      selection.setTimeWindow('instant', {start: 0.5, end: 1, unit: 'microsecond'})
    ).toThrow('safe integer');
    expect(selection.readbackByteLength).toBe(0);
  } finally {
    await closeSelection(device, selection);
    snapshots.forEach(buffer => buffer.destroy());
    for (const vector of [timestamps, endTimestamps, validity, values]) vector.destroy();
  }
});

it('chart-origin time predicates map only actual nulls to epoch while ordinary sliders reject them', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const timestamps = makeTimeVector(device, [[-1n, 0n, 55n], [], [500n, 8_640_000_000_000_001n]]);
  const validity = makeVector(device, 'uint32', [[1, 1, 0], [], [1, 1]]);
  const spies = [...timestamps.data, ...validity.data].map(chunk =>
    vi.spyOn(chunk.buffer, 'readAsync')
  );
  const selection = new GPUKeplerScalarSelection({
    device,
    filters: [
      {
        filterId: 'persisted-chart-id',
        kind: 'time',
        timestamps,
        validity,
        unit: 'millisecond',
        chartNullAsEpoch: true,
        window: {start: -1, end: 0, unit: 'millisecond'}
      },
      {
        filterId: 'ordinary-slider',
        kind: 'time',
        timestamps,
        validity,
        unit: 'millisecond',
        window: {start: -1, end: 0, unit: 'millisecond'}
      }
    ]
  });
  const snapshots: Buffer[] = [];
  try {
    selection.setTimeWindow('persisted-chart-id', {start: -1, end: 0, unit: 'millisecond'});
    selection.setTimeWindow('ordinary-slider', {start: -1, end: 0, unit: 'millisecond'});
    submit(device, selection);
    expect(await readMask(selection.getMask('persisted-chart-id'))).toEqual([1, 1, 1, 0, 0]);
    expect(await readMask(selection.getMask('ordinary-slider'))).toEqual([1, 1, 0, 0, 0]);
    expect(selection.getMask('persisted-chart-id').data.map(chunk => chunk.length)).toEqual([
      3, 0, 2
    ]);
    const saved = JSON.parse(JSON.stringify(selection.getControls()));
    const revision = selection.acquireRevision();
    const encoder = device.createCommandEncoder();
    selection.setTimeWindow('persisted-chart-id', {start: 0, end: 0, unit: 'millisecond'});
    selection.encode(encoder);
    snapshots.push(copyMask(device, encoder, selection.getMask('persisted-chart-id')));
    selection.setTimeWindow('persisted-chart-id', {start: 1, end: 500, unit: 'millisecond'});
    expect(selection.isCurrent(revision)).toBe(false);
    selection.encode(encoder);
    snapshots.push(copyMask(device, encoder, selection.getMask('persisted-chart-id')));
    device.submit(encoder.finish());
    expect(await Promise.all(snapshots.map(readWords))).toEqual([
      [0, 1, 1, 0, 0],
      [0, 0, 0, 1, 0]
    ]);
    selection.setControls(saved);
    submit(device, selection);
    expect(await readMask(selection.getMask('persisted-chart-id'))).toEqual([1, 1, 1, 0, 0]);
    selection.setEmpty('persisted-chart-id');
    submit(device, selection);
    expect(await readMask(selection.getMask('persisted-chart-id'))).toEqual([0, 0, 0, 0, 0]);
    expect(spies.every(spy => spy.mock.calls.length === 0)).toBe(true);
  } finally {
    await closeSelection(device, selection);
    snapshots.forEach(snapshot => snapshot.destroy());
    spies.forEach(spy => spy.mockRestore());
    timestamps.destroy();
    validity.destroy();
  }
});

it('scalar reservations roll back on failure and survive until the completion fence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = makeVector(device, 'uint32', [[0, 1, 2]]);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024 * 1024});
  registry.registerDataset({
    id: 'source',
    value: values,
    batches: [{id: 'batch', rowCount: 3}],
    resources: []
  });
  const props = {
    device,
    columns: {values},
    filters: [{filterId: 'range', kind: 'range' as const, column: 'values'}],
    registry,
    datasetId: 'source'
  };
  try {
    expect(() => new GPUKeplerScalarSelection({...props, workingMemoryBudgetBytes: 1})).toThrow(
      'reservation'
    );
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(0);
    const selection = new GPUKeplerScalarSelection({
      ...props,
      workingMemoryBudgetBytes: 256 * 1024
    });
    expect(selection.accountedResidentByteLength).toBeGreaterThan(36);
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(256 * 1024);
    submit(device, selection);
    expect(await readMask(selection.getMask('range'))).toEqual([1, 1, 1]);
    const maskBuffer = selection.getMask('range').data[0].buffer;
    let release!: () => void;
    const completion = new Promise<void>(resolve => {
      release = resolve;
    });
    const closed = selection.close(completion);
    expect(maskBuffer.destroyed).toBe(false);
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(256 * 1024);
    expect(() => selection.clear('range')).toThrow('closed');
    release();
    await closed;
    expect(maskBuffer.destroyed).toBe(true);
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(0);
    expect(selection.accountedResidentByteLength).toBe(0);
    const stale = new GPUKeplerScalarSelection({...props, workingMemoryBudgetBytes: 256 * 1024});
    const snapshot = stale.acquireRevision();
    registry.registerDataset({
      id: 'source',
      value: values,
      batches: [{id: 'batch', rowCount: 3}],
      resources: []
    });
    expect(stale.isCurrent(snapshot)).toBe(false);
    expect(() => stale.setRange('range', 0, 1)).toThrow('stale');
    await stale.close(Promise.resolve());
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(0);
    const failedFence = new GPUKeplerScalarSelection({
      ...props,
      workingMemoryBudgetBytes: 256 * 1024
    });
    await expect(failedFence.close(Promise.reject(new Error('submission failed')))).rejects.toThrow(
      'submission failed'
    );
    expect(registry.memoryUsage.reservedBytes.gpu).toBe(0);
  } finally {
    values.destroy();
    await registry.destroy();
  }
});

it('scalar filters preserve empty physical chunks and reject topology replacement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunks of [[], [[]]] as number[][][]) {
    const values = makeVector(device, 'uint32', chunks);
    const selection = new GPUKeplerScalarSelection({
      device,
      columns: {values},
      filters: [{filterId: 'empty', kind: 'category', column: 'values', capacity: 1}]
    });
    try {
      selection.setCategories('empty', [1]);
      submit(device, selection);
      expect(selection.getMask('empty').data.length).toBe(chunks.length);
      expect(await readMask(selection.getMask('empty'))).toEqual([]);
      const replacement = makeVector(device, 'uint32', [[]]);
      const originalChunks = [...values.data];
      const encoder = device.createCommandEncoder();
      try {
        values.data.splice(0, values.data.length, ...replacement.data);
        expect(() => selection.encode(encoder)).toThrow('topology');
      } finally {
        values.data.splice(0, values.data.length, ...originalChunks);
        encoder.destroy();
        replacement.destroy();
      }
    } finally {
      await closeSelection(device, selection);
      values.destroy();
    }
  }
});

it('native fractional range controls match mathematical membership over integer and Float32 columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const integers = makeVector(device, 'sint32', [[-1, 0, 1, 2]]);
  const floats = makeVector(device, 'float32', [[1 - 2 ** -24, 1, 1 + 2 ** -23, 2]]);
  const selection = new GPUKeplerScalarSelection({
    device,
    columns: {integers, floats},
    filters: [
      {filterId: 'integer', kind: 'range', column: 'integers'},
      {filterId: 'float', kind: 'range', column: 'floats'},
      {filterId: 'exclusive', kind: 'range', column: 'floats', exclusiveMaximum: true}
    ]
  });
  try {
    selection.setRange('integer', -0.8, 1.8);
    selection.setRange('float', 1 + Number.EPSILON, 2);
    selection.setRange('exclusive', 0, 1);
    submit(device, selection);
    expect(await readMask(selection.getMask('integer'))).toEqual([0, 1, 1, 0]);
    expect(await readMask(selection.getMask('float'))).toEqual([0, 0, 1, 1]);
    expect(await readMask(selection.getMask('exclusive'))).toEqual([1, 0, 0, 0]);
    selection.setRange('integer', 0.2, 0.8);
    selection.setRange('float', 0, 1 - Number.EPSILON);
    submit(device, selection);
    expect(await readMask(selection.getMask('integer'))).toEqual([0, 0, 0, 0]);
    expect(await readMask(selection.getMask('float'))).toEqual([1, 0, 0, 0]);
  } finally {
    await closeSelection(device, selection);
    integers.destroy();
    floats.destroy();
  }
});

function makeVector<Format extends 'float32' | 'uint32' | 'sint32'>(
  device: Device,
  format: Format,
  chunks: number[][]
): GPUVector<Format> {
  const ArrayType =
    format === 'float32' ? Float32Array : format === 'sint32' ? Int32Array : Uint32Array;
  return new GPUVector({
    type: 'data',
    name: 'source',
    format,
    ownsData: true,
    data: chunks.map(
      values =>
        new GPUData({
          buffer: device.createBuffer({
            ...(values.length ? {data: ArrayType.from(values)} : {byteLength: 4}),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format,
          length: values.length,
          ownsBuffer: true
        })
    )
  });
}

function makeTimeVector(device: Device, chunks: bigint[][]): GPUVector<'uint32x2'> {
  return new GPUVector({
    type: 'data',
    name: 'timestamps',
    format: 'uint32x2',
    ownsData: true,
    data: chunks.map(
      values =>
        new GPUData({
          buffer: device.createBuffer({
            ...(values.length
              ? {data: getInt64TimeWords(BigInt64Array.from(values))}
              : {byteLength: 8}),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
          }),
          format: 'uint32x2',
          length: values.length,
          ownsBuffer: true
        })
    )
  });
}

function submit(device: Device, selection: GPUKeplerScalarSelection): void {
  const encoder = device.createCommandEncoder();
  selection.encode(encoder);
  device.submit(encoder.finish());
}

async function closeSelection(device: Device, selection: GPUKeplerScalarSelection): Promise<void> {
  const fence = device.createFence();
  try {
    await selection.close(fence.signaled);
  } finally {
    fence.destroy();
  }
}

async function readMask(mask: GPUVector<'uint32'>): Promise<number[]> {
  return (
    await Promise.all(
      mask.data.map(async chunk => {
        if (!chunk.length) return [];
        const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
        return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
      })
    )
  ).flat();
}

async function readWords(buffer: Buffer): Promise<number[]> {
  const bytes = await buffer.readAsync();
  return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
}

function copyMask(device: Device, encoder: CommandEncoder, mask: GPUVector<'uint32'>): Buffer {
  const destination = device.createBuffer({
    byteLength: mask.length * 4,
    usage: Buffer.COPY_DST | Buffer.COPY_SRC
  });
  let offset = 0;
  for (const chunk of mask.data) {
    if (chunk.length)
      encoder.copyBufferToBuffer({
        sourceBuffer: chunk.buffer as Buffer,
        sourceOffset: chunk.byteOffset,
        destinationBuffer: destination,
        destinationOffset: offset,
        size: chunk.length * 4
      });
    offset += chunk.length * 4;
  }
  return destination;
}
