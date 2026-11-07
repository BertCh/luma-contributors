// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {KeplerResidentStyle, type KeplerResidentStylePlan} from '../../src/styles';

const RED = 0xff0000ff;
const BLUE = 0xffff0000;

async function readWords(vector: GPUVector, floating = false): Promise<number[]> {
  const result: number[] = [];
  for (const chunk of vector.data) {
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    result.push(
      ...(floating
        ? new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length)
        : new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length))
    );
  }
  return result;
}

// Independent finite native d3 log/normalization formula, including its constant-domain exception.
function logPosition(value: number | null, domain: readonly number[] | null): number | null {
  if (
    value === null ||
    !domain ||
    !((domain[0] > 0 && domain[1] > 0) || (domain[0] < 0 && domain[1] < 0))
  )
    return null;
  if (domain[0] === domain[1]) return 0.5;
  if (value === 0 || value < 0 !== domain[0] < 0) return null;
  return (
    (Math.log(Math.abs(value)) - Math.log(Math.abs(domain[0]))) /
    (Math.log(Math.abs(domain[1])) - Math.log(Math.abs(domain[0])))
  );
}

test('native source log domain uses all finite valid chunks and keeps zero inputs unmapped', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for native log verification');
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 32 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'source-log',
    table: new arrow.Table({value: arrow.vectorFromArray([0, null], new arrow.Float32())}).concat(
      new arrow.Table({value: arrow.vectorFromArray([1, 100], new arrow.Float32())})
    )
  });
  const style = new KeplerResidentStyle({
    session,
    plan: {
      id: 'source-log',
      source: {datasetId: dataset.id, revision: dataset.revision},
      column: 'value',
      kind: 'scalar',
      scale: 'log',
      domain: {kind: 'source'},
      range: [0, 70]
    },
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  try {
    const encoder = device.createCommandEncoder();
    style.encode(encoder);
    device.submit(encoder.finish());
    const legend = await style.readLegend();
    expect(legend?.domain).toEqual([Math.fround(1e-5), 100]);
    expect(legend?.domainStatus).toBe('valid');
    expect(legend?.finiteCount).toBe(3);
    expect(style.values!.data.map(chunk => chunk.length)).toEqual([2, 2]);
    expect(await readWords(style.validity!)).toEqual([0, 0, 1, 1]);
    const values = await readWords(style.values!, true);
    expect(values[0]).toBe(0);
    expect(values[1]).toBe(0);
    expect(values[2]).toBeCloseTo(50, 3);
    expect(values[3]).toBeCloseTo(70, 3);
    expect(style.readbackByteLength).toBe(12);
  } finally {
    await style.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
  }
});

test('native selected logarithmic domains retain global zero/negative populations and diagnose unsupported domains', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for native log verification');
  const rows = [-100, -10, null, 0, 1, 10, 100, Math.fround(1e-6)];
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'log-source',
    table: new arrow.Table({
      value: arrow.vectorFromArray(rows.slice(0, 4), new arrow.Float32())
    }).concat(new arrow.Table({value: arrow.vectorFromArray(rows.slice(4), new arrow.Float32())}))
  });
  const maskBuffers = [0, 1].map(() =>
    device.createBuffer({data: new Uint32Array(4), usage: Buffer.STORAGE | Buffer.COPY_DST})
  );
  const mask = new GPUVector({
    type: 'data',
    format: 'uint32',
    ownsData: true,
    data: maskBuffers.map(
      buffer => new GPUData({buffer, format: 'uint32', length: 4, ownsBuffer: false})
    )
  });
  const common = {
    source: {datasetId: dataset.id, revision: dataset.revision},
    column: 'value',
    scale: 'log' as const,
    domain: {kind: 'selection' as const, revision: 1}
  };
  const colorPlan: KeplerResidentStylePlan = {
    ...common,
    id: 'log-color',
    kind: 'color',
    palette: [RED, BLUE],
    outsideDomain: 'extrapolate'
  };
  const scalarPlan: KeplerResidentStylePlan = {
    ...common,
    id: 'log-radius',
    kind: 'scalar',
    range: [10, 20]
  };
  const color = new KeplerResidentStyle({
    session,
    plan: colorPlan,
    selectionMask: mask,
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const scalar = new KeplerResidentStyle({
    session,
    plan: scalarPlan,
    selectionMask: mask,
    workingMemoryBudgetBytes: 8 * 1024 * 1024
  });
  const cases = [
    {indices: [4, 6], domain: [1, 100]},
    {indices: [0, 1], domain: [-100, -10]},
    {indices: [3, 6], domain: [Math.fround(1e-5), 100]},
    {indices: [0, 3], domain: [-100, 0]},
    {indices: [0, 6], domain: [-100, 100]},
    {indices: [3], domain: [Math.fround(1e-5), 0]},
    {indices: [3, 7], domain: [Math.fround(1e-5), Math.fround(1e-6)]},
    {indices: [4], domain: [1, 1]},
    {indices: [0], domain: [-100, -100]},
    {indices: [2], domain: null}
  ];
  try {
    expect(color.colors!.data.map(chunk => chunk.length)).toEqual([4, 4]);
    expect(scalar.values!.data.map(chunk => chunk.length)).toEqual([4, 4]);
    for (const [index, entry] of cases.entries()) {
      maskBuffers.forEach((buffer, chunk) =>
        buffer.write(
          new Uint32Array(4).map((_, local) => Number(entry.indices.includes(chunk * 4 + local)))
        )
      );
      const domain = {kind: 'selection' as const, revision: index + 1};
      color.update({...colorPlan, domain});
      scalar.update({...scalarPlan, domain});
      const encoder = device.createCommandEncoder();
      color.encode(encoder);
      scalar.encode(encoder);
      device.submit(encoder.finish());
      const [colorLegend, scalarLegend, colors, values, validity] = await Promise.all([
        color.readLegend(),
        scalar.readLegend(),
        readWords(color.colors!),
        readWords(scalar.values!, true),
        readWords(scalar.validity!)
      ]);
      const status = !entry.domain
        ? 'empty'
        : logPosition(1, entry.domain) === null && logPosition(-1, entry.domain) === null
          ? 'invalid-log-domain'
          : 'valid';
      for (const legend of [colorLegend, scalarLegend]) {
        expect(legend?.domain).toEqual(entry.domain);
        expect(legend?.domainStatus).toBe(status);
        expect(legend?.selectionRevision).toBe(index + 1);
        expect(legend?.finiteCount).toBe(entry.indices.filter(row => rows[row] !== null).length);
      }
      rows.forEach((value, row) => {
        const position = logPosition(value, entry.domain);
        expect(validity[row]).toBe(position === null ? 0 : 1);
        if (position === null) {
          expect(values[row]).toBe(0);
          expect(colors[row]).toBe(0);
        } else {
          expect(values[row]).toBeCloseTo(10 + position * 10, 3);
          const clampByte = (channel: number) => Math.max(0, Math.min(255, Math.round(channel)));
          expect(
            Math.abs((colors[row] & 255) - clampByte(255 * (1 - position)))
          ).toBeLessThanOrEqual(1);
          expect(
            Math.abs(((colors[row] >>> 16) & 255) - clampByte(255 * position))
          ).toBeLessThanOrEqual(1);
          expect(colors[row] >>> 24).toBe(255);
        }
      });
    }
    // Updating the scale must remove a previously applied logarithmic zero floor, while keeping
    // the same source/output chunks and compiled mapping topology.
    maskBuffers[0].write(new Uint32Array([0, 0, 0, 1]));
    maskBuffers[1].write(new Uint32Array([0, 0, 1, 0]));
    color.update({...colorPlan, domain: {kind: 'selection', revision: 20}});
    let encoder = device.createCommandEncoder();
    color.encode(encoder);
    device.submit(encoder.finish());
    expect((await color.readLegend())?.domain).toEqual([Math.fround(1e-5), 100]);
    const pendingLegend = color.readLegend();
    color.update({...colorPlan, scale: 'linear', domain: {kind: 'selection', revision: 20}});
    expect(await pendingLegend).toBeNull();
    encoder = device.createCommandEncoder();
    color.encode(encoder);
    device.submit(encoder.finish());
    expect((await color.readLegend())?.domain).toEqual([0, 100]);
    expect((await color.readLegend())?.domainStatus).toBe('valid');
  } finally {
    await Promise.all([color.destroy(Promise.resolve()), scalar.destroy(Promise.resolve())]);
    mask.destroy();
    for (const buffer of maskBuffers) buffer.destroy();
    await session.destroy(Promise.resolve());
  }
});
