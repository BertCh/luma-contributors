// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUContinuousSpatialSample} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '@luma-contributors/graph/testing';
import {requireTestDevice} from '@luma-contributors/graph/testing';

function makeHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const view = <Format extends 'float32' | 'float32x2' | 'uint32' | 'uint32x4'>(
    format: Format,
    values: number[] | Float32Array | Uint32Array
  ) => {
    const data = format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values);
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const components = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `continuous-${buffers.length}`,
        buffer,
        format,
        data.length / components
      ),
      read: async () => {
        const bytes = await buffer.readAsync();
        return Array.from(
          format.startsWith('uint')
            ? new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
            : new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
        );
      }
    };
  };
  const output = (waves: number, capacity: number) => ({
    positions: view('float32x2', new Float32Array(waves * capacity * 2)),
    designWeights: view('float32', new Float32Array(waves * capacity)),
    provenance: view('uint32x4', new Uint32Array(waves * capacity * 4)),
    status: view('uint32x4', new Uint32Array(waves * 4))
  });
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    view,
    output,
    run: async () => {
      compiled ??= graph.compile();
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
    },
    destroy: () => {
      compiled?.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}
function outputViews(output: ReturnType<ReturnType<typeof makeHarness>['output']>) {
  return {
    positions: output.positions.view,
    designWeights: output.designWeights.view,
    provenance: output.provenance.view,
    status: output.status.view
  };
}

it('continuous triangle sampling has area-weighted mass and first moments over 4096 independent phases', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = makeHarness(device);
  try {
    const frame = harness.view('float32x2', [0, 0, 1, 0, 0, 1]);
    const waves = 4096;
    const capacity = 16;
    const output = harness.output(waves, capacity);
    const sample = new GPUContinuousSpatialSample({
      frame: {type: 'area', positions: frame.view, bounds: [0, 0, 1, 1]},
      spacing: 0.25,
      seed: 431,
      waves,
      capacity,
      output: outputViews(output)
    });
    harness.graph.add(sample);
    await harness.run();
    const [positions, weights, provenance, status] = await Promise.all([
      output.positions.read(),
      output.designWeights.read(),
      output.provenance.read(),
      output.status.read()
    ]);
    let mass = 0,
      firstMoment = 0,
      secondMoment = 0;
    const quarterCounts = [0, 0, 0, 0];
    for (let wave = 0; wave < waves; wave++) {
      expect(status[wave * 4 + 3]).toBe(0);
      const count = status[wave * 4];
      for (let slot = 0; slot < count; slot++) {
        const row = wave * capacity + slot;
        const x = positions[row * 2];
        const y = positions[row * 2 + 1];
        expect(x >= 0 && y >= 0 && x + y <= 1 + 1e-6).toBe(true);
        expect(weights[row]).toBe(0.0625);
        expect(provenance[row * 4 + 1]).toBe(wave);
        expect(provenance[row * 4 + 3]).toBe(wave === 0 ? 0 : 1);
        mass += weights[row];
        firstMoment += weights[row] * x;
        secondMoment += weights[row] * y;
        quarterCounts[Math.min(3, Math.floor(x * 4))]++;
        for (let previous = 0; previous < slot; previous++) {
          const other = (wave * capacity + previous) * 2;
          expect(Math.hypot(x - positions[other], y - positions[other + 1])).toBeGreaterThanOrEqual(
            0.25 - 1e-6
          );
        }
      }
    }
    expect(mass / waves).toBeCloseTo(0.5, 2);
    expect(firstMoment / waves).toBeCloseTo(1 / 6, 2);
    expect(secondMoment / waves).toBeCloseTo(1 / 6, 2);
    // Analytic triangle-strip masses integrate 1-x; this catches equal triangle/cell candidate weighting.
    const total = quarterCounts.reduce((sum, value) => sum + value, 0);
    [7 / 16, 5 / 16, 3 / 16, 1 / 16].forEach((probability, index) =>
      expect(Math.abs(quarterCounts[index] / total - probability)).toBeLessThan(0.015)
    );
    expect(sample.samplingIntensity).toBe(16);
  } finally {
    harness.destroy();
  }
});

it('legacy area sites exclude disks without changing eligible-area weights; overlapping frame triangles do not double count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = makeHarness(device);
  try {
    const frame = harness.view('float32x2', [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1]);
    const legacy = harness.view('float32x2', [0.5, 0.5]);
    const waves = 2048;
    const capacity = 26;
    const output = harness.output(waves, capacity);
    harness.graph.add(
      new GPUContinuousSpatialSample({
        frame: {
          type: 'area',
          positions: frame.view,
          bounds: [0, 0, 1, 1],
          legacySites: legacy.view
        },
        spacing: 0.2,
        seed: 893,
        waves,
        capacity,
        output: outputViews(output)
      })
    );
    await harness.run();
    const [positions, weights, status, provenance] = await Promise.all([
      output.positions.read(),
      output.designWeights.read(),
      output.status.read(),
      output.provenance.read()
    ]);
    let total = 0;
    for (let wave = 0; wave < waves; wave++) {
      expect(status[wave * 4 + 3]).toBe(0);
      const base = wave * capacity;
      expect(provenance[base * 4 + 3]).toBe(2);
      expect(weights[base]).toBe(0);
      for (let slot = 1; slot < status[wave * 4]; slot++) {
        const row = base + slot;
        expect(
          Math.hypot(positions[row * 2] - 0.5, positions[row * 2 + 1] - 0.5)
        ).toBeGreaterThanOrEqual(0.2 - 1e-6);
        total += weights[row];
      }
    }
    expect(Math.abs(total / waves - (1 - Math.PI * 0.2 ** 2))).toBeLessThan(0.006);
  } finally {
    harness.destroy();
  }
});

it('polyline sampling uses continuous arc length, legacy intervals and replayable independent replacement waves', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = makeHarness(device);
  try {
    const frame = harness.view('float32x2', [0, 0, 1, 0, 1, 2]);
    const legacy = harness.view('float32', [1.5]);
    const parameters = harness.view('uint32', [9, 10]);
    const waves = 4;
    const capacity = 8;
    const output = harness.output(waves, capacity);
    harness.graph.add(
      new GPUContinuousSpatialSample({
        frame: {type: 'line', positions: frame.view, legacySites: legacy.view},
        spacing: 0.5,
        seed: 0,
        parameters: parameters.view,
        waves,
        capacity,
        output: outputViews(output)
      })
    );
    await harness.run();
    const [positions, weights, status, provenance] = await Promise.all([
      output.positions.read(),
      output.designWeights.read(),
      output.status.read(),
      output.provenance.read()
    ]);
    for (let wave = 0; wave < waves; wave++) {
      expect(status.slice(wave * 4, wave * 4 + 4)).toEqual([5, 5, 4, 0]);
      const base = wave * capacity;
      expect(positions.slice(base * 2, base * 2 + 2)).toEqual([1, 0.5]);
      expect(weights[base]).toBe(0);
      let previous = -Infinity;
      for (let slot = 1; slot < 5; slot++) {
        const row = base + slot;
        const measure = positions[row * 2] + positions[row * 2 + 1];
        expect(Math.abs(measure - 1.5)).toBeGreaterThanOrEqual(0.5 - 1e-6);
        expect(measure - previous).toBeGreaterThanOrEqual(0.5 - 1e-6);
        previous = measure;
        expect(provenance[row * 4 + 1]).toBe(10 + wave);
        expect(weights[row]).toBe(0.5);
      }
    }
    await harness.run();
    expect(await output.positions.read()).toEqual(positions);
    parameters.buffer.write(new Uint32Array([10, 20]));
    await harness.run();
    expect(await output.positions.read()).not.toEqual(positions);
    expect((await output.provenance.read())[1]).toBe(20);
    legacy.buffer.write(new Float32Array([4]));
    await harness.run();
    expect((await output.status.read()).filter((_, index) => index % 4 === 3)).toEqual([
      1, 1, 1, 1
    ]);
    legacy.buffer.write(new Float32Array([1.5]));
    parameters.buffer.write(new Uint32Array([10, 0xffffffff]));
    await harness.run();
    const wrappedStatus = await output.status.read();
    expect(wrappedStatus.filter((_, index) => index % 4 === 3)).toEqual([0, 1, 1, 1]);
    expect(wrappedStatus.filter((_, index) => index % 4 === 0)).toEqual([5, 0, 0, 0]);
    expect((await output.designWeights.read()).slice(capacity)).toEqual(
      new Array(3 * capacity).fill(0)
    );
    parameters.buffer.write(new Uint32Array([10, 30]));
    await harness.run();
    expect((await output.status.read()).filter((_, index) => index % 4 === 3)).toEqual([
      0, 0, 0, 0
    ]);
  } finally {
    harness.destroy();
  }
});

it('capacity and candidate budget are explicit incomplete results and invalid geometry never produces weighted successes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = makeHarness(device);
  try {
    const frame = harness.view('float32x2', [0, 0, 1, 0]);
    const truncated = harness.output(1, 1);
    const bounded = harness.output(1, 16);
    harness.graph.add(
      new GPUContinuousSpatialSample({
        id: 'truncated',
        frame: {type: 'line', positions: frame.view},
        spacing: 0.1,
        seed: 7,
        capacity: 1,
        output: outputViews(truncated)
      })
    );
    harness.graph.add(
      new GPUContinuousSpatialSample({
        id: 'bounded',
        frame: {type: 'line', positions: frame.view},
        spacing: 0.1,
        seed: 7,
        capacity: 16,
        maximumCandidates: 2,
        output: outputViews(bounded)
      })
    );
    await harness.run();
    expect(await truncated.status.read()).toEqual([1, 10, 10, 2]);
    expect(await bounded.status.read()).toEqual([2, 2, 2, 4]);
    frame.buffer.write(new Float32Array([0, 0, NaN, 0]));
    await harness.run();
    expect(await truncated.status.read()).toEqual([0, 0, 0, 1]);
    expect(await truncated.designWeights.read()).toEqual([0]);
  } finally {
    harness.destroy();
  }
});

it('continuous area frames preserve holes and disconnected components with area-proportional expected counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = makeHarness(device);
  try {
    const rectangle = (minimumX: number, minimumY: number, maximumX: number, maximumY: number) => [
      minimumX,
      minimumY,
      maximumX,
      minimumY,
      maximumX,
      maximumY,
      minimumX,
      minimumY,
      maximumX,
      maximumY,
      minimumX,
      maximumY
    ];
    const triangles = [
      ...rectangle(0, 0, 1, 0.25),
      ...rectangle(0, 0.75, 1, 1),
      ...rectangle(0, 0.25, 0.25, 0.75),
      ...rectangle(0.75, 0.25, 1, 0.75),
      ...rectangle(2, 0, 3, 1)
    ];
    const frame = harness.view('float32x2', triangles);
    const waves = 512;
    const capacity = 100;
    const output = harness.output(waves, capacity);
    harness.graph.add(
      new GPUContinuousSpatialSample({
        frame: {type: 'area', positions: frame.view, bounds: [0, 0, 3, 1]},
        spacing: 0.15,
        seed: 7834,
        waves,
        capacity,
        output: outputViews(output)
      })
    );
    await harness.run();
    const [positions, weights, status] = await Promise.all([
      output.positions.read(),
      output.designWeights.read(),
      output.status.read()
    ]);
    const masses = [0, 0];
    for (let wave = 0; wave < waves; wave++) {
      expect(status[wave * 4 + 3]).toBe(0);
      for (let slot = 0; slot < status[wave * 4]; slot++) {
        const row = wave * capacity + slot;
        const x = positions[row * 2];
        const y = positions[row * 2 + 1];
        expect(x < 1 || x >= 2).toBe(true);
        expect(x <= 0.25 || x >= 0.75 || y <= 0.25 || y >= 0.75).toBe(true);
        masses[x < 1 ? 0 : 1] += weights[row];
      }
    }
    expect(Math.abs(masses[0] / waves - 0.75)).toBeLessThan(0.015);
    expect(Math.abs(masses[1] / waves - 1)).toBeLessThan(0.015);
  } finally {
    harness.destroy();
  }
});

it('executes the registered continuous sampling lens and captures complete weighted waves', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await import(
    '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/continuous-spatial-sample.lens'
  );
  const {getLensScenario} = await import('../../../src/computation-lenses/registry');
  const descriptor = getLensScenario('gpu-spatial-analysis.continuous-spatial-sample')!;
  const loaded = await descriptor.load();
  const result = await loaded.run(device);
  expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
  expect(
    Array.from(result.captures!.snapshots.find(capture => capture.name === 'designWeights')!.values)
  ).toEqual(new Array(8).fill(0.25));
  expect(
    Array.from(result.captures!.snapshots.find(capture => capture.name === 'status')!.values)
  ).toEqual([4, 4, 4, 0, 4, 4, 4, 0]);
});

it('noninteger line length has the expected random count and ignores zero-length vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = makeHarness(device);
  try {
    const frame = harness.view('float32x2', [0, 0, 0.5, 0, 0.5, 0, 1.1, 0]);
    const waves = 2048;
    const capacity = 4;
    const output = harness.output(waves, capacity);
    harness.graph.add(
      new GPUContinuousSpatialSample({
        frame: {type: 'line', positions: frame.view},
        spacing: 0.3,
        seed: 987,
        waves,
        capacity,
        output: outputViews(output)
      })
    );
    await harness.run();
    const [positions, weights, status, provenance] = await Promise.all([
      output.positions.read(),
      output.designWeights.read(),
      output.status.read(),
      output.provenance.read()
    ]);
    let mass = 0;
    let moment = 0;
    const counts = new Set<number>();
    for (let wave = 0; wave < waves; wave++) {
      expect(status[wave * 4 + 3]).toBe(0);
      counts.add(status[wave * 4]);
      for (let slot = 0; slot < status[wave * 4]; slot++) {
        const row = wave * capacity + slot;
        expect(provenance[row * 4]).not.toBe(1);
        mass += weights[row];
        moment += weights[row] * positions[row * 2];
      }
    }
    expect([...counts].sort()).toEqual([3, 4]);
    expect(Math.abs(mass / waves - 1.1)).toBeLessThan(0.012);
    expect(Math.abs(moment / waves - 1.1 ** 2 / 2)).toBeLessThan(0.012);
  } finally {
    harness.destroy();
  }
});
