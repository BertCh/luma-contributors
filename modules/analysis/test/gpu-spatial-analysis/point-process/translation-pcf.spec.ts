// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUInhomogeneousPointPattern,
  type GPUInhomogeneousPointPatternProps
} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './translation-pcf-fixtures.json';
import complexWindow from './polygon-overlap-fixture.json';

type Input = {
  vertices: number[][];
  ringOffsets: number[];
  polygonOffsets: number[];
  positions: number[][];
  radii: number[];
  intensity?: number[];
  bandwidth?: number;
  selfContribution?: 'include' | 'exclude';
  pairCorrelation?: GPUInhomogeneousPointPatternProps['pairCorrelation'];
  maximumOverlapEdgePairs?: number;
};
function prepare(device: Device, input: Input) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => {
    const buffer = device.createBuffer({
      data: format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `translation-${buffers.length}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const window = new GPUStudyWindowPreparation({
    positions: view('float32x2', input.vertices.flat()).view,
    ringOffsets: view('uint32', input.ringOffsets).view,
    polygonOffsets: view('uint32', input.polygonOffsets).view,
    featureOffsets: view('uint32', [0, input.polygonOffsets.length - 1]).view,
    samplingGrid: [8, 8]
  });
  graph.add(window);
  const intensity = view('float32', new Array(input.positions.length).fill(0));
  const k = view('float32', new Array(input.radii.length).fill(0));
  const pairCorrelation = view('float32', new Array(input.radii.length).fill(0));
  const status = view('uint32', new Array(input.radii.length).fill(0));
  const supplied = input.intensity ? view('float32', input.intensity) : undefined;
  graph.add(
    new GPUInhomogeneousPointPattern({
      positions: view('float32x2', input.positions.flat()).view,
      studyWindow: window,
      edgeCorrection: 'translation',
      ...(input.intensity
        ? {suppliedIntensity: supplied!.view}
        : {bandwidth: input.bandwidth, selfContribution: input.selfContribution}),
      radii: view('float32', input.radii).view,
      pairCorrelation: input.pairCorrelation,
      maximumOverlapEdgePairs: input.maximumOverlapEdgePairs,
      output: {
        intensity: intensity.view,
        k: k.view,
        pairCorrelation: pairCorrelation.view,
        status: status.view
      }
    })
  );
  const compiled = graph.compile();
  return {
    execute: () => submitGraph(device, compiled, undefined),
    updateIntensity: (values: number[]) => supplied!.buffer.write(Float32Array.from(values)),
    async read() {
      return {
        intensity: await readFloat32(intensity.buffer, input.positions.length),
        k: await readFloat32(k.buffer, input.radii.length),
        pairCorrelation: await readFloat32(pairCorrelation.buffer, input.radii.length),
        status: await readUint32(status.buffer, input.radii.length)
      };
    },
    destroy() {
      compiled.destroy();
      window.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}
for (const fixture of fixtures.fixtures)
  for (const mode of fixture.modes) {
    it(`translation K and smoothed pcf: pinned spatstat ${fixture.name}, ${mode.mode} intensity`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const correlation of mode.correlations) {
        const input: Input = {
          ...fixture,
          ...(mode.mode === 'supplied'
            ? {intensity: mode.intensity}
            : {selfContribution: mode.mode as 'include' | 'exclude'}),
          pairCorrelation: {
            kernel: 'epanechnikov',
            bandwidth: fixture.pairBandwidth,
            divisor: correlation.divisor as 'radius' | 'distance',
            zeroCorrection: correlation.zeroCorrection as 'none' | 'reflection'
          }
        };
        const state = prepare(device, input);
        try {
          state.execute();
          const actual = await state.read();
          expect(actual.status).toEqual(fixture.radii.map(() => 0));
          const relativeTolerance = mode.mode === 'supplied' ? 0.0003 : 0.012;
          mode.intensity.forEach((value, index) =>
            expect(Math.abs(actual.intensity[index] - value) / value).toBeLessThan(
              mode.mode === 'supplied' ? 0.000001 : 0.003
            )
          );
          mode.k.forEach((value, index) =>
            expect(Math.abs(actual.k[index] - value)).toBeLessThan(
              0.00001 + relativeTolerance * Math.abs(value)
            )
          );
          correlation.exact.forEach((value, index) =>
            expect(Math.abs(actual.pairCorrelation[index] - value)).toBeLessThan(
              0.00001 + relativeTolerance * Math.abs(value)
            )
          );
          // Direct upstream functions use rasterized set covariance for polygon windows.
          mode.directK.forEach((value, index) => {
            if (value !== null)
              expect(Math.abs(actual.k[index] - value)).toBeLessThan(
                0.0002 + 0.02 * Math.abs(value)
              );
          });
          correlation.direct.forEach((value, index) => {
            if (value !== null)
              expect(Math.abs(actual.pairCorrelation[index] - value)).toBeLessThan(
                0.0002 + 0.02 * Math.abs(value)
              );
          });
        } finally {
          state.destroy();
        }
      }
    }, 60000);
  }
const rectangle = {
  vertices: [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
    [0, 0]
  ],
  ringOffsets: [0, 5],
  polygonOffsets: [0, 1]
};
it('translation boundary distances, duplicate events, outside/missing points and annular compatibility', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const state = prepare(device, {
    ...rectangle,
    positions: [
      [1, 1],
      [1, 1],
      [2, 1],
      [50, 50],
      [NaN, 2]
    ],
    intensity: [1, 1, 1, 0, NaN],
    radii: [0.5, 1, 2]
  });
  try {
    state.execute();
    const actual = await state.read();
    expect(actual.status).toEqual([0, 0, 0]);
    expect(actual.k[0]).toBeCloseTo(2 / 100, 6);
    expect(actual.k[1]).toBeCloseTo(2 / 100 + 4 / 90, 6);
    expect(actual.k[2]).toBeCloseTo(actual.k[1], 6);
    expect(actual.pairCorrelation[2]).toBe(0);
    expect(actual.intensity[3]).toBeNaN();
    expect(actual.intensity[4]).toBeNaN();
  } finally {
    state.destroy();
  }
});
for (const problem of [
  'budget',
  'invalid-intensity',
  'coincident-distance',
  'radii',
  'sliver'
] as const) {
  it(`translation reports ${problem} without presenting a partial estimate`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const state = prepare(device, {
      ...rectangle,
      positions:
        problem === 'sliver'
          ? [
              [0.000001, 1],
              [9.999999, 1]
            ]
          : problem === 'coincident-distance'
            ? [
                [1, 1],
                [1, 1]
              ]
            : [
                [1, 1],
                [2, 2]
              ],
      intensity: problem === 'invalid-intensity' ? [1, NaN] : [1, 1],
      radii: problem === 'radii' ? [2, 1] : [10],
      maximumOverlapEdgePairs: problem === 'budget' ? 1 : undefined,
      pairCorrelation:
        problem === 'coincident-distance'
          ? {kernel: 'epanechnikov', bandwidth: 1, divisor: 'distance'}
          : undefined
    });
    try {
      state.execute();
      const actual = await state.read();
      const expected = problem === 'budget' ? 3 : problem === 'sliver' ? 4 : 2;
      expect(actual.status.every(value => value === expected)).toBe(true);
      expect(actual.k.every(Number.isNaN)).toBe(true);
      expect(actual.pairCorrelation.every(Number.isNaN)).toBe(true);
    } finally {
      state.destroy();
    }
  });
}
it('4096 events reuse resident polygon overlaps across radii, including warm and updated execution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = Array.from({length: 4096}, (_, row) => [
    0.5 + (row % 64),
    0.5 + Math.floor(row / 64)
  ]);
  const radii = [0.9, 1.01, 1.45, 2.01];
  const started = performance.now();
  const state = prepare(device, {
    vertices: [
      [0, 0],
      [65, 0],
      [65, 65],
      [0, 65],
      [0, 0]
    ],
    ringOffsets: [0, 5],
    polygonOffsets: [0, 1],
    positions,
    intensity: positions.map(() => 1),
    radii,
    pairCorrelation: {kernel: 'epanechnikov', bandwidth: 0.2}
  });
  const preparationMilliseconds = performance.now() - started;
  try {
    const times = [];
    for (let repeat = 0; repeat < 3; repeat++) {
      const start = performance.now();
      state.execute();
      await device.handle.queue.onSubmittedWorkDone();
      times.push(performance.now() - start);
    }
    const actual = await state.read();
    expect(actual.status).toEqual(radii.map(() => 0));
    // Independent lattice displacement multiplicities and rectangle intersection formula.
    const expected = radii.map(radius => {
      let value = 0;
      for (let horizontal = -3; horizontal <= 3; horizontal++)
        for (let vertical = -3; vertical <= 3; vertical++) {
          if (horizontal === 0 && vertical === 0) continue;
          if (Math.hypot(horizontal, vertical) <= radius)
            value +=
              ((64 - Math.abs(horizontal)) * (64 - Math.abs(vertical))) /
              ((65 - Math.abs(horizontal)) * (65 - Math.abs(vertical)));
        }
      return value;
    });
    actual.k.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 3));
    state.updateIntensity(positions.map(() => 2));
    state.execute();
    const updated = await state.read();
    expect(updated.status).toEqual(radii.map(() => 0));
    updated.k.forEach((value, index) => expect(value).toBeCloseTo(expected[index] / 4, 4));
    updated.pairCorrelation.forEach((value, index) =>
      expect(value).toBeCloseTo(actual.pairCorrelation[index] / 4, 5)
    );
    console.log(
      `TRANSLATION_PCF_SCALE ${JSON.stringify({events: 4096, radii: 4, pairCount: (4096 * 4095) / 2, pairBytes: (8 * 4096 * 4095) / 2, preparationMilliseconds, executionMilliseconds: times})}`
    );
  } finally {
    state.destroy();
  }
}, 120000);

it('104-edge polygon with25holes agrees with pinned Shapely overlay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const state = prepare(device, {
    ...complexWindow,
    intensity: complexWindow.positions.map(() => 1)
  });
  try {
    state.execute();
    const actual = await state.read();
    expect(actual.status).toEqual(complexWindow.radii.map(() => 0));
    actual.k.forEach((value, index) => expect(value).toBeCloseTo(complexWindow.k[index], 5));
  } finally {
    state.destroy();
  }
});
