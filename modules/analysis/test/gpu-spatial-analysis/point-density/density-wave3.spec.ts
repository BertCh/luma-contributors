// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {getWindowMassWGSL} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-wgsl';
import {getTemporalWindowMassWGSL} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-wgsl';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPUColorScale} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import {getGPUColorScaleParameterValues} from '../../../src/dataframe-analysis/column-classification/color-scale-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';

async function readFloat32Exactly(buffer: Buffer, length: number) {
  const values = await readFloat32(buffer, length);
  expect(values.length).toBe(length);
  return values;
}
async function readUint32Exactly(buffer: Buffer, length: number) {
  const values = await readUint32(buffer, length);
  expect(values.length).toBe(length);
  return values;
}

function expectRelative(actual: number, expected: number, tolerance = 2e-5) {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / expected - 1)).toBeLessThan(tolerance);
}

it('NUM-10/NUM-13 evaluates raw Gaussian intervals and compact caps before the mass cutoff', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // SciPy integrals of the encoded endpoints: float32(.999999) = .9999989867210388.
  for (const [kernel, expected] of [
    ['gaussian', [7.978845608028655e-9, 5.447074055985301e-9, 1.1285122074235991e-19]],
    ['epanechnikov', [9.92535781429567e-16, 9.92535781429567e-16, 7.700504297879542e-13]],
    ['quartic', [1.7240831669376526e-21, 1.7240831669376526e-21, 1.3004592835347933e-18]]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const output = createOutputBuffer(device, 3);
    const view = importGraphBuffer(graph, 'raw-masses', output, 'float32', 3);
    graph.add(
      createWGSLKernelNode(graph, {
        id: `raw-${kernel}`,
        operation: 'DensityOverlapProbe',
        invocationCount: 1,
        bindings: [{name: 'result', view, type: 'f32', access: 'read_write'}],
        declarations: `${getWindowMassWGSL(kernel)}\n${getTemporalWindowMassWGSL(kernel, kernel !== 'gaussian')}`,
        body:
          kernel === 'gaussian'
            ? `
result[resultOffset] = getTemporalWindowMass(0.0, -1e-8, 1e-8, 1.0);
result[resultOffset + 1u] = getWindowMass(vec2f(0.0), vec2f(-1e-8, -1.0), vec2f(1e-8, 1.0), 1.0);
result[resultOffset + 2u] = getTemporalWindowMass(0.0, 9.0, 10.0, 1.0);`
            : `
result[resultOffset] = getWindowMass(vec2f(0.0), vec2f(.999999, -1.0), vec2f(1.0), 1.0);
result[resultOffset + 1u] = getWindowMass(vec2f(0.0), vec2f(-1.0), vec2f(-.999999, 1.0), 1.0);
result[resultOffset + 2u] = getTemporalWindowMass(0.0, .999999, 1.0, 1.0);`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32Exactly(output, 3);
      for (let index = 0; index < 3; index++) expectRelative(actual[index], expected[index]);
    } finally {
      compiled.destroy();
      output.destroy();
    }
  }
});

it('UPT-11/UPT-10/UPT-15 preserves weight units and selects the requested bandwidth population', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rule of ['scott', 'silverman'] as const) {
    for (const kernel of ['gaussian', 'quartic'] as const) {
      for (const scale of [1, 1e20, 1e-24]) {
        for (const population of ['included', 'all-finite'] as const) {
          for (const appendOutlier of [false, true]) {
            const graph = new GPUCommandGraph(device);
            const buffers: Buffer[] = [];
            const track = (buffer: Buffer) => {
              buffers.push(buffer);
              return buffer;
            };
            const points = appendOutlier ? [0, 0, 1, 0, 0, 1, 1000, 1000] : [0, 0, 1, 0, 0, 1];
            const rowCount = points.length / 2;
            const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
            const positions = importGraphBuffer(
              graph,
              'positions',
              track(createInputBuffer(device, Float32Array.from(points))),
              'float32x2',
              rowCount
            );
            const weights = importGraphBuffer(
              graph,
              'weights',
              track(createInputBuffer(device, new Float32Array(rowCount).fill(scale))),
              'float32',
              rowCount
            );
            const parameterWords = getGPUKernelDensityParameterValues({bandwidth: 1}, grid);
            const parameters = importGraphBuffer(
              graph,
              'parameters',
              track(createInputBuffer(device, parameterWords)),
              'float32',
              9
            );
            const bandwidthBuffer = track(createOutputBuffer(device, 1));
            const weightBuffer = track(createOutputBuffer(device, 1));
            const countBuffer = track(createOutputBuffer(device, 1));
            const invalidBuffer = track(createOutputBuffer(device, 1));
            const valueBuffer = track(createOutputBuffer(device, 1));
            graph.add(
              new GPUKernelDensity({
                positions,
                weights,
                parameters,
                grid,
                indexGridSize: [1, 1],
                indexBounds: [-1, -1, 2, 2],
                edgeCorrection: 'none',
                kernel,
                bandwidthRule: rule,
                bandwidthPopulation: population,
                output: {
                  values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', 1),
                  bandwidth: importGraphBuffer(graph, 'bandwidth', bandwidthBuffer, 'float32', 1),
                  weightSum: importGraphBuffer(graph, 'weight-sum', weightBuffer, 'float32', 1),
                  bandwidthPopulationCount: importGraphBuffer(
                    graph,
                    'population-count',
                    countBuffer,
                    'uint32',
                    1
                  ),
                  status: {
                    invalidCount: importGraphBuffer(graph, 'invalid', invalidBuffer, 'uint32', 1)
                  }
                }
              })
            );
            const compiled = graph.compile();
            try {
              submitGraph(device, compiled, undefined);
              // SciPy gaussian_kde on the triangle; outlier changes only the all-finite population.
              expectRelative(
                (await readFloat32Exactly(bandwidthBuffer, 1))[0],
                (appendOutlier && population === 'all-finite'
                  ? 396.7181560077202
                  : 0.4807498567691361) * (kernel === 'quartic' ? 2.77919324 : 1)
              );
              expectRelative(
                (await readFloat32Exactly(weightBuffer, 1))[0],
                3 * Math.fround(scale)
              );
              expect(await readUint32Exactly(countBuffer, 1)).toEqual([
                population === 'included' ? 3 : rowCount
              ]);
              expect(await readUint32Exactly(invalidBuffer, 1)).toEqual([appendOutlier ? 1 : 0]);
            } finally {
              compiled.destroy();
              for (const buffer of buffers) buffer.destroy();
            }
          }
        }
      }
    }
  }
});

it('LIF-04 writes only the captured point-density destination after caller mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const original = createOutputBuffer(device, 2);
  const replacement = createOutputBuffer(device, 2);
  const points = createInputBuffer(device, new Float32Array([0.5, 0.5]));
  original.write(new Float32Array([81, 82]));
  replacement.write(new Float32Array([91, 92]));
  const props = {
    positions: importGraphBuffer(graph, 'positions', points, 'float32x2', 1),
    gridSize: [2, 1] as [number, number],
    bounds: [0, 0, 2, 1] as const,
    output: {values: importGraphBuffer(graph, 'original', original, 'float32', 2)}
  };
  const density = new GPUPointDensity(props);
  props.gridSize.reverse();
  props.output.values = importGraphBuffer(graph, 'replacement', replacement, 'float32', 2);
  graph.add(density);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readFloat32Exactly(original, 2))).toEqual([1, 0]);
    expect(Array.from(await readFloat32Exactly(replacement, 2))).toEqual([91, 92]);
  } finally {
    compiled.destroy();
    original.destroy();
    replacement.destroy();
    points.destroy();
  }
});

it('FST-06/PCK-09 rejects collapsed projected coordinates and evaluates recentered time slices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  expect(() =>
    getGPUKernelDensityParameterValues(
      {bandwidth: 1},
      {...grid, affine: [0.1, 0, 500000, 0, -0.1, 6000000]}
    )
  ).toThrow(/y.*origin/);
  const settings = {
    bandwidth: 1,
    temporalBandwidth: 1e-9,
    firstSliceTime: 0.7,
    sliceInterval: 1e-9
  };
  expect(() => getGPUSpaceTimeKernelDensityParameterValues(settings, grid, 3)).toThrow(
    /slice 1.*time origin/
  );
  const graph = new GPUCommandGraph(device);
  const positionsBuffer = createInputBuffer(device, new Float32Array([0.5, 0.5]));
  const timesBuffer = createInputBuffer(device, new Float32Array([0]));
  const parameterWords = getGPUSpaceTimeKernelDensityParameterValues(
    {...settings, firstSliceTime: 0},
    grid,
    3
  );
  const parameterBuffer = createInputBuffer(device, parameterWords);
  const output = createOutputBuffer(device, 3);
  graph.add(
    new GPUSpaceTimeKernelDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 1),
      times: importGraphBuffer(graph, 'times', timesBuffer, 'float32', 1),
      parameters: importGraphBuffer(graph, 'params', parameterBuffer, 'float32', 14),
      grid,
      indexGridSize: [1, 1],
      indexBounds: [0, 0, 1, 1],
      sliceCount: 3,
      spatialEdgeCorrection: 'none',
      temporalEdgeCorrection: 'none',
      output: {values: importGraphBuffer(graph, 'values', output, 'float32', 3)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32Exactly(output, 3);
    // Independent product of Gaussian densities: exp(-k²/2)/(2*pi*sqrt(2*pi)*float32(1e-9)).
    for (const [index, expected] of [
      63493637.72996369, 38510837.97990983, 8592929.445907515
    ].entries())
      expectRelative(actual[index], expected);
  } finally {
    compiled.destroy();
    positionsBuffer.destroy();
    timesBuffer.destroy();
    parameterBuffer.destroy();
    output.destroy();
  }
});

it('CMP-05 classifies and colors only the offset-preserving density grid prefix', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const input = (name: string, words: Float32Array) =>
    importGraphBuffer(
      graph,
      name,
      track(createInputBuffer(device, words)),
      'float32',
      words.length
    );
  const allocation = track(createOutputBuffer(device, 8));
  allocation.write(new Float32Array([91, 92, 77, 77, 77, 77, 1000, 93]));
  const handle = graph.importBuffer(
    {id: 'allocation', byteLength: allocation.byteLength, usage: allocation.usage},
    allocation
  );
  const grid = makeExtentGrid(2, 2, [0, 0, 2, 2]);
  const density = new GPUKernelDensity({
    positions: importGraphBuffer(
      graph,
      'positions',
      track(createInputBuffer(device, new Float32Array([0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5]))),
      'float32x2',
      4
    ),
    weights: input('weights', new Float32Array([2, 4, 6, 8])),
    parameters: input('density-params', getGPUKernelDensityParameterValues({bandwidth: 0.1}, grid)),
    grid,
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 2, 2],
    kernel: 'epanechnikov',
    edgeCorrection: 'none',
    output: {values: graph.createDataView(handle, {format: 'float32', byteOffset: 8, length: 5})}
  });
  graph.add(density);
  const values = density.outputBand.storage.values;
  const breaksBuffer = track(createOutputBuffer(device, 3));
  const classBuffer = track(createOutputBuffer(device, 1));
  const countsBuffer = track(createOutputBuffer(device, 2));
  const colorsBuffer = track(createOutputBuffer(device, 4));
  const breaks = importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', 3);
  const classCount = importGraphBuffer(graph, 'class-count', classBuffer, 'uint32', 1);
  graph.add(
    new GPUClassBreaks({
      values,
      maximumClassCount: 2,
      methods: ['equal-interval'],
      parameters: input(
        'breaks-params',
        getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
      ),
      output: {
        breaks,
        classCount,
        classCounts: importGraphBuffer(graph, 'class-counts', countsBuffer, 'uint32', 2)
      }
    })
  );
  graph.add(
    new GPUColorScale({
      values,
      domain: breaks,
      classCount,
      maximumDomainCount: 3,
      maximumPaletteCount: 2,
      palette: importGraphBuffer(
        graph,
        'palette',
        track(createInputBuffer(device, new Uint32Array([0xff0000ff, 0xff00ff00]))),
        'uint32',
        2
      ),
      parameters: input(
        'color-params',
        getGPUColorScaleParameterValues({scale: 'threshold', domainCount: 3, paletteCount: 2})
      ),
      output: {colors: importGraphBuffer(graph, 'colors', colorsBuffer, 'uint32', 4)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32Exactly(breaksBuffer, 3);
    for (const [index, expected] of [400 / Math.PI, 1000 / Math.PI, 1600 / Math.PI].entries())
      expectRelative(actual[index], expected);
    expect(await readUint32Exactly(countsBuffer, 2)).toEqual([2, 2]);
    expect(await readUint32Exactly(colorsBuffer, 4)).toEqual([
      0xff0000ff, 0xff0000ff, 0xff00ff00, 0xff00ff00
    ]);
    expect((await readFloat32Exactly(allocation, 8))[6]).toBe(1000);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('REVIEW B3/B4/B5 preserves off-center widths, clipped chords and centered compact intervals on GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kernel of ['gaussian', 'quartic'] as const) {
    const graph = new GPUCommandGraph(device);
    const output = createOutputBuffer(device, 4);
    graph.add(
      createWGSLKernelNode(graph, {
        id: `review-raw-${kernel}`,
        operation: 'DensityReviewProbe',
        invocationCount: 1,
        bindings: [
          {
            name: 'result',
            view: importGraphBuffer(graph, 'raw', output, 'float32', 4),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations:
          getWindowMassWGSL(kernel) + getTemporalWindowMassWGSL(kernel, kernel !== 'gaussian'),
        body:
          kernel === 'gaussian'
            ? `result[resultOffset] = getTemporalWindowMass(1.0, -1e-8, 1e-8, 1.0);
result[resultOffset + 1u] = getWindowMass(vec2f(1.0, 0.0), vec2f(-1e-8, -1.0), vec2f(1e-8, 1.0), 1.0);
result[resultOffset + 2u] = 1.0; result[resultOffset + 3u] = 1.0;`
            : `result[resultOffset] = getWindowMass(vec2f(0.0), vec2f(.999999, 0.0), vec2f(1.0), 1.0);
result[resultOffset + 1u] = getWindowMass(vec2f(0.0), vec2f(.995, 0.0), vec2f(1.0), 1.0);
result[resultOffset + 2u] = getWindowMass(vec2f(0.0), vec2f(.995, .02), vec2f(1.0), 1.0);
result[resultOffset + 3u] = getTemporalWindowMass(0.0, -2.9e-8, 2.9e-8, 1.0);`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32Exactly(output, 4);
      // Independent SciPy integrals on encoded endpoints (reference.py, review round 1).
      const expected =
        kernel === 'gaussian'
          ? [4.839414490382868e-9, 3.3038174206803345e-9, 1, 1]
          : [8.620415834688263e-22, 7.240316852684013e-9, 4.193236664650067e-9, 5.4375e-8];
      for (let index = 0; index < expected.length; index++)
        expectRelative(actual[index], expected[index], 1e-4);
    } finally {
      compiled.destroy();
      output.destroy();
    }
  }
});

it('REVIEW B2 uniform temporal correction keeps the fused final slice inside its default window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  const points = createInputBuffer(device, new Float32Array([0.5, 0.5]));
  const times = createInputBuffer(device, new Float32Array([0.1]));
  const parameters = createInputBuffer(
    device,
    getGPUSpaceTimeKernelDensityParameterValues(
      {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0.1, sliceInterval: 0.05},
      grid,
      8
    )
  );
  const output = createOutputBuffer(device, 8);
  graph.add(
    new GPUSpaceTimeKernelDensity({
      positions: importGraphBuffer(graph, 'points', points, 'float32x2', 1),
      times: importGraphBuffer(graph, 'times', times, 'float32', 1),
      parameters: importGraphBuffer(graph, 'params', parameters, 'float32', 14),
      grid,
      sliceCount: 8,
      indexGridSize: [1, 1],
      indexBounds: [0, 0, 1, 1],
      spatialEdgeCorrection: 'none',
      temporalEdgeCorrection: 'uniform',
      output: {values: importGraphBuffer(graph, 'output', output, 'float32', 8)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32Exactly(output, 8);
    for (const value of actual) expect(Number.isFinite(value)).toBe(true);
    // scipy quad(phi, float32(.1)-fma(7,float32(.05),float32(.1)), 0), divided into phi(offset)/(2*pi).
    expectRelative(actual[7], 0.4364618052979016);
  } finally {
    compiled.destroy();
    points.destroy();
    times.destroy();
    parameters.destroy();
    output.destroy();
  }
});

it('REVIEW B1/S3 compares unequal weights and zero-weight rows in both bandwidth populations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const population of ['included', 'all-finite'] as const) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const input = (name: string, data: Float32Array) =>
      importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, data)),
        'float32',
        data.length
      );
    const grid = makeExtentGrid(1, 1, [-1, -1, 1, 1]);
    // Positive weights differ; one zero-weight row is outside the active window, another inside.
    // The positive row just beyond x=.5 is the origin-translation membership regression.
    const positions = importGraphBuffer(
      graph,
      'positions',
      track(
        createInputBuffer(
          device,
          new Float32Array([0, 0, 0.25, 0, 0, 0.25, 0.75, 0, 0.5000000596046448, 0, 0.125, 0.125])
        )
      ),
      'float32x2',
      6
    );
    const bandwidthBuffer = track(createOutputBuffer(device, 1));
    const weightBuffer = track(createOutputBuffer(device, 1));
    const countBuffer = track(createOutputBuffer(device, 1));
    const invalidBuffer = track(createOutputBuffer(device, 1));
    graph.add(
      new GPUKernelDensity({
        positions,
        weights: input('weights', new Float32Array([1, 2, 4, 0, 3, 0])),
        parameters: input(
          'params',
          getGPUKernelDensityParameterValues({bandwidth: 1, window: [-1, -1, 0.5, 0.5]}, grid)
        ),
        grid,
        indexGridSize: [1, 1],
        indexBounds: [-1, -1, 2, 2],
        bandwidthRule: 'scott',
        bandwidthPopulation: population,
        edgeCorrection: 'uniform',
        output: {
          values: importGraphBuffer(
            graph,
            'values',
            track(createOutputBuffer(device, 1)),
            'float32',
            1
          ),
          bandwidth: importGraphBuffer(graph, 'bandwidth', bandwidthBuffer, 'float32', 1),
          weightSum: importGraphBuffer(graph, 'weight-sum', weightBuffer, 'float32', 1),
          bandwidthPopulationCount: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidBuffer, 'uint32', 1)}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Independent scipy.stats.gaussian_kde covariance trace and weighted effective count.
      expectRelative(
        (await readFloat32Exactly(bandwidthBuffer, 1))[0],
        population === 'included' ? 0.13605918693404473 : 0.17287395607033731
      );
      expect(Array.from(await readFloat32Exactly(weightBuffer, 1))).toEqual([7]);
      expect(await readUint32Exactly(countBuffer, 1)).toEqual([population === 'included' ? 4 : 6]);
      expect(await readUint32Exactly(invalidBuffer, 1)).toEqual([2]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('REVIEW B7 auto-route shader agrees at the reciprocal boundary and rejects zero encoded area', async () => {
  const {DENSITY_AUTO_ROUTE_WGSL} = await import(
    '../../../src/gpu-spatial-analysis/point-density/density-auto-route'
  );
  const {IS_FINITE_VALUE_WGSL} = await import('../../../src/analysis-core/kernels/wgsl-common');
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const controls of [
    [4.8, 0.75, 0.75],
    [1e-29, 1e-30, 1e-30]
  ]) {
    const graph = new GPUCommandGraph(device);
    const parameters = createInputBuffer(device, Float32Array.from(controls));
    const output = createOutputBuffer(device, 2);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'route-probe',
        operation: 'DensityReviewProbe',
        invocationCount: 1,
        bindings: [
          {
            name: 'params',
            view: importGraphBuffer(graph, 'params', parameters, 'float32', 3),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'result',
            view: importGraphBuffer(graph, 'result', output, 'uint32', 2),
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: IS_FINITE_VALUE_WGSL + DENSITY_AUTO_ROUTE_WGSL,
        body: `let bandwidth = params[paramsOffset]; let cell = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
result[resultOffset] = select(0u, 1u, isDensityFrameValid(bandwidth, cell));
result[resultOffset + 1u] = select(0u, 1u, useDensityBinnedRoute(bandwidth, cell, 33u));`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32Exactly(output, 2)).toEqual(controls[0] === 4.8 ? [1, 0] : [0, 0]);
    } finally {
      compiled.destroy();
      parameters.destroy();
      output.destroy();
    }
  }
});
