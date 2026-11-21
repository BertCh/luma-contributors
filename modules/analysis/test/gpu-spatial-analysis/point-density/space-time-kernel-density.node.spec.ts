// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  evaluateGPUSpaceTimeTemporalKernel,
  getGPUSpaceTimeKernelDensityParameterValues,
  getGPUSpaceTimeTemporalSupportRadius,
  getGPUSpaceTimeTemporalWindowOverlap,
  GPUSpaceTimeKernelDensity,
  GPUSpaceTimeRelativeRisk,
  type GPUSpaceTimeTemporalKernel
} from '../../../src/gpu-spatial-analysis/point-density/index';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

const KERNELS: GPUSpaceTimeTemporalKernel[] = ['gaussian', 'quartic', 'epanechnikov'];
const GRID = makeExtentGrid(4, 3, [0, 0, 4, 3]);

it('temporal kernels integrate to one over the line', () => {
  for (const kernel of KERNELS) {
    for (const bandwidth of [0.5, 2]) {
      const radius = getGPUSpaceTimeTemporalSupportRadius(kernel, bandwidth);
      const stepCount = 20000;
      const step = (2 * radius) / stepCount;
      let mass = 0;
      for (let index = 0; index < stepCount; index++) {
        mass +=
          evaluateGPUSpaceTimeTemporalKernel(kernel, bandwidth, -radius + (index + 0.5) * step) *
          step;
      }
      // The gaussian truncation at 5 bandwidths loses 5.7e-7 of the mass.
      expect(mass).toBeCloseTo(1, kernel === 'gaussian' ? 5 : 7);
    }
  }
});

it('temporal window mass is the integral of the kernel over the window', () => {
  for (const kernel of KERNELS) {
    for (const [time, window] of [
      [0.2, [-1, 3]],
      [0, [0, 10]],
      [2.4, [0, 2]],
      [9, [0, 2]],
      [-0.5, [-8, 8]]
    ] as const) {
      const bandwidth = 1.3;
      const stepCount = 200000;
      const low = Math.max(window[0], time - 12 * bandwidth);
      const high = Math.min(window[1], time + 12 * bandwidth);
      let integral = 0;
      if (high > low) {
        const step = (high - low) / stepCount;
        for (let index = 0; index < stepCount; index++) {
          // The gaussian mass is untruncated, so integrate the untruncated density.
          const offset = low + (index + 0.5) * step - time;
          integral +=
            (kernel === 'gaussian'
              ? Math.exp(-0.5 * (offset / bandwidth) ** 2) / Math.sqrt(2 * Math.PI) / bandwidth
              : evaluateGPUSpaceTimeTemporalKernel(kernel, bandwidth, offset)) * step;
        }
      }
      expect(getGPUSpaceTimeTemporalWindowOverlap(kernel, bandwidth, time, window)).toBeCloseTo(
        integral,
        7
      );
    }
  }
});

it('packs the parameter layout and rejects invalid settings', () => {
  const values = getGPUSpaceTimeKernelDensityParameterValues(
    {bandwidth: 0.5, temporalBandwidth: 2, firstSliceTime: 1, sliceInterval: 3},
    GRID,
    5
  );
  expect(Array.from(values)).toEqual([0, 0, 4, 3, 0.5, 0, 0, 4, 3, 2, 1, 3, 1, 13]);
  expect(
    Array.from(
      getGPUSpaceTimeKernelDensityParameterValues(
        {
          bandwidth: 0.5,
          temporalBandwidth: 2,
          firstSliceTime: 1,
          sliceInterval: 3,
          timeWindow: [-4, 20]
        },
        GRID,
        5
      ).slice(12)
    )
  ).toEqual([-4, 20]);
  const base = {bandwidth: 0.5, temporalBandwidth: 2, firstSliceTime: 1, sliceInterval: 3};
  for (const bad of [
    {...base, temporalBandwidth: 0},
    {...base, temporalBandwidth: Number.NaN},
    {...base, sliceInterval: -1},
    {...base, firstSliceTime: Number.POSITIVE_INFINITY},
    {...base, timeWindow: [3, 3] as const},
    {...base, bandwidth: -1},
    {...base, window: 'study-window' as const}
  ]) {
    expect(() => getGPUSpaceTimeKernelDensityParameterValues(bad, GRID, 5)).toThrow();
  }
  expect(() =>
    getGPUSpaceTimeKernelDensityParameterValues(base, GRID, 5, new Float32Array(13))
  ).toThrow(/target must hold 14/);
});

it('packs one slice without a time window (math correctness review, interpolation-density-07)', () => {
  // `sliceCount` is documented as "at least 1". The default window of one slice is the instant
  // of that slice; the packer used to reject it as unordered, also when the window is ignored.
  const base = {bandwidth: 0.5, temporalBandwidth: 2, firstSliceTime: 1, sliceInterval: 3};
  const values = getGPUSpaceTimeKernelDensityParameterValues(base, GRID, 1);
  expect(Array.from(values.slice(9))).toEqual([2, 1, 3, 1, 1]);
  // An explicit window still has to have a length, for one slice as for many.
  expect(() =>
    getGPUSpaceTimeKernelDensityParameterValues({...base, timeWindow: [1, 1]}, GRID, 1)
  ).toThrow(/start < end/);
  expect(
    Array.from(
      getGPUSpaceTimeKernelDensityParameterValues({...base, timeWindow: [0, 2]}, GRID, 1).slice(12)
    )
  ).toEqual([0, 2]);
  // The slice count itself is still checked.
  for (const sliceCount of [0, -1, 1.5, Number.NaN]) {
    expect(() => getGPUSpaceTimeKernelDensityParameterValues(base, GRID, sliceCount)).toThrow(
      /sliceCount/
    );
  }
});

function createDensity(overrides: Record<string, unknown> = {}, rows = 8, sliceCount = 5) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    times: createTransientView(graph, 'times', 'float32', rows),
    parameters: createTransientView(graph, 'parameters', 'float32', 14),
    grid: GRID,
    sliceCount,
    indexGridSize: [4, 3] as const,
    indexBounds: [0, 0, 4, 3] as const,
    output: {values: createTransientView(graph, 'values', 'float32', 12 * sliceCount)},
    ...overrides
  };
  return {device, graph, density: new GPUSpaceTimeKernelDensity(props as never)};
}

it('validates props at construction with the contributor ID and the prop', () => {
  expect(createDensity().density.voxelCount).toBe(60);
  expect(() => createDensity({spatialKernel: 'box'})).toThrow(
    /space-time-kernel-density spatialKernel.*box.*gaussian, quartic, epanechnikov, bisquare/
  );
  expect(() => createDensity({temporalKernel: 'box'})).toThrow(
    /space-time-kernel-density temporalKernel.*box.*gaussian, quartic, epanechnikov, bisquare/
  );
  expect(() => createDensity({temporalEdgeCorrection: 'diggle'})).toThrow(/edge corrections/);
  expect(() => createDensity({normalization: 'count'})).toThrow(/normalization/);
  expect(() => createDensity({sliceCount: 0})).toThrow(/sliceCount/);
  expect(() => createDensity({sliceCount: 1.5})).toThrow(/sliceCount/);
  expect(() => createDensity({indexBounds: [0, 0, -1, 3]})).toThrow(/indexBounds/);
  expect(() => createDensity({indexGridSize: [0, 3]})).toThrow(/indexGridSize/);
  expect(() => createDensity({sliceCount: 9})).toThrow(/output.values must hold/);
  expect(() =>
    createDensity({
      spatialContext: {coordinateSystem: 'longitude-latitude'}
    } as never)
  ).toThrow();
  const {graph} = createDensity();
  expect(() =>
    createDensity({times: createTransientView(graph, 'short-times', 'float32', 3)})
  ).toThrow(/times length must equal positions length/);
  expect(() =>
    createDensity({parameters: createTransientView(graph, 'short-parameters', 'float32', 13)})
  ).toThrow(/parameters must hold 14/);
  expect(() =>
    createDensity({
      temporalContext: {
        unit: 'second',
        epoch: 'unix',
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      }
    })
  ).toThrow(/float32-relative/);
  const declared = createDensity({
    temporalContext: {
      unit: 'second',
      epoch: 1_700_000_000n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    }
  });
  expect(declared.density.temporalContext?.unit).toBe('second');
});

it('emits kernels that pass the WGSL audit and stay within eight storage bindings', () => {
  for (const spatialKernel of ['gaussian', 'quartic'] as const) {
    for (const temporalKernel of KERNELS) {
      for (const edge of ['none', 'uniform'] as const) {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device);
        const rows = 16;
        const sliceCount = 4;
        const density = new GPUSpaceTimeKernelDensity({
          positions: createTransientView(graph, 'positions', 'float32x2', rows),
          times: createTransientView(graph, 'times', 'float32', rows),
          weights: createTransientView(graph, 'weights', 'float32', rows),
          parameters: createTransientView(graph, 'parameters', 'float32', 14),
          grid: GRID,
          sliceCount,
          indexGridSize: [4, 3],
          indexBounds: [0, 0, 4, 3],
          spatialKernel,
          temporalKernel,
          spatialEdgeCorrection: edge,
          temporalEdgeCorrection: edge,
          normalization: 'probability',
          output: {
            values: createTransientView(graph, 'values', 'float32', 12 * sliceCount),
            weightSum: createTransientView(graph, 'weight-sum', 'float32', 1),
            status: {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)}
          }
        });
        const nodes = density.getCommandNodes(graph);
        const sources = getGPUCommandNodeKernelSources(nodes);
        expect(sources.length).toBeGreaterThan(0);
        for (const {source, constants} of sources) {
          expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
          expect(auditWGSLModule(source, constants)).toEqual([]);
        }
        const gather = sources.find(({source}) =>
          source.includes('getTemporalKernelValue(sliceTime')
        );
        expect(gather).toBeDefined();
        // The erfc fit is declared exactly once however the two edge windows combine.
        expect(gather!.source.match(/fn erfcValue\(/g)?.length ?? 0).toBe(
          edge === 'uniform' && (spatialKernel === 'gaussian' || temporalKernel === 'gaussian')
            ? 1
            : 0
        );
        device.destroy();
      }
    }
  }
});

it('rejects a voxel count above the output binding limit when the nodes are built', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = 4;
  const density = new GPUSpaceTimeKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    times: createTransientView(graph, 'times', 'float32', rows),
    parameters: createTransientView(graph, 'parameters', 'float32', 14),
    grid: GRID,
    // 12 cells * 2^22 slices = 50331648 voxels, above the 2^25 rows a 128 MiB binding holds.
    sliceCount: 2 ** 22,
    indexGridSize: [4, 3],
    indexBounds: [0, 0, 4, 3],
    output: {values: createTransientView(graph, 'values', 'float32', 12 * 2 ** 22)}
  });
  expect(() => density.getCommandNodes(graph)).toThrow(/voxels/);
});

it('relative risk validates at construction and appends one ratio kernel to two densities', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32x2' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const props = {
    casePositions: view('case-positions', 'float32x2', 6),
    caseTimes: view('case-times', 'float32', 6),
    controlPositions: view('control-positions', 'float32x2', 9),
    controlTimes: view('control-times', 'float32', 9),
    parameters: view('parameters', 'float32', 14),
    grid: GRID,
    sliceCount: 3,
    indexGridSize: [4, 3] as const,
    indexBounds: [0, 0, 4, 3] as const,
    output: {values: view('risk', 'float32', 36)}
  };
  const risk = new GPUSpaceTimeRelativeRisk(props);
  expect(risk.scale).toBe('log');
  const nodes = risk.getCommandNodes(graph);
  const density = new GPUSpaceTimeKernelDensity({
    positions: props.casePositions,
    times: props.caseTimes,
    parameters: props.parameters,
    grid: GRID,
    sliceCount: 3,
    indexGridSize: [4, 3],
    indexBounds: [0, 0, 4, 3],
    output: {values: view('single', 'float32', 36)},
    normalization: 'probability'
  }).getCommandNodes(graph);
  expect(nodes.length).toBe(2 * density.length + 1);
  expect(() => new GPUSpaceTimeRelativeRisk({...props, scale: 'percent' as never})).toThrow(
    /scale/
  );
  expect(
    () => new GPUSpaceTimeRelativeRisk({...props, controlTimes: view('short', 'float32', 4)})
  ).toThrow(/times lengths/);
  expect(
    () => new GPUSpaceTimeRelativeRisk({...props, output: {values: view('small', 'float32', 10)}})
  ).toThrow(/output.values must hold/);
});
