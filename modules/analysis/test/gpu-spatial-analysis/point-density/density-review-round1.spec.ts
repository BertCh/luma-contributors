// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {
  GPULineDensity,
  getGPULineDensityParameterValues
} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createDensityExtremumNodes} from '../../../src/gpu-spatial-analysis/point-density/density-scratch';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('B2 three-cell diagnostic honors represented equality with either limiting axis on Metal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    parameterLength: 9,
    inputs: {positions: {format: 'float32x2', values: new Float32Array([0.5, 0.5])}},
    outputs: {values: {format: 'float32', length: 1}, resolution: {format: 'uint32', length: 1}},
    create: ({inputs, outputs, parameters}) =>
      new GPUKernelDensity({
        positions: inputs['positions'] as never,
        parameters,
        grid: makeExtentGrid(1, 1, [0, 0, 61, 1]),
        indexBounds: [0, 0, 61, 61],
        indexGridSize: [1, 1],
        method: 'binned',
        output: {
          values: outputs['values'] as never,
          status: {bandwidthBelowThreeCells: outputs['resolution'] as never}
        }
      })
  });
  try {
    for (const swapped of [false, true]) {
      for (const [cell, bandwidth, expected] of [
        [61, 183, 0],
        [15, 44.999996185302734, 1],
        [15, 45, 0]
      ]) {
        const grid = makeExtentGrid(1, 1, [0, 0, swapped ? 1 : cell, swapped ? cell : 1]);
        const result = await fixture.run(getGPUKernelDensityParameterValues({bandwidth}, grid));
        expect(result['resolution']).toEqual([expected]);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('B4 seam count distinguishes represented half-turn neighbors in both directions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array([179, 0, -1, 0])},
      offsets: {format: 'uint32', values: new Uint32Array([0, 2])}
    },
    outputs: {
      lengths: {format: 'float32', length: 1},
      overflow: {format: 'uint32', length: 1},
      seams: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPULineDensity({
        positions: inputs['positions'] as never,
        lineOffsets: inputs['offsets'] as never,
        parameters,
        columns: 1,
        rows: 1,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        output: {
          lengths: outputs['lengths'] as never,
          status: {
            overflow: outputs['overflow'] as never,
            antimeridianUncertainCount: outputs['seams'] as never
          }
        }
      })
  });
  try {
    for (const reversed of [false, true]) {
      for (const [second, expected] of [
        [-0.9999990463256836, 0],
        [-1, 0],
        [-1.0000009536743164, 1]
      ]) {
        fixture.writeInput(
          'positions',
          new Float32Array(reversed ? [second, 0, 179, 0] : [179, 0, second, 0])
        );
        const result = await fixture.run(
          getGPULineDensityParameterValues({minX: -180, minY: -0.5, cellWidth: 360, cellHeight: 1})
        );
        expect(result['seams']).toEqual([expected]);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  false,
  true
])('B6 automatic Scott bandwidth stays fixed when only the output grid pans (zero-weight outlier: %s)', async withZeroWeight => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    parameterLength: 9,
    inputs: {
      positions: {
        format: 'float32x2',
        values: new Float32Array([
          500000,
          5000000,
          500000.03125,
          5000000,
          500000,
          5000000.5,
          500000.03125,
          5000000.5,
          ...(withZeroWeight ? [0, 0] : [])
        ])
      },
      ...(withZeroWeight
        ? {weights: {format: 'float32' as const, values: new Float32Array([1, 1, 1, 1, 0])}}
        : {})
    },
    outputs: {values: {format: 'float32', length: 1}, bandwidth: {format: 'float32', length: 1}},
    create: ({inputs, outputs, parameters}) =>
      new GPUKernelDensity({
        positions: inputs['positions'] as never,
        weights: inputs['weights'] as never,
        parameters,
        grid: makeExtentGrid(1, 1, [0, 0, 2, 2]),
        indexBounds: [499999, 4999999, 500001, 5000001],
        indexGridSize: [2, 2],
        bandwidthRule: 'scott',
        edgeCorrection: 'none',
        output: {values: outputs['values'] as never, bandwidth: outputs['bandwidth'] as never}
      })
  });
  try {
    // Unbiased variances (.03125)^2/3 and (.5)^2/3; h=4^(-1/6)*sqrt((vx+vy)/2).
    for (const [startX, startY] of [
      [0, 0],
      [499999, 4999999],
      [-1000000, -1000000]
    ]) {
      const result = await fixture.run(
        getGPUKernelDensityParameterValues(
          {bandwidth: 99},
          makeExtentGrid(1, 1, [startX, startY, startX + 2, startY + 2])
        )
      );
      expect(result['bandwidth']).toHaveLength(1);
      expect(Number.isFinite(result['bandwidth'][0])).toBe(true);
      expect(result['bandwidth'][0]).toBeCloseTo(0.16232956552540634, 7);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  'quartic',
  'epanechnikov'
] as const)('B7 auto %s clears the diagnostic on its exact route', async kernel => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  const fixture = createGeometryFixture(device, {
    parameterLength: 9,
    inputs: {positions: {format: 'float32x2', values: new Float32Array([0.5, 0.5])}},
    outputs: {values: {format: 'float32', length: 1}, resolution: {format: 'uint32', length: 1}},
    create: ({inputs, outputs, parameters}) =>
      new GPUKernelDensity({
        positions: inputs['positions'] as never,
        parameters,
        grid,
        indexBounds: [0, 0, 1, 1],
        indexGridSize: [1, 1],
        method: 'auto',
        kernel,
        output: {
          values: outputs['values'] as never,
          status: {bandwidthBelowThreeCells: outputs['resolution'] as never}
        }
      })
  });
  try {
    for (const bandwidth of [0.5, 10]) {
      const result = await fixture.run(getGPUKernelDensityParameterValues({bandwidth}, grid));
      expect(result['resolution']).toEqual([0]);
      expect(result['values']).toHaveLength(1);
      expect(Number.isFinite(result['values'][0])).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('S2 auto clears a small-axis warning when only padding forces the exact fallback', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeExtentGrid(4, 3, [0, 0, 1, 12]);
  const fixture = createGeometryFixture(device, {
    parameterLength: 9,
    inputs: {positions: {format: 'float32x2', values: new Float32Array([0.5, 4])}},
    outputs: {values: {format: 'float32', length: 12}, resolution: {format: 'uint32', length: 1}},
    create: ({inputs, outputs, parameters}) =>
      new GPUKernelDensity({
        positions: inputs['positions'] as never,
        parameters,
        grid,
        indexBounds: [0, 0, 1, 12],
        indexGridSize: [4, 3],
        method: 'auto',
        binnedPadding: 32,
        output: {
          values: outputs['values'] as never,
          status: {bandwidthBelowThreeCells: outputs['resolution'] as never}
        }
      })
  });
  try {
    for (const [bandwidth, expected] of [
      [2, 1],
      [4, 0],
      [2, 1]
    ]) {
      const result = await fixture.run(getGPUKernelDensityParameterValues({bandwidth}, grid));
      expect(result['resolution']).toEqual([expected]);
      expect(result['values']).toHaveLength(12);
      expect(result['values'].every(Number.isFinite)).toBe(true);
      // One positive Gaussian event; the farthest center has d² = 6² + .375² < (5*2)².
      // Every cell must receive positive mass, so a discarded invalid command buffer cannot pass.
      expect(result['values'].every(value => value > 0)).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('t1 chunked finite extrema preserve signed coordinates and binding prefixes on Metal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunks of [
    [[-3, -0], [], [0, 1e-24, 1e20, 5]],
    [[-3], [-0, 0, 1e-24], [1e20], [5]]
  ]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const data = chunks.map((values, index) => {
      const buffer = createInputBuffer(device, new Float32Array([-1e30, -1e30, ...values, 1e30]));
      buffers.push(buffer);
      const whole = importGraphBuffer(
        graph,
        `values-${index}`,
        buffer,
        'float32',
        values.length + 3
      );
      return graph.createDataView(whole.buffer, {
        format: 'float32',
        byteOffset: 8,
        length: values.length
      });
    });
    const input = new GraphVectorView({
      id: 'chunked-values',
      name: 'chunked-values',
      format: 'float32',
      length: 6,
      valueLength: 6,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data
    });
    const output = createOutputBuffer(device, 4);
    output.write(new Float32Array([71, 72, 73, 74]));
    const allocation = importGraphBuffer(graph, 'extrema', output, 'float32', 4);
    for (const [operation, byteOffset] of [
      ['min', 4],
      ['max', 8]
    ] as const) {
      const result = graph.createDataView(allocation.buffer, {
        format: 'float32',
        byteOffset,
        length: 1
      });
      for (const node of createDensityExtremumNodes(
        graph,
        `extremum-${operation}`,
        input,
        result,
        operation
      ))
        graph.add(node);
    }
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const result = await readFloat32(output, 4);
      expect(result).toHaveLength(4);
      // Hand-selected finite extrema of [-3, -0, +0, 1e-24, 1e20, 5], not a second reduction.
      expect(Array.from(result)).toEqual([71, -3, Math.fround(1e20), 74]);
    } finally {
      compiled.destroy();
      output.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
