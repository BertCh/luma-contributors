// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsTransform,
  getGPUSpatialWeightsKernelValue,
  type GPUSpatialWeightsKernel,
  type GPUSpatialWeightsTransformOperation
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeWeightsAuditKernel} from './weights-audit-executor';
import {
  TRANSFORM_AUDIT_INPUT as INPUT,
  TRANSFORM_AUDIT_EXPECTED as EXPECTED,
  KERNEL_AUDIT_EXPECTED
} from './weights-transform-audit-fixture';
import {
  getTransformAuditTolerance,
  expectTransformAuditResult
} from './weights-transform-audit-fixture';
import {computeTransformOracle, getKernelWeight} from './spatial-weights-oracle';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {SPATIAL_WEIGHTS_TRANSFORM_ARITHMETIC_WGSL} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-transform-arithmetic';

it('NAM-16 the independent oracle covers both quartic shape spellings', () => {
  // (15/16) * (1 - (3/4)^2)^2 = (15/16) * (7/16)^2 = 735/4096.
  for (const kernel of ['quartic', 'bisquare'] as const) {
    expect(getKernelWeight(kernel, 0.75, 1)).toBe(735 / 4096);
    expect(getGPUSpatialWeightsKernelValue(kernel, 0.75)).toBe(735 / 4096);
  }
  expect(Object.keys(KERNEL_AUDIT_EXPECTED).sort()).toEqual([
    'bisquare',
    'epanechnikov',
    'gaussian',
    'quartic',
    'triangular',
    'uniform'
  ]);
});

it('review S2 the oracle rejects unknown kernel names before checking the normalized distance', () => {
  for (const distance of [0.625, NaN, Infinity, -1]) {
    expect(() => getKernelWeight('mistyped', distance, 1)).toThrow(/kernel/);
    expect(() =>
      getGPUSpatialWeightsKernelValue('mistyped' as GPUSpatialWeightsKernel, distance)
    ).toThrow(/kernel/);
  }
  expect(getKernelWeight('uniform', 0.625, 1)).toBe(0.5);
});

it('double scale selection avoids overflowing quotients and keeps adjacent finite quotients', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const totals = createTransientView(graph, 'scale-totals', 'float32', 8),
      targets = createTransientView(graph, 'scale-targets', 'float32', 8),
      result = createTransientView(graph, 'scale-result', 'float32', 16);
    const [kernel] = getGPUCommandNodeKernelSources([
      createWGSLKernelNode(graph, {
        id: 'scales',
        operation: 'audit',
        invocationCount: 8,
        bindings: [
          {name: 'totals', view: totals, type: 'f32', access: 'read'},
          {name: 'targets', view: targets, type: 'f32', access: 'read'},
          {name: 'result', view: result, type: 'f32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_FLOAT_WGSL}\n${SPATIAL_WEIGHTS_TRANSFORM_ARITHMETIC_WGSL}`,
        body: 'let state=getSpatialWeightsDoubleScale(totals[totalsOffset+index],targets[targetsOffset+index]); result[resultOffset+2u*index]=state.value; result[resultOffset+2u*index+1u]=f32(state.useScaledQuotient);'
      })
    ]);
    const buffers = {
      totals: Float32Array.of(
        2 ** -125,
        2 ** -124,
        2 ** -125,
        2 ** -126,
        0,
        Infinity,
        1,
        1.5 * 2 ** -125
      ),
      targets: Float32Array.of(8, 8, 4, 8, 8, 8, 8, 8),
      result: new Float32Array(16)
    };
    executeWeightsAuditKernel(kernel, buffers);
    // Quotients 2^128 and 2^129 do not fit; 2^127 and (2^128)/1.5 do.
    expect(Array.from(buffers.result)).toEqual([
      0, 1, 1.7014118346046923e38, 0, 1.7014118346046923e38, 0, 0, 1, 0, 0, 0, 0, 8, 0,
      2.2685491804142884e38, 0
    ]);
  } finally {
    device.destroy();
  }
});

it('audit interpreter scalar frexp and ldexp obey literal IEEE binary32 decompositions', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const source = createTransientView(graph, 'builtin-source', 'float32', 8),
      result = createTransientView(graph, 'builtin-result', 'float32', 24);
    const [kernel] = getGPUCommandNodeKernelSources([
      createWGSLKernelNode(graph, {
        id: 'builtins',
        operation: 'audit',
        invocationCount: 8,
        bindings: [
          {name: 'source', view: source, type: 'f32', access: 'read'},
          {name: 'result', view: result, type: 'f32', access: 'read_write'}
        ],
        body: 'let parts=frexp(source[sourceOffset+index]); result[resultOffset+3u*index]=parts.fract; result[resultOffset+3u*index+1u]=f32(parts.exp); result[resultOffset+3u*index+2u]=ldexp(parts.fract,parts.exp);'
      })
    ]);
    const buffers = {
      source: Float32Array.of(
        2 ** -125,
        8,
        0,
        -2,
        3.4028234663852886e38,
        2 ** -126,
        2 ** -149,
        1.5
      ),
      result: new Float32Array(24)
    };
    executeWeightsAuditKernel(kernel, buffers);
    expect(Array.from(buffers.result)).toEqual([
      0.5,
      -124,
      2 ** -125,
      0.5,
      4,
      8,
      0,
      0,
      0,
      -0.5,
      2,
      -2,
      0.9999999403953552,
      128,
      3.4028234663852886e38,
      0.5,
      -125,
      2 ** -126,
      0.5,
      -148,
      2 ** -149,
      0.75,
      1,
      1.5
    ]);
  } finally {
    device.destroy();
  }
});

it('symmetrization tolerance rejects zero at the smallest audited positive scale', () => {
  expect(() =>
    expectNumericArrayClose([0], [2e-30], {absoluteTolerance: 2e-36, relativeTolerance: 2e-6})
  ).toThrow();
});

it.each([
  false,
  true
])('average-reciprocal preserves equal large finite reciprocal words (chunked=%s)', chunked => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `large-${serial++}`, format, length);
    const weights = {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      output = view('float32', 2);
    const transform = chunked
      ? new GPUChunkedSpatialWeightsTransform({
          operation: 'average-reciprocal',
          weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
          output: [output]
        })
      : new GPUSpatialWeightsTransform({operation: 'average-reciprocal', weights, output});
    const kernel = getGPUCommandNodeKernelSources(transform.getCommandNodes(graph)).find(kernel =>
      kernel.id.includes('-average-reciprocal')
    )!;
    for (const [sourceValues, expected] of [
      [[2e38, 2e38], 1.9999999360571385e38],
      [[2e38, 3e38], 2.500000072189495e38],
      [[2e38, 0], 9.999999680285692e37]
    ] as const) {
      const buffers = {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        source: Float32Array.from(sourceValues),
        values: new Float32Array(2),
        reverseOffsets: Uint32Array.of(0, 1, 2),
        reverseNeighbors: Uint32Array.of(1, 0),
        reverseWeights: Float32Array.of(sourceValues[1], sourceValues[0])
      };
      executeWeightsAuditKernel(kernel, buffers);
      expect(Array.from(buffers.values)).toEqual([expected, expected]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  {chunked: false, rowCount: 7},
  {chunked: false, rowCount: 8},
  {chunked: false, rowCount: 9},
  {chunked: true, rowCount: 7},
  {chunked: true, rowCount: 8},
  {chunked: true, rowCount: 9}
])('double rows retains a representable result when its scale overflows (chunked=$chunked rows=$rowCount)', ({
  chunked,
  rowCount
}) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `tiny-${serial++}`, format, length);
    const weights = {
        offsets: view('uint32', rowCount + 1),
        neighbors: view('uint32', 1),
        weights: view('float32', 1)
      },
      output = view('float32', 1);
    const transform = chunked
      ? new GPUChunkedSpatialWeightsTransform({
          operation: 'double',
          doubleSum: 'rows',
          weights: {rowCount, chunks: [{rowStart: 0, rowCount, weights}]},
          output: [output]
        })
      : new GPUSpatialWeightsTransform({operation: 'double', doubleSum: 'rows', weights, output});
    const kernels = getGPUCommandNodeKernelSources(transform.getCommandNodes(graph));
    const buffers = {
      offsets: Uint32Array.from({length: rowCount + 1}, (_, index) => Number(index > 0)),
      source: Float32Array.of(2 ** -125),
      values: Float32Array.of(99),
      partials: new Float32Array(rowCount),
      total: Float32Array.of(2 ** -125)
    };
    executeWeightsAuditKernel(kernels.find(kernel => kernel.id.includes('-partials'))!, buffers);
    expect(Array.from(buffers.partials)).toEqual([2 ** -125, ...Array(rowCount - 1).fill(0)]);
    executeWeightsAuditKernel(kernels.find(kernel => kernel.id.includes('-apply'))!, buffers);
    // One nonzero weight normalizes exactly to the target row count.
    expect(Array.from(buffers.values)).toEqual([rowCount]);
  } finally {
    device.destroy();
  }
});

it('fixed kernels cannot infer symmetric results from numerical symmetry of zero-weight input', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const chunked of [false, true]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `kernel-review-${serial++}`, format, length);
      const weights = {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 1),
        weights: view('float32', 1),
        distances: view('float32', 1),
        information: {rowCount: 2, transform: 'raw' as const, symmetric: true as const}
      };
      const output = view('float32', 1);
      const transform = chunked
        ? new GPUChunkedSpatialWeightsTransform({
            operation: 'kernel',
            bandwidth: 1,
            weights: {
              rowCount: 2,
              chunks: [{rowStart: 0, rowCount: 2, weights}],
              information: weights.information
            },
            output: [output]
          })
        : new GPUSpatialWeightsTransform({operation: 'kernel', bandwidth: 1, weights, output});
      const kernel = getGPUCommandNodeKernelSources(transform.getCommandNodes(graph)).find(record =>
        record.id.endsWith('-kernel')
      )!;
      const values = Float32Array.of(99);
      executeWeightsAuditKernel(kernel, {
        offsets: Uint32Array.of(0, 1, 1),
        distances: Float32Array.of(0.5),
        values,
        parameters: Float32Array.of(1)
      });
      expect(Array.from(values)).toEqual([0.5]);
      expect(transform.outputWeights.information?.symmetric).toBe('unknown');
    }
  } finally {
    device.destroy();
  }
});

function executeTransform(
  operation: GPUSpatialWeightsTransformOperation,
  scale: number,
  kernel?: GPUSpatialWeightsKernel,
  doubleSum?: 'one' | 'rows',
  sourceWeights = INPUT.weights
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `audit-${serial++}`, format, length);
    const nodes = new GPUSpatialWeightsTransform({
      operation,
      kernel,
      doubleSum,
      bandwidth: operation === 'kernel' ? 1 : undefined,
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 13),
        weights: view('float32', 13),
        distances: view('float32', 13)
      },
      output: view('float32', 13)
    }).getCommandNodes(graph);
    const buffers = {
      offsets: Uint32Array.from(INPUT.offsets),
      neighbors: Uint32Array.from([...INPUT.neighbors, 0, 0]),
      source: Float32Array.from([...sourceWeights.map(value => value * scale), -99, -99]),
      distances: Float32Array.from([...INPUT.distances, 0, 0]),
      values: new Float32Array(13).fill(-99),
      parameters: Float32Array.of(1),
      partials: new Float32Array(8),
      total: new Float32Array(1)
    };
    const records = getGPUCommandNodeKernelSources(nodes);
    if (operation === 'double' || operation === 'variance') {
      executeWeightsAuditKernel(records.find(record => record.id.endsWith('-partials'))!, buffers);
      // Only the upstream collective reduction is replaced by a host sum. All
      // first-party arithmetic is executed from the emitted WGSL, unchanged.
      buffers.total[0] = buffers.partials.reduce((total, value) => total + value, 0);
      executeWeightsAuditKernel(records.find(record => record.id.endsWith('-apply'))!, buffers);
    } else {
      executeWeightsAuditKernel(
        records.find(record => record.id.endsWith(`-${operation}`))!,
        buffers
      );
    }
    return Array.from(buffers.values);
  } finally {
    device.destroy();
  }
}

it('double standardization suppresses every row when S0 overflows float32', () => {
  // All input weights are finite float32, but 2e38 + 2e38 exceeds FLT_MAX.
  // The documented nonfinite-S0 convention is all zero, including other rows.
  const weights = [2e38, 2e38, 1, 2, 0, 2, 1, 0, 2, 2, 0];
  expect(executeTransform('double', 1, undefined, 'one', weights)).toEqual([
    ...Array(11).fill(0),
    -99,
    -99
  ]);
});

for (const operation of ['row', 'binary', 'double', 'variance', 'average-reciprocal'] as const) {
  it(`transform ${operation} agrees with the independent matrix definition`, () => {
    for (const scale of [1, 2, 1e20, 1e-30]) {
      for (const doubleSum of operation === 'double' ? (['one', 'rows'] as const) : [undefined]) {
        const expected =
          operation === 'double'
            ? EXPECTED[doubleSum === 'rows' ? 'doubleRows' : 'doubleOne']
            : operation === 'average-reciprocal'
              ? EXPECTED['average-reciprocal'].map(value => value * scale)
              : EXPECTED[operation];
        const actual = executeTransform(operation, scale, undefined, doubleSum);
        expectTransformAuditResult(
          actual,
          expected,
          operation,
          scale,
          `${operation} scale=${scale}`
        );
        expectNumericArrayClose(
          computeTransformOracle(
            {...INPUT, weights: INPUT.weights.map(value => value * scale)},
            operation,
            {doubleSum}
          ),
          expected,
          getTransformAuditTolerance(operation, scale, 'cpu')
        );
      }
    }
  });
}

for (const kernel of Object.keys(KERNEL_AUDIT_EXPECTED) as GPUSpatialWeightsKernel[]) {
  it(`kernel ${kernel} CPU helper and emitted WGSL match the independent profile`, () => {
    const expected = KERNEL_AUDIT_EXPECTED[kernel];
    expectNumericArrayClose(
      INPUT.distances.map(z => getGPUSpatialWeightsKernelValue(kernel, z)),
      expected,
      {
        absoluteTolerance: 1e-14,
        relativeTolerance: 1e-14
      }
    );
    expectNumericArrayClose(executeTransform('kernel', 1, kernel), [...expected, -99, -99], {
      absoluteTolerance: 2e-6,
      relativeTolerance: 2e-6
    });
  });
}
