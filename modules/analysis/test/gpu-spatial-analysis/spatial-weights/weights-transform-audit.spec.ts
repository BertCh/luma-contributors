// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expect, it} from 'vitest';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {
  GPUSpatialWeightsTransform,
  type GPUSpatialWeightsKernel
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {
  TRANSFORM_AUDIT_INPUT as INPUT,
  TRANSFORM_AUDIT_EXPECTED as EXPECTED,
  KERNEL_AUDIT_EXPECTED
} from './weights-transform-audit-fixture';
import {expectTransformAuditResult} from './weights-transform-audit-fixture';

it.each([
  false,
  true
])('tiny double row partials and total match exact normal-word literals (chunked=%s)', async chunked => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights({
      offsets: [0, 1, 1, 1, 1, 1, 1, 1, 1],
      neighbors: [1],
      weights: [2 ** -125],
      distances: []
    });
    const output = rig.output('float32', 1),
      report = rig.output('float32', 9);
    const transform = chunked
      ? new GPUChunkedSpatialWeightsTransform({
          operation: 'double',
          doubleSum: 'rows',
          weights: {rowCount: 8, chunks: [{rowStart: 0, rowCount: 8, weights}]},
          output: [output.view]
        })
      : new GPUSpatialWeightsTransform({
          operation: 'double',
          doubleSum: 'rows',
          weights,
          output: output.view
        });
    const nodes = transform.getCommandNodes(rig.graph);
    const views = nodes.flatMap(node =>
      (node.resources ?? []).flatMap(resource =>
        'buffer' in resource && resource.buffer instanceof GraphDataView ? [resource.buffer] : []
      )
    );
    const partials = views.find(view => view.buffer.id.endsWith('-partials'))!,
      total = views.find(view => view.buffer.id.endsWith('-total'))!;
    expect(partials?.format).toBe('float32');
    expect(total?.format).toBe('float32');
    rig.graph.add(nodes);
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'audit-tiny-partials-total',
        operation: 'audit',
        invocationCount: 9,
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'total', view: total, type: 'f32', access: 'read'},
          {name: 'report', view: report.view, type: 'f32', access: 'read_write'}
        ],
        body: 'if(index<8u){report[reportOffset+index]=partials[partialsOffset+index];}else{report[reportOffset+index]=total[totalOffset];}'
      })
    );
    rig.run();
    expect(await readFloat32(report.buffer, 9)).toEqual([
      2 ** -125,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2 ** -125
    ]);
    expect(await readFloat32(output.buffer, 1)).toEqual([8]);
  } finally {
    rig.destroy();
  }
});

it('fixed kernel turns a one-way stored zero into a positive weight without claiming symmetry', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const weights = {
        ...rig.uploadWeights({offsets: [0, 1, 1], neighbors: [1], weights: [0], distances: [0.5]}),
        information: {rowCount: 2, transform: 'raw' as const, symmetric: true as const}
      };
      const output = rig.output('float32', 1);
      const transform = chunked
        ? new GPUChunkedSpatialWeightsTransform({
            operation: 'kernel',
            bandwidth: 1,
            weights: {
              rowCount: 2,
              chunks: [{rowStart: 0, rowCount: 2, weights}],
              information: weights.information
            },
            output: [output.view]
          })
        : new GPUSpatialWeightsTransform({
            operation: 'kernel',
            bandwidth: 1,
            weights,
            output: output.view
          });
      rig.run(transform);
      expect(await readFloat32(output.buffer, 1)).toEqual([0.5]);
      expect(transform.outputWeights.information?.symmetric).toBe('unknown');
    } finally {
      rig.destroy();
    }
  }
});

it('binary packed and chunked transforms preserve disjoint distances in the destination buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const packed = rig.input(
        Float32Array.from({length: 66}, (_, index) => (index === 0 ? 0.5 : index === 1 ? 1.5 : 99)),
        'float32',
        66
      );
      const distances = rig.graph.createDataView(packed.buffer, {format: 'float32', length: 2}),
        output = rig.graph.createDataView(packed.buffer, {
          format: 'float32',
          byteOffset: 256,
          length: 2
        });
      const weights = {
        ...rig.uploadWeights({
          offsets: [0, 1, 2],
          neighbors: [1, 0],
          weights: [2, 0],
          distances: []
        }),
        distances
      };
      rig.run(
        chunked
          ? new GPUChunkedSpatialWeightsTransform({
              operation: 'binary',
              weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
              output: [output]
            })
          : new GPUSpatialWeightsTransform({operation: 'binary', weights, output})
      );
      expect(await readFloat32(rig.bufferOf(packed), 66)).toEqual([
        0.5,
        1.5,
        ...Array(62).fill(99),
        1,
        0
      ]);
    } finally {
      rig.destroy();
    }
  }
});

it.each([
  {chunked: false, inPlace: false},
  {chunked: false, inPlace: true},
  {chunked: true, inPlace: false},
  {chunked: true, inPlace: true}
])('double rows returns exactly eight (chunked=$chunked inPlace=$inPlace)', async ({
  chunked,
  inPlace
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights({
      offsets: [0, 1, 1, 1, 1, 1, 1, 1, 1],
      neighbors: [1],
      weights: [2 ** -125],
      distances: []
    });
    const output = rig.output('float32', 1);
    const transform = chunked
      ? new GPUChunkedSpatialWeightsTransform({
          operation: 'double',
          doubleSum: 'rows',
          weights: {rowCount: 8, chunks: [{rowStart: 0, rowCount: 8, weights}]},
          output: inPlace ? undefined : [output.view]
        })
      : new GPUSpatialWeightsTransform({
          operation: 'double',
          doubleSum: 'rows',
          weights,
          output: inPlace ? undefined : output.view
        });
    rig.run(transform);
    expect(await readFloat32(inPlace ? rig.bufferOf(weights.weights) : output.buffer, 1)).toEqual([
      8
    ]);
  } finally {
    rig.destroy();
  }
});

it.each([
  false,
  true
])('average-reciprocal keeps large reciprocal weights finite and bit-exact (chunked=%s)', async chunked => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [sourceValues, expected] of [
    [[2e38, 2e38], 1.9999999360571385e38],
    [[2e38, 3e38], 2.500000072189495e38],
    [[2e38, 0], 9.999999680285692e37]
  ] as const) {
    const rig = new WeightsRig(device);
    try {
      const weights = rig.uploadWeights({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [...sourceValues],
        distances: []
      });
      const output = rig.output('float32', 2);
      const transform = chunked
        ? new GPUChunkedSpatialWeightsTransform({
            operation: 'average-reciprocal',
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}]},
            output: [output.view]
          })
        : new GPUSpatialWeightsTransform({
            operation: 'average-reciprocal',
            weights,
            output: output.view
          });
      rig.run(transform);
      expect(await readFloat32(output.buffer, 2)).toEqual([expected, expected]);
    } finally {
      rig.destroy();
    }
  }
});

it('all transforms match independent matrix literals in place and out of place, including rescaled rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const inPlace of [false, true]) {
    for (const operation of [
      'row',
      'binary',
      'double',
      'variance',
      'average-reciprocal'
    ] as const) {
      if (inPlace && operation === 'average-reciprocal') continue; // Explicitly requires separate output.
      for (const scale of [1, 2, 1e20, 1e-30]) {
        for (const doubleSum of operation === 'double' ? (['one', 'rows'] as const) : [undefined]) {
          const rig = new WeightsRig(device);
          try {
            const sourceWeights = INPUT.weights.map(value => value * scale);
            const weights = rig.uploadWeights({...INPUT, weights: sourceWeights}, 2);
            const output = rig.output('float32', weights.neighbors.length);
            output.buffer.write(new Float32Array(13).fill(-99));
            rig.bufferOf(weights.weights).write(Float32Array.from([...sourceWeights, -99, -99]));
            rig.run(
              new GPUSpatialWeightsTransform({
                operation,
                doubleSum,
                weights,
                output: inPlace ? undefined : output.view
              })
            );
            const expected =
              operation === 'double'
                ? EXPECTED[doubleSum === 'rows' ? 'doubleRows' : 'doubleOne']
                : operation === 'average-reciprocal'
                  ? EXPECTED['average-reciprocal'].map(value => value * scale)
                  : EXPECTED[operation];
            expectTransformAuditResult(
              await readFloat32(inPlace ? rig.bufferOf(weights.weights) : output.buffer, 13),
              expected,
              operation,
              scale,
              `${operation} ${scale} inPlace=${inPlace}`
            );
          } finally {
            rig.destroy();
          }
        }
      }
    }
  }
});

it('distance kernels match independent profiles and reject negative and nonfinite z', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const inPlace of [false, true]) {
    for (const kernel of Object.keys(KERNEL_AUDIT_EXPECTED) as GPUSpatialWeightsKernel[]) {
      const rig = new WeightsRig(device);
      try {
        const weights = rig.uploadWeights(INPUT, 2);
        const output = rig.output('float32', weights.neighbors.length);
        rig.bufferOf(weights.weights).write(Float32Array.from([...INPUT.weights, -99, -99]));
        output.buffer.write(new Float32Array(13).fill(-99));
        rig.run(
          new GPUSpatialWeightsTransform({
            operation: 'kernel',
            kernel,
            bandwidth: 1,
            weights,
            output: inPlace ? undefined : output.view
          })
        );
        expectNumericArrayClose(
          await readFloat32(inPlace ? rig.bufferOf(weights.weights) : output.buffer, 13),
          [...KERNEL_AUDIT_EXPECTED[kernel], -99, -99],
          // Float32 exp and the handful of profile multiplies/divides.
          {label: kernel, absoluteTolerance: 2e-6, relativeTolerance: 2e-6}
        );
      } finally {
        rig.destroy();
      }
    }
  }
});

it('double standardization writes zero everywhere after a finite-input row sum overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const inPlace of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const sourceWeights = [2e38, 2e38, 1, 2, 0, 2, 1, 0, 2, 2, 0];
      const weights = rig.uploadWeights({...INPUT, weights: sourceWeights}, 2);
      const output = rig.output('float32', 13);
      output.buffer.write(new Float32Array(13).fill(-99));
      rig.bufferOf(weights.weights).write(Float32Array.from([...sourceWeights, -99, -99]));
      rig.run(
        new GPUSpatialWeightsTransform({
          operation: 'double',
          weights,
          output: inPlace ? undefined : output.view
        })
      );
      expectNumericArrayClose(
        await readFloat32(inPlace ? rig.bufferOf(weights.weights) : output.buffer, 13),
        [...Array(11).fill(0), -99, -99],
        {absoluteTolerance: 0, relativeTolerance: 0}
      );
    } finally {
      rig.destroy();
    }
  }
});
