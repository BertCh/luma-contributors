// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsSummaryParameterValues,
  GPUSpatialWeightsSummary,
  getGPULatticeWeightsParameterValues,
  getGPUSpatialWeightsTransformParameterValues,
  GPULatticeWeights,
  GPUSpatialWeightsTransform,
  GPUChunkedSpatialWeightsTransform
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from './spatial-weights-harness';

it('fixed kernels read new bandwidths from an offset parameter view on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    const parameters = new GPUParameterBuffer(device, {
      id: 'bandwidth',
      format: 'float32',
      length: 3,
      values: new Float32Array([99, 2, 99])
    });
    const imported = parameters.importToGraph(rig.graph);
    const bandwidth = rig.graph.createDataView(imported.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: 4
    });
    const first = rig.uploadWeights(
      {offsets: [0, 1, 1, 2], neighbors: [2, 0], weights: [1, 1], distances: [1, 1]},
      2
    );
    const second = rig.uploadWeights(
      {offsets: [0, 1, 2], neighbors: [0, 3], weights: [1, 1], distances: [2, 2]},
      1
    );
    const firstOutput = rig.output('float32', 4),
      secondOutput = rig.output('float32', 3);
    rig.graph.add(
      chunked
        ? new GPUChunkedSpatialWeightsTransform({
            operation: 'kernel',
            bandwidth,
            weights: {
              rowCount: 5,
              chunks: [
                {rowStart: 0, rowCount: 3, weights: first},
                {rowStart: 3, rowCount: 2, weights: second}
              ]
            },
            output: [firstOutput.view, secondOutput.view]
          })
        : new GPUSpatialWeightsTransform({
            operation: 'kernel',
            bandwidth,
            weights: first,
            output: firstOutput.view
          })
    );
    const compiled = rig.graph.compile();
    try {
      for (const value of [2, 4, 0.5, 2]) {
        parameters.write(getGPUSpatialWeightsTransformParameterValues({bandwidth: value}), 1);
        submitGraph(device, compiled, undefined);
        const firstValues = await readFloat32(firstOutput.buffer, 4);
        expect(firstValues.slice(0, 2)).toEqual([
          Math.max(1 - 1 / value, 0),
          Math.max(1 - 1 / value, 0)
        ]);
        expect(firstValues[2]).toBe(firstValues[3]);
        expect(firstValues[2]).toBeGreaterThan(1e30); // Unused capacity is untouched.
        if (chunked) {
          const secondValues = await readFloat32(secondOutput.buffer, 3);
          expect(secondValues.slice(0, 2)).toEqual([
            Math.max(1 - 2 / value, 0),
            Math.max(1 - 2 / value, 0)
          ]);
          expect(secondValues[2]).toBeGreaterThan(1e30);
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  }
});

it('lattice distance scales update without changing topology, masks or capacity status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'cell-size',
    format: 'float32',
    length: 3,
    values: new Float32Array([99, 1, 1])
  });
  const imported = parameters.importToGraph(rig.graph);
  const cellSize = rig.graph.createDataView(imported.buffer, {
    format: 'float32',
    length: 2,
    byteOffset: 4
  });
  const output = rig.weightsOutput(4, 4, true);
  const overflow = rig.output('uint32', 1),
    requiredCount = rig.output('uint32', 1);
  const mask = rig.input(new Uint32Array([1, 0, 1, 1]), 'uint32', 4);
  rig.graph.add(
    new GPULatticeWeights({
      width: 2,
      height: 2,
      criterion: 'queen',
      cellSize,
      mask,
      output: {
        ...output.spatialWeights,
        status: {overflow: overflow.view, requiredCount: requiredCount.view}
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const scale of [
      [1, 1],
      [3, 4],
      [1, 1]
    ] as const) {
      parameters.write(getGPULatticeWeightsParameterValues({cellSize: scale}), 1);
      submitGraph(device, compiled, undefined);
      const actual = await output.read();
      expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
      expect(await readUint32(requiredCount.buffer, 1)).toEqual([6]);
      expect(actual.offsets).toEqual([0, 2, 2, 4, 4]);
      expect(actual.neighbors).toEqual([2, 3, 0, 3]);
      const diagonal = Math.hypot(...scale);
      for (const [index, expected] of [scale[1], diagonal, scale[1], scale[0]].entries()) {
        expect(actual.distances[index]).toBeCloseTo(expected, 5);
      }
    }
  } finally {
    compiled.destroy();
    parameters.destroy();
    rig.destroy();
  }
});

it('summary rereads offset symmetry tolerances without recompiling whole or chunked weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    const parameters = new GPUParameterBuffer(device, {
      id: 'summary-tolerance',
      format: 'float32',
      length: 3,
      values: new Float32Array([99, 0, 99])
    });
    const imported = parameters.importToGraph(rig.graph);
    const tolerance = rig.graph.createDataView(imported.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: 4
    });
    // Reverse weights differ by exactly 1/8; row 2 is empty and slot padding is ignored.
    const first = rig.uploadWeights(
      {offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1.125], distances: []},
      3
    );
    const empty = rig.uploadWeights(
      {offsets: [0, 0], neighbors: [], weights: [], distances: []},
      2
    );
    const whole = rig.uploadWeights(
      {offsets: [0, 1, 2, 2], neighbors: [1, 0], weights: [1, 1.125], distances: []},
      5
    );
    const statistics = rig.output('float32', 3);
    const counts = rig.output('uint32', 6);
    rig.graph.add(
      new GPUSpatialWeightsSummary({
        weights: chunked
          ? {
              rowCount: 3,
              chunks: [
                {rowStart: 0, rowCount: 2, weights: first},
                {rowStart: 2, rowCount: 1, weights: empty}
              ]
            }
          : whole,
        symmetryTolerance: tolerance,
        output: {statistics: statistics.view, counts: counts.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const value of [0, 0.125, 0.0625, 0.25, 0]) {
        parameters.write(getGPUSpatialWeightsSummaryParameterValues({symmetryTolerance: value}), 1);
        submitGraph(device, compiled, undefined);
        expect(await readUint32(counts.buffer, 6)).toEqual([2, value < 0.125 ? 2 : 0, 1, 0, 1, 1]);
        // Hand-derived: S0 = 17/8; S1 = (17/8)^2; S2 = 2 * (17/8)^2.
        expect(await readFloat32(statistics.buffer, 3)).toEqual([2.125, 4.515625, 9.03125]);
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  }
});
