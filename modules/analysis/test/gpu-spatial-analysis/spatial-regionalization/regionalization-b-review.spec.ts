// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {GPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {GPURedcapSingleLinkage} from '../../../src/gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createStandardizeColumnsNode} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-inputs';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getWorkgroupArgminWGSL} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-wgsl';

for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
  for (const data of [
    [1000000, 0, 2 / 128, 3 / 128],
    [2 ** 60, 0, 2, 3]
  ]) {
    it(`review 2 R1 preserves the small raw Ward edge: ${mergeSchedule}, first=${data[0]}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3, 5, 6)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 2, 1, 3, 2)},
          weights: {format: 'float32', values: new Float32Array(6).fill(1)},
          values: {format: 'float32', values: Float32Array.from(data)},
          parameters: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [
          new GPUWardRegionsOperation({
            rowCount: 4,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            values: 'values',
            parameters: 'parameters',
            standardize: false,
            targetRegionCount: 3,
            maximumIterations: 1,
            mergeSchedule,
            output: {labels: 'labels', objective: 'objective', status: 'status'}
          })
        ],
        outputs: ['labels', 'objective', 'status']
      });
      expect(Array.from(result['labels'])).toEqual([0, 1, 2, 2]);
      expect(Array.from(result['status'])).toEqual([1, 1, 0, 0]);
      expect(Array.from(result['objective'])).toEqual([data[2] === 2 ? 0.5 : 1 / 32768]);
    });
  }
}

it('W5-17 preserves the positive population score of a sub-ULP residual', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const values = rig.input(Float32Array.of(1, 2, -0.6, 0.8), 'float32', 4);
    const scaled = rig.output('float32', 4);
    rig.graph.add({
      getCommandNodes: graph => [
        createStandardizeColumnsNode(graph, {
          id: 'sign-standardize',
          operation: 'ResidualSignReview',
          values,
          scaled: scaled.view,
          rowCount: 4,
          columnCount: 1
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = (await readFloat32(scaled.buffer, 4))[3];
      // Float64 population score over the exact stored f32 inputs.
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeGreaterThan(0);
      expect(Math.abs(actual - 1.6068335e-8)).toBeLessThan(1e-13);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('review 2 R3 keeps the positive NumPy score after correcting the rounded mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const rowCount = 30000;
    const data = new Float32Array(rowCount).fill(0.3);
    data[0] = 0;
    data[1] = 0.5999;
    const values = rig.input(data, 'float32', rowCount);
    const scaled = rig.output('float32', rowCount);
    rig.graph.add({
      getCommandNodes: graph => [
        createStandardizeColumnsNode(graph, {
          id: 'residual-standardize',
          operation: 'ResidualMeanReview',
          values,
          scaled: scaled.view,
          rowCount,
          columnCount: 1
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(scaled.buffer, rowCount);
      // NumPy float64 population scores of the exact f32 inputs, in the adjacent oracle.
      // 3e-5 absolute for the two large scores, 1e-8 for the common residual score.
      expect(Number.isFinite(actual[0])).toBe(true);
      expect(Number.isFinite(actual[1])).toBe(true);
      expect(Math.abs(actual[0] + 122.494903)).toBeLessThan(3e-5);
      expect(Math.abs(actual[1] - 122.454068)).toBeLessThan(3e-5);
      for (const value of actual.slice(2)) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThan(0);
        expect(Math.abs(value - 1.36128029e-6)).toBeLessThan(1e-8);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

for (const withIsland of [false, true]) {
  it(`review 2 R4 retains the finite SKATER gain beside an extreme constant feature, island=${withIsland}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rowCount = withIsland ? 3 : 2;
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {
          format: 'uint32',
          values: Uint32Array.from(withIsland ? [0, 1, 2, 2] : [0, 1, 2])
        },
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)},
        values: {
          format: 'float32',
          values: Float32Array.from(
            withIsland ? [2e38, -1e19, 2e38, 1e19, -2e38, 7] : [2e38, -1e19, 2e38, 1e19]
          )
        },
        treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0)},
        componentLabels: {
          format: 'uint32',
          values: Uint32Array.from(withIsland ? [0, 0, 2] : [0, 0])
        },
        parameters: {format: 'uint32', values: Uint32Array.of(rowCount, 1)}
      },
      operations: [
        new GPUSkaterRegionsOperation({
          rowCount,
          edgeCount: 2,
          columnCount: 2,
          maximumRegionCount: rowCount,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          values: 'values',
          treeEdgeFlags: 'treeEdgeFlags',
          componentLabels: 'componentLabels',
          parameters: 'parameters',
          output: {
            labels: 'labels',
            regionCount: 'regionCount',
            cutEdges: 'cutEdges',
            cutGains: 'cutGains'
          }
        })
      ],
      outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
    });
    expect(Array.from(result['labels'])).toEqual(withIsland ? [0, 1, 2] : [0, 1]);
    expect(Array.from(result['regionCount'])).toEqual([rowCount]);
    expect(Array.from(result['cutEdges'])).toEqual(withIsland ? [0, 0xffffffff] : [0]);
    expect(Number.isFinite(result['cutGains'][0])).toBe(true);
    // The constant column contributes zero. The other contributes 2*float32(1e19)^2.
    expect(Math.abs(result['cutGains'][0] / 1.9999999360571385e38 - 1)).toBeLessThan(2e-6);
    if (withIsland) expect(result['cutGains'][1]).toBe(0);
  });
}

for (const invalid of [Infinity, NaN]) {
  it(`review 2 R4 preserves nonfinite-tree suppression for ${invalid}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
        weights: {format: 'float32', values: Float32Array.of(1, 1)},
        values: {format: 'float32', values: Float32Array.of(invalid, -1, invalid, 1)},
        treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0)},
        componentLabels: {format: 'uint32', values: Uint32Array.of(0, 0)},
        parameters: {format: 'uint32', values: Uint32Array.of(2, 1)}
      },
      operations: [
        new GPUSkaterRegionsOperation({
          rowCount: 2,
          edgeCount: 2,
          columnCount: 2,
          maximumRegionCount: 2,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          values: 'values',
          treeEdgeFlags: 'treeEdgeFlags',
          componentLabels: 'componentLabels',
          parameters: 'parameters',
          output: {
            labels: 'labels',
            regionCount: 'regionCount',
            cutEdges: 'cutEdges',
            cutGains: 'cutGains'
          }
        })
      ],
      outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
    });
    expect(Array.from(result['labels'])).toEqual([0, 0]);
    expect(Array.from(result['regionCount'])).toEqual([1]);
    expect(Array.from(result['cutEdges'])).toEqual([0xffffffff]);
    expect(Array.from(result['cutGains'])).toEqual([0]);
  });
}

for (const shift of [0, 1000000]) {
  it(`R1 preserves the raw Ward partition and SSD after translation ${shift}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3, 5, 6)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 2, 1, 3, 2)},
        weights: {format: 'float32', values: new Float32Array(6).fill(1)},
        values: {
          format: 'float32',
          values: Float32Array.from([0, 1 / 16, 3 / 16, 6 / 16], value => value + shift)
        },
        parameters: {format: 'uint32', values: Uint32Array.of(0)}
      },
      operations: [
        new GPUWardRegionsOperation({
          rowCount: 4,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          values: 'values',
          parameters: 'parameters',
          targetRegionCount: 2,
          maximumIterations: 2,
          standardize: false,
          output: {labels: 'labels', objective: 'objective', status: 'status'}
        })
      ],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual([0, 0, 0, 3]);
    expect(Array.from(result['status'])).toEqual([2, 1, 0, 0]);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    // Two f32 merge costs; NumPy centered-SSD oracle, 1e-8 absolute tolerance.
    expect(Math.abs(result['objective'][0] - 0.018229166666666668)).toBeLessThan(1e-8);
  });
}

it('R3 standardizes one zero and 65535 repeated values using the closed-form z-scores', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const rowCount = 65536;
    const data = new Float32Array(rowCount).fill(0.1);
    data[0] = 0;
    const values = rig.input(data, 'float32', rowCount);
    const scaled = rig.output('float32', rowCount);
    rig.graph.add({
      getCommandNodes: graph => [
        createStandardizeColumnsNode(graph, {
          id: 'review-standardize',
          operation: 'ReviewStandardize',
          values,
          scaled: scaled.view,
          rowCount,
          columnCount: 1
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(scaled.buffer, rowCount);
      // [0,c,...,c] has population z(0)=-sqrt(n-1), z(c)=1/sqrt(n-1), for any c>0.
      // Mean rounding matters relative to the small common score; 3e-5 absolute distinguishes
      // the old 0.0032997376 result while permitting the final f32 mean/division rounding.
      expect(Number.isFinite(actual[0])).toBe(true);
      expect(Math.abs(actual[0] + 255.99804686754936)).toBeLessThan(0.01);
      for (const value of actual.slice(1)) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThan(0);
        expect(Math.abs(value - 0.0039062798026634525)).toBeLessThan(3e-5);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('R4 publishes every SKATER output for a finite gain whose intermediate square overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
      weights: {format: 'float32', values: Float32Array.of(1, 1)},
      values: {format: 'float32', values: Float32Array.of(-1e19, 1e19)},
      treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0)},
      componentLabels: {format: 'uint32', values: Uint32Array.of(0, 0)},
      parameters: {format: 'uint32', values: Uint32Array.of(2, 1)}
    },
    operations: [
      new GPUSkaterRegionsOperation({
        rowCount: 2,
        edgeCount: 2,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        values: 'values',
        treeEdgeFlags: 'treeEdgeFlags',
        componentLabels: 'componentLabels',
        parameters: 'parameters',
        maximumRegionCount: 2,
        output: {
          labels: 'labels',
          regionCount: 'regionCount',
          cutEdges: 'cutEdges',
          cutGains: 'cutGains'
        }
      })
    ],
    outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
  });
  expect(Array.from(result['labels'])).toEqual([0, 1]);
  expect(Array.from(result['regionCount'])).toEqual([2]);
  expect(Array.from(result['cutEdges'])).toEqual([0]);
  expect(Number.isFinite(result['cutGains'][0])).toBe(true);
  // Closed form 2 * float32(1e19)^2, rounded to f32; permit 2e-6 relative rounding error.
  expect(Math.abs(result['cutGains'][0] / 1.9999999360571385e38 - 1)).toBeLessThan(2e-6);
});

it('R5 retains the zero-cost cycle edge even when sklearn changes its numeric ordering', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const labels = rig.output('uint32', 4);
    const regionCount = rig.output('uint32', 1);
    const converged = rig.output('uint32', 1);
    const iterationCount = rig.output('uint32', 1);
    const iterationLimitReached = rig.output('uint32', 1);
    const invalidCount = rig.output('uint32', 1);
    rig.graph.add(
      new GPURedcapSingleLinkage({
        weights: rig.uploadWeights({
          offsets: [0, 2, 4, 6, 8],
          neighbors: [1, 3, 0, 2, 1, 3, 0, 2],
          weights: new Array(8).fill(1),
          distances: []
        }),
        values: rig.input(Float32Array.of(0, 0, 1e-18, 3e-18), 'float32', 4),
        parameters: rig.input(Uint32Array.of(3), 'uint32', 1),
        standardize: false,
        componentIterations: 8,
        output: {
          labels: labels.view,
          regionCount: regionCount.view,
          status: {
            converged: converged.view,
            iterationCount: iterationCount.view,
            iterationLimitReached: iterationLimitReached.view,
            invalidCount: invalidCount.view
          }
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(labels.buffer, 4)).toEqual([0, 0, 2, 3]);
      expect(await readUint32(regionCount.buffer, 1)).toEqual([3]);
      expect(await readUint32(converged.buffer, 1)).toEqual([1]);
      expect(await readUint32(iterationCount.buffer, 1)).toEqual([8]);
      expect(await readUint32(iterationLimitReached.buffer, 1)).toEqual([0]);
      expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});

it('R7 retains the unique lane-2 minimum when six candidates are padded to eight lanes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const scores = rig.input(Float32Array.of(9, 8, -1, 7, 6, 5, 1e30, 1e30), 'float32', 8);
    const result = rig.output('uint32', 1);
    rig.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'six-padded-argmin',
          operation: 'ArgminReview',
          invocationCount: 8,
          workgroupSize: 8,
          guardIndex: false,
          declarations: getWorkgroupArgminWGSL(8),
          bindings: [
            {name: 'scores', view: scores, type: 'f32', access: 'read'},
            {name: 'result', view: result.view, type: 'u32', access: 'read_write'}
          ],
          body: `let winner = reduceArgmin(localInvocationIndex, scores[scoresOffset + index], 0u, index);
        if (localInvocationIndex == 0u) { result[resultOffset] = winner; }`
        })
      ]
    });
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Independent minimum of six real scores [9,8,-1,7,6,5] is at lane 2.
      expect(await readUint32(result.buffer, 1)).toEqual([2]);
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
