// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {TRAJECTORY_LOCATION_MEMBERSHIP_WGSL} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/trajectory-location-membership';
import {MODAL_SPEED_COMPARISON_WGSL} from '../../../src/gpu-spatial-analysis/trajectory-mobility/modal-speed-comparison';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
function executePredicates(declarations: string, calls: string[]): Uint32Array {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output = createTransientView(graph, 'result', 'uint32', calls.length);
    const node = createWGSLKernelNode(graph, {
      id: 'predicates',
      operation: 'ReviewPredicates',
      invocationCount: calls.length,
      bindings: [{name: 'result', view: output, type: 'u32', access: 'read_write'}],
      declarations,
      body: calls
        .map(
          (call, index) => `if(index==${index}u){result[resultOffset+index]=select(0u,1u,${call});}`
        )
        .join('\n')
    });
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([node], memory);
    return memory.readUint32(output);
  } finally {
    device.destroy();
  }
}
it('B5 executes exact WGSL membership for subtraction loss, tiny squares and inclusive boundaries', () => {
  const scalar = (value: number) => getWGSLFloatLiteral(Math.fround(value));
  const cases = [
    [-(2 ** -25), 0, 1, 0, 1],
    [0, 0, 1, 2 ** -26, 1],
    [0, 0, 1, 2 ** -13, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 3, 4, 5],
    [2 ** 100, 2 ** 100, 2 ** 100, 2 ** 100, 0]
  ];
  const calls = cases.map(
    values =>
      `withinRadius(vec2<f32>(${scalar(values[0])},${scalar(values[1])}),vec2<f32>(${scalar(values[2])},${scalar(values[3])}),${scalar(values[4])})`
  );
  expect(Array.from(executePredicates(TRAJECTORY_LOCATION_MEMBERSHIP_WGSL, calls))).toEqual([
    0, 0, 0, 1, 1, 1
  ]);
});
it('B4 executes WGSL exclusive cross-product comparisons without reciprocal division', () => {
  const scalar = (value: number) => getWGSLFloatLiteral(Math.fround(value));
  const cases = [
    [5, 25, 0.2],
    [15 / 3.6, 1, 15 / 3.6],
    [4.166666030883789, 1, 15 / 3.6],
    [1, 2, 0.5],
    [1, 2, 1]
  ];
  expect(
    Array.from(
      executePredicates(
        MODAL_SPEED_COMPARISON_WGSL,
        cases.map(values => `modalSpeedBelowBound(${values.map(scalar).join(',')})`)
      )
    )
  ).toEqual([1, 0, 1, 0, 1]);
});
it('B4 enforces binary32 nearest-even path-sum rounding using integer arithmetic', () => {
  const declaration =
    MODAL_SPEED_COMPARISON_WGSL +
    `\nfn modalFifteenSteps()->f32 {var total=0.0;for(var step=0u;step<15u;step++){total=modalAddLength(total,0.277777761220932);}return total;}`;
  expect(
    Array.from(
      executePredicates(declaration, [
        'modalAddLength(1.0,0.000000059604644775390625)==1.0',
        'modalAddLength(1.0,0.000000178813934326171875)==1.0000002384185791015625',
        'modalFifteenSteps()==4.166666030883789'
      ])
    )
  ).toEqual([1, 1, 1]);
});
import {GPUModalSplit} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split';
it('B4 executes the actual GPU prediction kernel for both reviewed triplegs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(format: Format, length = 1) =>
    createTransientView(graph, `predict-${sequence++}`, format, length);
  try {
    for (const multistep of [false, true]) {
      const rowCount = multistep ? 16 : 2;
      const positions = view('float32x2', rowCount),
        timestamps = view('float32', rowCount),
        starts = view('uint32'),
        ends = view('uint32'),
        count = view('uint32'),
        predicted = view('uint32');
      const nodes = new GPUModalSplit({
        id: `modal-${multistep}`,
        positions,
        timestamps,
        trackOffsets: view('uint32', 2),
        triplegs: {tracks: view('uint32'), startRows: starts, endRows: ends, status: {count}},
        modeCount: 2,
        metric: 'count',
        modeSpeedUpperBoundsFloat32: [multistep ? 15 / 3.6 : Math.fround(0.2), Infinity],
        output: {modalSplit: view('float32', 2), predictedModes: predicted}
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(
        positions,
        multistep
          ? Float32Array.from({length: 32}, (_, index) =>
              index % 2 === 0 && Math.floor(index / 2) % 2 === 1 ? 0.277777761220932 : 0
            )
          : Float32Array.of(0, 0, 5, 0)
      );
      memory.write(
        timestamps,
        multistep
          ? Float32Array.from({length: 16}, (_, index) => index / 15)
          : Float32Array.of(0, 25)
      );
      memory.write(starts, Uint32Array.of(0));
      memory.write(ends, Uint32Array.of(rowCount));
      memory.write(count, Uint32Array.of(1));
      const prediction = nodes.find(node => node.id === `modal-${multistep}-predict`);
      expect(prediction).toBeDefined();
      memory.write(predicted, Uint32Array.of(99));
      executeCommandNodesOnCPU([prediction!], memory);
      expect(memory.readUint32(predicted)).toEqual(Uint32Array.of(0));
    }
  } finally {
    device.destroy();
  }
});
