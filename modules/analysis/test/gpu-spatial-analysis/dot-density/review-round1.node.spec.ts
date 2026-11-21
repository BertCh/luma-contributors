// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUDotDensity,
  getGPUDotDensityParameterValues
} from '../../../src/gpu-spatial-analysis/dot-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {generateDotCountsOnCPU} from './dot-density-cpu';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const props = {
    id: 'review-dots',
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2),
    parameters: view('uint32', 8),
    values: view('float32', 1),
    mask: {weights: view('float32', 8), width: 1, height: 8, normalizeFeatures: true},
    output: {
      positions: view('float32x2', 8),
      dots: {
        ids: view('uint32', 8),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    }
  };
  return {graph, props};
}

it('B1 computes the normalized mask maximum with negative Y steps', () => {
  const {graph, props} = setup();
  const nodes = new GPUDotDensity(props).getCommandNodes(graph);
  const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
    kernel.id.endsWith('-feature-mask-maximum')
  )!.source;
  // Execute the exact production corner/range calculation independently of the cooperative
  // reductions: the interpreter cannot synchronize workgroup barriers. Bounding coordinates
  // and the expected maximum of these eight rows are hand-worked literals.
  const cornerRange = source.slice(source.indexOf('let origin ='), source.indexOf('var best ='));
  const maximum = createTransientView(graph, 'mask-maximum-probe', 'float32', 1);
  const probe = createWGSLKernelNode(graph, {
    id: 'corner-range-probe',
    operation: 'mask probe',
    invocationCount: 1,
    bindings: [
      {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
      {name: 'maskWeights', view: props.mask.weights, type: 'f32', access: 'read'},
      {name: 'maximumOut', view: maximum, type: 'f32', access: 'read_write'}
    ],
    declarations: `const MASK_WIDTH:u32=1u; const MASK_HEIGHT:u32=8u;
      fn isNanBits(value:f32)->bool { return (bitcast<u32>(value)&0x7fffffffu)>0x7f800000u; }
      var<private> partialMinimum:array<vec2f,1>; var<private> partialMaximum:array<vec2f,1>;`,
    body: `partialMinimum[0]=vec2f(0.25,0.25); partialMaximum[0]=vec2f(0.75,7.75);
      ${cornerRange}
      var best=0.0; for(var item=0u;item<cellCount;item++) {
        let row=u32(firstRow)+item/columnCount; let column=u32(firstColumn)+item%columnCount;
        best=max(best,maskWeights[maskWeightsOffset+row*MASK_WIDTH+column]);
      } maximumOut[maximumOutOffset]=best;`
  });
  const memory = createCPUGraphMemory();
  memory.write(props.positions, Float32Array.of(0.25, 0.25, 0.75, 0.25, 0.75, 7.75, 0.25, 7.75));
  memory.write(props.featureOffsets, Uint32Array.of(0, 1));
  memory.write(props.polygonOffsets, Uint32Array.of(0, 1));
  memory.write(props.ringOffsets, Uint32Array.of(0, 4));
  memory.write(
    props.parameters,
    getGPUDotDensityParameterValues({seed: 1, maskPlacement: {origin: [0, 8], cellSize: [1, -1]}})
  );
  memory.write(props.mask.weights, Float32Array.of(0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75));
  executeGeometryNodesOnCPU([probe], memory);
  // Maximum under all eight intersected rows is exactly 3/4, regardless of row direction.
  expect(memory.readFloat32(maximum)[0]).toBe(0.75);
});

it('S4 diagnoses missing compact status before snapshotting its fields', () => {
  const {props} = setup();
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        output: {...props.output, dots: {ids: props.output.dots.ids} as never}
      })
  ).toThrow(/^review-dots output.dots.status/);
});

it('S5 exposes the same CPU slotStarts array and the clamped final interval', () => {
  const result = generateDotCountsOnCPU({
    positions: Float32Array.of(0, 0, 2, 0, 0, 2, 10, 0, 12, 0, 10, 2),
    featureOffsets: Uint32Array.of(0, 1, 2),
    polygonOffsets: Uint32Array.of(0, 1, 2),
    ringOffsets: Uint32Array.of(0, 3, 6),
    counts: Uint32Array.of(2, 2),
    parameters: getGPUDotDensityParameterValues({seed: 1}),
    capacity: 3
  });
  expect(Array.from(result.slotStarts)).toEqual([0, 2]);
  expect(Math.min(result.slotStarts[1] + result.slotCounts[1], result.count)).toBe(3);
});
