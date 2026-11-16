// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {TERRAIN_VIEWSHED_BATCH_START_WGSL} from '../../src/gpu-terrain/terrain-analysis/terrain-viewshed-batches';
import {createTerrainSweepSectorTable} from '../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {GPUTerrainHorizon} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createKernelAuditContext} from '../analysis-core/kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../utils/gpu-contributor-test-utils';

it('addresses independent viewshed batches exactly beyond uint32 products', () => {
  const cases = [
    [0, 23, 37],
    [1, 23, 37],
    [123456, 100000, 333333],
    [1000000, 1000000, 2000001]
  ];
  const input = Uint32Array.from(cases.flat());
  const output = new Uint32Array(cases.length * 2);
  const source = `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
${TERRAIN_VIEWSHED_BATCH_START_WGSL}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
 let index = invocation.x;
 output[index] = getViewshedBatchStart(input[3u * index], input[3u * index + 1u], input[3u * index + 2u]);
}`;
  expect(auditWGSLModule(source)).toEqual([]);
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', cases.length, {
    0: {0: input, 1: output}
  });
  expect(Array.from(output)).toEqual(
    cases.flatMap(([batch, size, pixels]) => {
      const pair = BigInt(batch) * BigInt(size);
      return [Number(pair / BigInt(pixels)), Number(pair % BigInt(pixels))];
    })
  );
});

it('initializes every sweep record word in parallel', () => {
  const context = createKernelAuditContext('parallel-sweep-records');
  try {
    const directions = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
      [1, 1]
    ] as const;
    const table = createTerrainSweepSectorTable(context.graph, {
      id: 'records',
      width: 4,
      height: 3,
      directions
    });
    expect(table.writeNode.workload?.maximumInvocationCount).toBe(35);
    const [kernel] = getGPUCommandNodeKernelSources([table.writeNode]);
    expect(auditWGSLModule(kernel.source)).toEqual([]);
  } finally {
    context.device.destroy();
  }
});

it('initializes horizon ranges with one invocation per range', () => {
  const context = createKernelAuditContext('parallel-horizon-ranges');
  try {
    const grid = {
      width: 4,
      height: 4,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const contributor = new GPUTerrainHorizon({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: context.view('float32', 16)}
      },
      parameters: context.view('float32', 8),
      directionCount: 16,
      maximumRadius: 2,
      maximumStepsPerDispatch: 1,
      output: {skyViewFactor: context.view('float32', 16)}
    });
    const initializer = contributor
      .getCommandNodes(context.graph)
      .find(node => node.id.endsWith('-sector-ranges'))!;
    expect(initializer.workload?.maximumInvocationCount).toBe(16);
    const [kernel] = getGPUCommandNodeKernelSources([initializer]);
    expect(auditWGSLModule(kernel.source)).toEqual([]);
  } finally {
    context.device.destroy();
  }
});
