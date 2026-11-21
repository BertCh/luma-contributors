// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {computeNetworkTimeRoutingOnCPU} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('piecewise identity routing audits the emitted cooperative source and traverses a zero-duration edge (B3)', () => {
  // Closed form: samples (0,0),(41,41) define A(t)=t, so A(1)=1. Both float32
  // division routes round the raw interpolation below 1; it must never improve a self-loop.
  for (const fraction of [Math.fround(1 / 41), Math.fround(1 * Math.fround(1 / 41))]) {
    const unguarded = Math.fround(41 * fraction);
    expect(unguarded).toBe(0.9999999403953552);
    expect(Math.max(1, unguarded)).toBe(1);
  }
  const referenceRouting = computeNetworkTimeRoutingOnCPU({
    stateVertices: Uint32Array.of(0, 1),
    transitions: Uint32Array.of(0, 0, 0, 1),
    costs: Float32Array.of(0, 0, 41, 41, 0, 0, 41, 41),
    endpoints: Uint32Array.of(0, 1),
    departure: Float32Array.of(1),
    costModel: 'piecewise-arrival',
    samplesPerTransition: 2
  });
  expect(referenceRouting.arrivals).toEqual(Float32Array.of(1, 1));
  expect(referenceRouting.predecessorTransitions).toEqual(Uint32Array.of(0xffffffff, 1));
  expect(referenceRouting.summary).toEqual(Uint32Array.of(1, 1, 1, 0x3f800000));
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `interpolation-${serial++}`, format, length);
  try {
    const contributor = new GPUNetworkTimeRouting({
      stateVertices: view('uint32', 2),
      transitions: view('uint32x2', 2),
      costs: view('float32x2', 4),
      endpoints: view('uint32', 2),
      departure: view('float32', 1),
      costModel: 'piecewise-arrival',
      samplesPerTransition: 2,
      output: {
        arrivals: view('float32', 2),
        elapsedTimes: view('float32', 2),
        predecessorTransitions: view('uint32', 2),
        summary: view('uint32', 4)
      }
    });
    const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
    let elapsedSource = sources.get('network-time-routing-elapsed')!;
    // Audit the unmodified cooperative source before removing synchronization for arithmetic.
    for (const kernel of sources.configurations) {
      expect.soft(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
    expect.soft(elapsedSource).not.toMatch(/if\s*\(index\s*>=\s*INVOCATION_COUNT\)/);
    expect(sources.has('network-time-routing')).toBe(false);
    // Project the emitted cooperative shader to one lane for arithmetic execution. The
    // interpreter has no barrier scheduler; the device fixture checks synchronization.
    elapsedSource = elapsedSource
      .replace(/@workgroup_size\(64\)/, '@workgroup_size(1)')
      .replaceAll('workgroupBarrier();', '')
      .replaceAll('workgroupUniformLoad(&running)', 'running')
      .replaceAll('+= 64u', '+= 1u')
      .replace(/index == [a-zA-Z]+ % 64u && /, '')
      .replaceAll('atomic<u32>', 'u32')
      .replace(/atomicLoad\(&([a-zA-Z]+)\)/g, '$1')
      .replace(/atomicStore\(&([a-zA-Z]+), ([^;\n]+)\);/g, '$1 = $2;');
    const values: Record<string, Float32Array | Uint32Array> = {
      vertices: Uint32Array.of(0, 1),
      transitions: Uint32Array.of(0, 0, 0, 1),
      costs: Float32Array.of(0, 0, 41, 41, 0, 0, 41, 41),
      endpoints: Uint32Array.of(0, 1),
      departure: Float32Array.of(1),
      elapsed: Float32Array.of(913, 913),
      arrivals: Float32Array.of(913, 913),
      predecessors: Uint32Array.of(913, 913),
      summary: new Uint32Array(4).fill(913)
    };
    const execute = (source: string) => {
      source = source
        .replace(/override INVOCATION_COUNT: u32;/, 'const INVOCATION_COUNT: u32 = 2u;')
        .replace(/override WGSL_KERNEL_DISPATCH_X: u32;/, 'const WGSL_KERNEL_DISPATCH_X: u32 = 1u;')
        .replace(/override WGSL_KERNEL_DISPATCH_Y: u32;/, 'const WGSL_KERNEL_DISPATCH_Y: u32 = 1u;')
        .replace(/^\s*_ = .*;$/gm, '');
      const bindings: Record<number, Float32Array | Uint32Array> = {};
      for (const match of source.matchAll(/@binding\((\d+)\)\s+var<storage,\s*\w+>\s+(\w+)/g)) {
        expect(values[match[2]], match[2]).toBeDefined();
        bindings[Number(match[1])] = values[match[2]];
      }
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: bindings});
    };
    execute(elapsedSource);
    execute(sources.get('network-time-routing-publish-arrivals')!);
    // Closed form A(t)=t cannot improve the initial self-loop, but must reach state 1 at zero duration.
    expect(values['elapsed']).toEqual(Float32Array.of(0, 0));
    expect(values['arrivals']).toEqual(Float32Array.of(1, 1));
    expect(values['predecessors']).toEqual(Uint32Array.of(0xffffffff, 1));
    expect(values['summary']).toEqual(Uint32Array.of(1, 1, 1, 0x3f800000));
  } finally {
    device.destroy();
  }
});
