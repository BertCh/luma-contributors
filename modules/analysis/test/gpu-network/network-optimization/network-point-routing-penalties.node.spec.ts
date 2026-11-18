// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslParser} from 'wgsl_reflect';
import {GPUNetworkPointRouting} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {setupPointRouting} from './network-point-routing-scalability-test-utils';

for (const mode of ['reference', 'scalable'] as const) {
  it(`validates and parses penalty-enabled ${mode} kernels within eight storage bindings`, () => {
    const {graph, props} = setupPointRouting({
      edgeCount: 3,
      turnCount: 2,
      queryCount: 2,
      mode,
      maximumRelaxations: 4
    });
    const turnPenalties = createTransientView(graph, 'turn-penalties', 'float32', 2);
    const contributor = new GPUNetworkPointRouting({...props, turnPenalties});
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels) {
      expect(() => new WgslParser().parse(kernel.source)).not.toThrow();
      expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
      expect(kernel.source).not.toContain('undefined');
    }
    expect(
      () =>
        new GPUNetworkPointRouting({
          ...props,
          turnPenalties: createTransientView(graph, 'wrong-length', 'float32', 1)
        })
    ).toThrow(/turnPenalties/);
    expect(() => new GPUNetworkPointRouting({...props, turnPenalties: props.output.costs})).toThrow(
      /network-point-routing/
    );
    if (mode === 'reference') {
      const memory = createCPUGraphMemory();
      memory.write(props.costs, Float32Array.of(2, 2, 10));
      memory.write(props.turns, Uint32Array.of(0, 2, 1, 2));
      memory.write(turnPenalties, Float32Array.of(7, 1));
      memory.write(props.queries, Uint32Array.of(0, 2, 1, 2));
      memory.write(props.fractions, Float32Array.of(0.5, 0.25, 0.5, 0.25));
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      expect(memory.readFloat32(props.output.costs)).toEqual(Float32Array.of(10.5, 4.5));
      expect(memory.readUint32(props.output.status)).toEqual(Uint32Array.of(0, 0));
    }
  });
}
