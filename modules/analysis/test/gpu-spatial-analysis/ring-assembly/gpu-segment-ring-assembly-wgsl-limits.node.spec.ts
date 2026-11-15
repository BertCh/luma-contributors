// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('GPUSegmentRingAssembly kernels stay within 8 storage bindings', () => {
  for (const pairedOpposites of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: `ring-limits-${pairedOpposites}`
    });
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `ring-limit-${serial++}`, format, length);
    const nodes = new GPUSegmentRingAssembly({
      endpoints: view('float32x4', 16),
      count: view('uint32', 1),
      groups: view('uint32', 16),
      pairedOpposites,
      cancelOpposingSegments: true,
      output: {
        ringOffsets: view('uint32', 5),
        positions: view('float32x2', 32),
        ringShells: view('uint32', 4),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.length).toBeGreaterThan(10);
    for (const {id, source} of sources) {
      expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    }
    expect(sources.some(({id}) => id.includes('quantize-extent'))).toBe(true);
  }
});

it('shell indexing accepts the maximum isoband ring capacity within 128 MiB bindings', () => {
  const bindingLimit = 128 * 1024 * 1024;
  const ringCapacity = bindingLimit / 16;
  const device = createNullWebGPUDevice();
  device.limits.maxStorageBufferBindingSize = bindingLimit;
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `maximum-shell-${serial++}`, format, length);
  // Transient views record shapes without allocating the maximum-sized buffers on the CPU.
  const nodes = new GPUSegmentRingAssembly({
    id: 'maximum-shells',
    endpoints: view('float32x4', 16),
    output: {
      positions: view('float32x2', 32),
      ringOffsets: view('uint32', ringCapacity + 1),
      ringShells: view('uint32', ringCapacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
  const traversal = getWGSLKernelNodeRecord(graph, 'maximum-shells-ring-shells')!;
  expect(traversal.bindings).toHaveLength(8);
  const records = traversal.bindings.find(binding => binding.name === 'records')!.view;
  const bounds = traversal.bindings.find(binding => binding.name === 'ringBounds')!.view;
  expect(records.length * 4).toBe(96 * 1024 * 1024);
  expect(bounds.length * 4).toBe(bindingLimit);
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
});
